from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig
from .media import VideoJob
from .openai_compatible import ChatResult, OpenAICompatibleClient
from .runner import ProcessRunner
from .subtitles import (
    Cue,
    align_transcript_to_cues,
    extract_json_array,
    find_source_subtitle,
    read_srt,
    semantic_cues,
    subtitle_transcript,
    write_srt,
)


DOMAIN_SYSTEM = """你是视频内容与专业术语分析专家。
输入包含视频信息、YouTube 文稿样本和 Qwen3-ASR 文稿样本。
判断主题和专业领域，整理专有名词、缩写、符号与公式的规范写法。
只返回 JSON 对象，不要使用 Markdown，不要臆造原内容中没有依据的信息。"""

TRANSCRIPT_REPAIR_SYSTEM = """你是严格的视频文稿校对专家。
输入是同一段视频的 YouTube 文稿和 Qwen3-ASR 文稿，两者都已移除字幕 id 和时间戳。
请把两份文稿合并为一份连贯、完整、原语言的校正文稿。两个来源互相印证，不要把 ASR 当作绝对正确。
结合上下文、视频主题和术语表修正错词、专有名词、断裂句子和明显漏字，但不得翻译、扩写或改变原意。
数学表达可以使用清晰的 LaTeX 行内公式。
只返回 JSON 对象：{"corrected_transcript": "..."}，不要使用 Markdown。"""

TRANSLATION_SYSTEM = """你是专业的视频字幕译者。
把已校正的连续文稿翻译为 {target_language}。输入已按完整句子分组，id 只是时间轴锚点，不是要求逐字对应。
在看完整个批次的上下文后再翻译，优先保证语义准确、表达流畅、术语一致和适合字幕阅读。
保留每个句子组的 id，不遗漏。只返回 JSON 数组，每项格式为 {"id": 1, "translated_text": "..."}，不要使用 Markdown。"""

SPEECH_REWRITE_SYSTEM = """你是 {language} 讲解稿改写专家。
把字幕逐条改写成适合 {language} 语音朗读、同时不改变信息内容的文本。
把公式、数学符号、英文缩写和无法直接朗读的写法改成自然口语。例如公式应表达其运算关系，而不是逐字符念出。
不要增加解释、总结或过渡句；保持每个 id 独立，不合并、不拆分、不遗漏。
只返回 JSON 数组，每项格式为 {"id": 1, "speech_text": "..."}。"""


@dataclass(frozen=True)
class RepairRecord:
    cue_id: int
    youtube_text: str
    asr_text: str
    corrected_text: str
    changed: bool


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if "```" in candidate:
        first_newline = candidate.find("\n", candidate.find("```"))
        end = candidate.rfind("```")
        if first_newline >= 0 and end > first_newline:
            candidate = candidate[first_newline + 1 : end].strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型返回内容中没有 JSON 对象")
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型返回的 JSON 顶层不是对象")
    return value


def _cue_payload(cues: list[Cue]) -> list[dict[str, Any]]:
    return [
        {
            "id": cue.index,
            "start": round(cue.start_ms / 1000, 3),
            "end": round(cue.end_ms / 1000, 3),
            "text": cue.text,
        }
        for cue in cues
    ]


def _sample_cues(cues: list[Cue], limit: int = 80) -> list[Cue]:
    if len(cues) <= limit:
        return cues
    selected = set(range(min(25, len(cues))))
    selected.update(range(max(0, len(cues) - 10), len(cues)))
    remaining = limit - len(selected)
    if remaining > 0:
        step = (len(cues) - 1) / max(1, remaining - 1)
        selected.update(round(i * step) for i in range(remaining))
    return [cues[index] for index in sorted(selected)[:limit]]


def _read_metadata(job: VideoJob) -> dict[str, str]:
    result = {"title": job.title, "description": ""}
    if not job.info_path:
        return result
    try:
        data = json.loads(job.info_path.read_text(encoding="utf-8"))
        result["title"] = str(data.get("title") or result["title"])
        result["description"] = str(data.get("description") or "")[:4000]
    except (OSError, ValueError):
        pass
    return result


class _TextWorkflow:
    def __init__(
        self,
        config: AppConfig,
        runner: ProcessRunner,
        *,
        api_key: str = "",
    ):
        self.config = config
        self.runner = runner
        self.client = OpenAICompatibleClient(
            config.subtitle_api_base_url,
            config.subtitle_model,
            api_key,
        )
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _record_usage(self, result: ChatResult) -> None:
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens

    def _call_object(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = 3000,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for _attempt in range(2):
            result = self.client.chat(system, prompt, max_tokens=max_tokens)
            self._record_usage(result)
            try:
                return _extract_json_object(result.text)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                prompt += (
                    "\n\n上一次输出无法解析。请严格只返回有效 JSON 对象。"
                    f"解析错误：{exc}"
                )
        raise RuntimeError(f"模型连续返回无效 JSON：{last_error}")

    def _call_array(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = 6000,
    ) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        for _attempt in range(2):
            result = self.client.chat(system, prompt, max_tokens=max_tokens)
            self._record_usage(result)
            try:
                return extract_json_array(result.text)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                prompt += (
                    "\n\n上一次输出无法解析。请严格只返回有效 JSON 数组。"
                    f"解析错误：{exc}"
                )
        raise RuntimeError(f"模型连续返回无效 JSON：{last_error}")

    @staticmethod
    def _validated_texts(
        rows: list[dict[str, Any]],
        expected: list[Cue],
        field: str,
    ) -> dict[int, str]:
        expected_ids = {cue.index for cue in expected}
        result: dict[int, str] = {}
        for row in rows:
            try:
                cue_id = int(row["id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("模型输出包含无效字幕 id") from exc
            text = str(row.get(field) or "").strip()
            if cue_id not in expected_ids or cue_id in result or not text:
                raise ValueError(f"模型输出的字幕 {cue_id} 无效、重复或为空")
            result[cue_id] = text
        if set(result) != expected_ids:
            missing = sorted(expected_ids - set(result))
            raise ValueError(f"模型输出遗漏字幕 id：{missing[:20]}")
        return result


class SubtitleRepairWorkflow(_TextWorkflow):
    def analyze_domain(
        self,
        job: VideoJob,
        youtube_cues: list[Cue],
        asr_cues: list[Cue],
    ) -> dict[str, Any]:
        prompt = json.dumps(
            {
                "video": _read_metadata(job),
                "youtube_transcript_sample": subtitle_transcript(
                    _sample_cues(youtube_cues)
                ),
                "qwen_asr_transcript_sample": subtitle_transcript(
                    _sample_cues(asr_cues)
                ),
                "required_schema": {
                    "domain": "string",
                    "summary": "string",
                    "glossary": [
                        {
                            "term": "string",
                            "preferred_zh": "string",
                            "notes": "string",
                        }
                    ],
                },
            },
            ensure_ascii=False,
        )
        return self._call_object(DOMAIN_SYSTEM, prompt)

    def repair(
        self,
        youtube_cues: list[Cue],
        asr_cues: list[Cue],
        domain: dict[str, Any],
    ) -> tuple[list[Cue], list[RepairRecord]]:
        self.runner.check_cancelled()
        youtube_transcript = subtitle_transcript(youtube_cues)
        asr_transcript = subtitle_transcript(asr_cues)
        result = self._call_object(
            TRANSCRIPT_REPAIR_SYSTEM,
            json.dumps(
                {
                    "domain": domain,
                    "youtube_transcript": youtube_transcript,
                    "qwen_asr_transcript": asr_transcript,
                },
                ensure_ascii=False,
            ),
            max_tokens=12_000,
        )
        corrected_transcript = str(result.get("corrected_transcript") or "").strip()
        if not corrected_transcript:
            raise ValueError("模型返回的校正文稿为空")
        corrected = align_transcript_to_cues(youtube_cues, corrected_transcript)
        if not corrected:
            raise ValueError("校正文稿无法映射到 YouTube 字幕时间轴")
        asr_aligned = align_transcript_to_cues(youtube_cues, asr_transcript)
        corrected_by_id = {cue.index: cue.text for cue in corrected}
        asr_by_id = {cue.index: cue.text for cue in asr_aligned}
        records = [
            RepairRecord(
                cue.index,
                cue.text,
                asr_by_id.get(cue.index, ""),
                corrected_by_id.get(cue.index, ""),
                corrected_by_id.get(cue.index, "") != cue.text,
            )
            for cue in youtube_cues
        ]
        self.runner.logger(
            f"文稿清洗完成，已映射回 {len(corrected)} 个字幕时间段。"
        )
        return corrected, records

    def translate(
        self,
        cues: list[Cue],
        domain: dict[str, Any],
    ) -> list[Cue]:
        translated: list[Cue] = []
        units = semantic_cues(cues)
        batch_size = self.config.subtitle_translation_batch_size
        system = TRANSLATION_SYSTEM.replace(
            "{target_language}", self.config.translation_language
        )
        for start in range(0, len(units), batch_size):
            self.runner.check_cancelled()
            batch = units[start : start + batch_size]
            prompt = json.dumps(
                {
                    "domain": domain,
                    "target_language": self.config.translation_language,
                    "transcript_sentences": [
                        {"id": cue.index, "text": cue.text} for cue in batch
                    ],
                },
                ensure_ascii=False,
            )
            result = self._call_array(system, prompt)
            texts = self._validated_texts(result, batch, "translated_text")
            translated.extend(
                Cue(cue.index, cue.start_ms, cue.end_ms, texts[cue.index])
                for cue in batch
            )
            self.runner.logger(
                f"{self.config.translation_language} 翻译进度："
                f"{min(start + len(batch), len(units))}/{len(units)}"
            )
        return translated

    def process_job(
        self,
        job: VideoJob,
        *,
        repair: bool = True,
        translate: bool = True,
    ) -> dict[str, Any]:
        if not repair and not translate:
            return {}
        source = find_source_subtitle(job.video_path)
        if source is None:
            raise RuntimeError(f"缺少 YouTube 字幕：{job.video_path.name}")
        if repair and not job.asr_subtitle_path.is_file():
            raise RuntimeError(f"缺少 Qwen3-ASR 字幕：{job.asr_subtitle_path.name}")
        youtube_cues = read_srt(source)
        asr_cues = (
            read_srt(job.asr_subtitle_path)
            if job.asr_subtitle_path.is_file()
            else youtube_cues
        )
        if not youtube_cues or not asr_cues:
            raise RuntimeError(f"字幕为空：{job.video_path.name}")
        job.corrected_subtitle_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_before = self.prompt_tokens
        completion_before = self.completion_tokens
        self.runner.logger("正在判断主题并建立术语表…")
        domain = self.analyze_domain(job, youtube_cues, asr_cues)
        repairs: list[RepairRecord] = []
        if repair:
            self.runner.logger("正在合并 YouTube 与 Qwen3-ASR 文稿…")
            corrected, repairs = self.repair(youtube_cues, asr_cues, domain)
            write_srt(job.corrected_subtitle_path, corrected)
        elif job.corrected_subtitle_path.is_file():
            corrected = read_srt(job.corrected_subtitle_path)
        else:
            corrected = youtube_cues
        if translate:
            chinese = self.translate(corrected, domain)
            write_srt(job.chinese_subtitle_path, chinese)
        report = {
            "version": 3,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "video": str(job.video_path),
            "youtube_subtitle": str(source),
            "asr_subtitle": str(job.asr_subtitle_path),
            "corrected_subtitle": str(job.corrected_subtitle_path),
            "chinese_subtitle": str(job.chinese_subtitle_path),
            "domain": domain,
            "repairs": [asdict(item) for item in repairs],
            "usage": {
                "prompt_tokens": self.prompt_tokens - prompt_before,
                "completion_tokens": self.completion_tokens - completion_before,
            },
            "model": self.config.subtitle_model,
            "translation_language": self.config.translation_language,
        }
        report_path = job.generated_base_path.with_name(
            job.generated_base_path.name + ".subtitle-report.json"
        )
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if repair:
            self.runner.logger(f"修复字幕：{job.corrected_subtitle_path}")
        if translate:
            self.runner.logger(f"翻译字幕：{job.chinese_subtitle_path}")
        return report


class SpeechSubtitleWorkflow(_TextWorkflow):
    def process_job(self, job: VideoJob) -> Path:
        source = job.chinese_subtitle_path
        if not source.is_file():
            raise RuntimeError(f"缺少翻译字幕：{source.name}")
        cues = read_srt(source)
        if not cues:
            raise RuntimeError(f"翻译字幕为空：{source}")
        spoken: list[Cue] = []
        batch_size = self.config.subtitle_translation_batch_size
        for start in range(0, len(cues), batch_size):
            self.runner.check_cancelled()
            batch = cues[start : start + batch_size]
            result = self._call_array(
                SPEECH_REWRITE_SYSTEM.replace(
                    "{language}", self.config.translation_language
                ),
                json.dumps({"subtitles": _cue_payload(batch)}, ensure_ascii=False),
            )
            texts = self._validated_texts(result, batch, "speech_text")
            spoken.extend(
                Cue(cue.index, cue.start_ms, cue.end_ms, texts[cue.index])
                for cue in batch
            )
            self.runner.logger(
                f"配音文本改写进度：{min(start + len(batch), len(cues))}/{len(cues)}"
            )
        write_srt(job.speech_subtitle_path, spoken)
        self.runner.logger(f"临时配音字幕：{job.speech_subtitle_path}")
        return job.speech_subtitle_path
