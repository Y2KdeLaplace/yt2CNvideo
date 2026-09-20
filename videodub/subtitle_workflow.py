from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig
from .media import VideoJob
from .openai_compatible import ChatResult, OpenAICompatibleClient
from .processing.repair import RepairRecord, repair_units
from .processing.translate import translate_units
from .runner import CancelledError, ProcessRunner
from .sentences import (SentenceUnit, build_sentence_units, display_cues, read_units,
                        write_units, validate_units, text_kind)
from .subtitles import (
    Cue,
    extract_json_array,
    find_source_subtitle,
    read_srt,
    subtitle_transcript,
    write_srt,
)


DOMAIN_SYSTEM = """你是视频内容与专业术语分析专家。
输入包含媒体信息、下载字幕文稿样本，并可能包含 Qwen3-ASR 文稿样本。
判断主题和专业领域，整理专有名词、缩写、符号与公式的规范写法。
只返回 JSON 对象，不要使用 Markdown，不要臆造原内容中没有依据的信息。"""

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
        result = self.client.chat(system, prompt, max_tokens=max_tokens)
        self._record_usage(result)
        return extract_json_array(result.text)


class SubtitleRepairWorkflow(_TextWorkflow):
    def analyze_domain(
        self,
        job: VideoJob,
        youtube_cues: list[Cue],
        asr_cues: list[Cue] | None,
    ) -> dict[str, Any]:
        samples: dict[str, Any] = {
            "media": _read_metadata(job),
            "downloaded_subtitle_transcript_sample": subtitle_transcript(
                _sample_cues(youtube_cues)
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
        }
        if asr_cues:
            samples["qwen_asr_transcript_sample"] = subtitle_transcript(
                _sample_cues(asr_cues)
            )
        prompt = json.dumps(
            samples,
            ensure_ascii=False,
        )
        return self._call_object(DOMAIN_SYSTEM, prompt)

    def _transform_units(
        self, units: list[SentenceUnit], domain: dict[str, Any], system: str,
        field: str, context: dict[str, Any],
    ) -> list[SentenceUnit]:
        validate_units(units)
        texts: dict[int, str] = {}
        # Full context, bounded outputs. Missing groups alone are retried.
        for first in range(0, len(units), 24):
            batch = units[first:first + 24]
            error = ""
            for attempt in range(3):
                pending = [u for u in batch if u.group_id not in texts]
                if not pending:
                    break
                self.runner.check_cancelled()
                request = {
                    **context, "domain": domain,
                    "source_sentence_groups": [
                        {"group_id": u.group_id, "text": u.text,
                         "start_ms": u.start_ms, "end_ms": u.end_ms,
                         "kind": u.kind} for u in pending
                    ],
                }
                if error:
                    request["previous_validation_error"] = error
                try:
                    rows = self._call_array(system, json.dumps(request, ensure_ascii=False), max_tokens=6000)
                    expected = {u.group_id: u for u in pending}
                    found: dict[int, str] = {}
                    for row in rows:
                        if not isinstance(row, dict) or type(row.get("group_id")) is not int:
                            raise ValueError("模型输出包含无效 group_id")
                        group_id = row["group_id"]
                        text = row.get(field)
                        if group_id not in expected or group_id in found:
                            raise ValueError(f"group_id {group_id} 重复或不在请求中")
                        if not isinstance(text, str) or text_kind(text) != expected[group_id].kind:
                            raise ValueError(f"group_id {group_id} 文本为空、纯标点或声效类型被改变")
                        found[group_id] = text.strip()
                    texts.update(found)
                    missing = set(expected) - set(found)
                    error = f"遗漏 group_id {sorted(missing)}" if missing else ""
                except CancelledError:
                    raise
                except (ValueError, RuntimeError) as exc:
                    error = str(exc)
                if error:
                    self.runner.logger(f"句级{field} 校验重试 {attempt + 1}/3：{error}")
            missing = [u.group_id for u in batch if u.group_id not in texts]
            if missing:
                raise RuntimeError(f"句级{field}失败，group_id={missing}：{error}")
        result = [replace(u, text=texts[u.group_id]) for u in units]
        validate_units(result)
        return result

    def repair(
        self, youtube_cues: list[Cue], asr_cues: list[Cue] | None,
        domain: dict[str, Any], *, source_units: list[SentenceUnit] | None = None,
    ) -> tuple[list[SentenceUnit], list[RepairRecord]]:
        return repair_units(
            self,
            youtube_cues,
            asr_cues,
            domain,
            source_units=source_units,
        )

    def translate(
        self, units: list[SentenceUnit], domain: dict[str, Any], *,
        transcript_path: Path | None = None,
    ) -> list[SentenceUnit]:
        return translate_units(
            self,
            units,
            domain,
            transcript_path=transcript_path,
        )

    def process_job(
        self,
        job: VideoJob,
        *,
        repair: bool = True,
        translate: bool = True,
    ) -> dict[str, Any]:
        if not repair and not translate:
            return {}
        source = job.source_subtitle_path or find_source_subtitle(job.video_path)
        if source is None:
            raise RuntimeError(f"缺少下载字幕：{job.video_path.name}")
        if repair and job.has_video and not job.asr_subtitle_path.is_file():
            raise RuntimeError(f"缺少 Qwen3-ASR 字幕：{job.asr_subtitle_path.name}")
        youtube_cues = read_srt(source)
        asr_cues = (
            read_srt(job.asr_subtitle_path)
            if job.has_video and job.asr_subtitle_path.is_file()
            else None
        )
        if not youtube_cues or (asr_cues is not None and not asr_cues):
            raise RuntimeError(f"字幕为空：{job.video_path.name}")
        job.corrected_subtitle_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_before = self.prompt_tokens
        completion_before = self.completion_tokens
        source_units = (
            read_units(job.corrected_subtitle_path)
            if not repair and job.corrected_subtitle_path.is_file()
            else build_sentence_units(asr_cues or youtube_cues, self.runner.logger)
        )
        self.runner.logger("正在判断主题并建立术语表…")
        domain = self.analyze_domain(job, youtube_cues, asr_cues)
        repairs: list[RepairRecord] = []
        if repair:
            if asr_cues:
                self.runner.logger("正在结合下载字幕逐句校正 Qwen3-ASR…")
            else:
                self.runner.logger("没有视频或 ASR 字幕，正在校正下载字幕…")
            corrected, repairs = self.repair(youtube_cues, asr_cues, domain, source_units=source_units)
            write_srt(job.corrected_subtitle_path, display_cues(corrected))
            write_units(job.corrected_subtitle_path, corrected)
        else:
            corrected = source_units
        translated_path = job.translated_subtitle_path(
            self.config.translation_language
        )
        if translate:
            translated = self.translate(
                corrected,
                domain,
                transcript_path=job.translated_transcript_path(
                    self.config.translation_language
                ),
            )
            write_srt(translated_path, display_cues(translated))
            write_units(translated_path, translated)
        report = {
            "version": 5,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "video": str(job.video_path) if job.has_video else "",
            "youtube_subtitle": str(source),
            "asr_subtitle": str(job.asr_subtitle_path) if asr_cues else "",
            "corrected_subtitle": str(job.corrected_subtitle_path),
            "translated_subtitle": str(translated_path),
            "chinese_subtitle": (
                str(translated_path)
                if self.config.translation_language == "Chinese"
                else ""
            ),
            "domain": domain,
            "repairs": [asdict(item) for item in repairs],
            "usage": {
                "prompt_tokens": self.prompt_tokens - prompt_before,
                "completion_tokens": self.completion_tokens - completion_before,
            },
            "model": self.config.subtitle_model,
            "translation_language": self.config.translation_language,
        }
        report_path = job.base_path.with_name(
            job.base_path.name + ".subtitle-report.json"
        )
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if repair:
            self.runner.logger(f"修复字幕：{job.corrected_subtitle_path}")
        if translate:
            self.runner.logger(f"翻译字幕：{translated_path}")
        return report
