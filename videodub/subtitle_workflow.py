from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig
from .media import VideoJob
from .openai_compatible import ChatResult, OpenAICompatibleClient
from .runner import CancelledError, ProcessRunner
from .subtitles import (
    Cue,
    SENTENCE_END_RE,
    align_transcript_to_cues,
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

TRANSCRIPT_REPAIR_SYSTEM = """你是严格的视频文稿校对专家。
输入至少包含下载字幕文稿，并可能包含同一内容的 Qwen3-ASR 文稿；所有文稿都已移除字幕 id 和时间戳。
如果有两份文稿，请把它们合并为一份连贯、完整、原语言的校正文稿，两者在内容上互相印证，不要把任何一个来源当作绝对正确。如果没有 ASR 文稿，只校正下载字幕，不得臆造另一份文稿的内容。
结合上下文、视频主题和术语表修正错词、专有名词、断裂句子和明显漏字，但不得翻译、扩写或改变原意。
有 ASR 时，最终时间分段由系统统一沿用 Qwen3-ASR 文稿；文本排版与书写风格也以 Qwen3-ASR 文稿为准，包括大小写、标点和句子格式。没有 ASR 时，系统沿用下载字幕的时间轴，并使用自然、可读的原语言书写格式。
YouTube 文稿只作为内容校对证据：即使它使用全大写、滚动字幕或不同断句，也不要继承这些格式。若 Qwen3-ASR 使用正常的句首大写，校正文稿也应保持正常大小写。
数学表达保持原语言中清晰、可读的写法。
只返回 JSON 对象：{"corrected_transcript": "..."}，不要使用 Markdown。"""

TRANSLATION_SYSTEM = """你是专业的视频文稿译者和校对者。
把无时间戳的完整校正文稿翻译为 {target_language}。先理解全文上下文，再逐个翻译 source_sentence_groups；这些句组按顺序拼接就是完整文稿，此阶段不做字幕分段或定时。
不得概括成讲义或摘要，不得省略推理、例子、剧情信息及有意义的重复；只可删去不承载含义的纯口头填充词。术语、变量、单位和符号前后一致。
针对科普、课程或技术内容，准确保留概念关系和推导。遇到数学公式、概率、函数或计算的口语表达时，可以直接整理成紧凑的 ASCII 公式，不要使用 LaTeX。公式本身保持 ASCII，解释文字使用目标语言。例如英文“one minus e to the negative r of t k times delta”中的公式应写成“1 - exp(-r(t_k)*delta)”。
针对电影、剧集或生活对白，优先保留人物口吻、称谓、关系、情绪、潜台词、幽默和语境；使用目标语言中的自然口语，不要改写成科普讲解或书面总结，也不要无故弱化对剧情有作用的粗口、停顿或重复。
目标语言要求：{language_guidance}
每个输入 group_id 必须且只能返回一次，不得遗漏、合并或新增 group_id。只返回 JSON 数组，每项格式为 {"group_id": 1, "translated_text": "..."}，不要使用 Markdown。"""

SUBTITLE_SEGMENTATION_SYSTEM = """你是字幕语义分段专家。
输入只包含一个小批次的原语言校正字幕，以及这些完整句组已经完成的 {target_language} 译文。视频画面不可用，也不应作为判断依据。
输入的 segmentation_groups 已按顺序排列；每组包含完整译文和按时间顺序排列的 source_parts。你的任务只是把该组译文分配到等长的 parts 数组；不得重新翻译、改写、增删或调换译文内容。不要生成任何字幕 id、句组 id 或时间戳。
输出数组必须与 segmentation_groups 等长；每项的 parts 必须与对应 source_parts 等长。若相邻原文窗口在目标语言中应当合并，把完整译文放在其中一个位置，其他被合并位置返回空字符串。每组至少有一个非空 part。
结合整句语义、目标语言语序和每个 source_part 的 duration_ms 自然分配译文。根据目标语言的自然语法边界、标点和书写系统分割文本，不要把中文规则机械套用到其他语言。
目标语言要求：{language_guidance}
每组所有非空 parts 按顺序拼接、忽略空白后，必须与该组 translated_text 完全一致。
只返回 JSON 数组，每项格式为 {"parts": ["第一段译文", "", "第三段译文"]}。不要使用 Markdown。"""

@dataclass(frozen=True)
class RepairRecord:
    cue_id: int
    youtube_text: str
    asr_text: str
    corrected_text: str
    changed: bool


def _language_guidance(language: str) -> str:
    if language == "Chinese":
        return "使用自然简体中文、中文语序和中文标点；字幕宜短而完整，公式保留 ASCII。"
    if language in {"Japanese", "Korean"}:
        return (
            f"使用自然的 {language} 语序、敬语层级、称谓和本语言标点；"
            "不要按空格机械断句，公式保留 ASCII。"
        )
    return (
        f"遵循自然、规范的 {language} 语法、大小写、词间空格、标点和称谓；"
        "根据人物关系保持正式或口语语域，公式保留 ASCII。"
    )


def _without_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text)


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


def _sentence_group_payload(
    cues: list[Cue],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    groups: list[dict[str, Any]] = []
    cue_groups: dict[int, int] = {}
    pending: list[Cue] = []
    for cue in cues:
        pending.append(cue)
        standalone_sound = bool(
            re.fullmatch(r"\s*[\[(（【].+[\])）】]\s*", cue.text)
        )
        if not SENTENCE_END_RE.search(cue.text) and not standalone_sound:
            continue
        group_id = len(groups) + 1
        ids = [item.index for item in pending]
        groups.append(
            {
                "group_id": group_id,
                "ids": ids,
                "source_text": " ".join(item.text for item in pending),
            }
        )
        cue_groups.update((cue_id, group_id) for cue_id in ids)
        pending = []
    if pending:
        group_id = len(groups) + 1
        ids = [item.index for item in pending]
        groups.append(
            {
                "group_id": group_id,
                "ids": ids,
                "source_text": " ".join(item.text for item in pending),
            }
        )
        cue_groups.update((cue_id, group_id) for cue_id in ids)
    return groups, cue_groups


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
        *,
        allow_missing: bool = False,
    ) -> dict[int, str]:
        expected_ids = {cue.index for cue in expected}
        result: dict[int, str] = {}
        conflicting_ids: set[int] = set()
        for row in rows:
            try:
                cue_id = int(row["id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("模型输出包含无效字幕 id") from exc
            if cue_id not in expected_ids:
                raise ValueError(f"模型输出的字幕 {cue_id} 无效、重复或为空")
            text = str(row.get(field) or "").strip()
            if not text:
                if allow_missing:
                    continue
                raise ValueError(f"模型输出的字幕 {cue_id} 无效、重复或为空")
            if cue_id in result:
                if allow_missing:
                    if result[cue_id] != text:
                        result.pop(cue_id)
                        conflicting_ids.add(cue_id)
                    continue
                raise ValueError(f"模型输出的字幕 {cue_id} 无效、重复或为空")
            if cue_id not in conflicting_ids:
                result[cue_id] = text
        if not allow_missing and set(result) != expected_ids:
            missing = sorted(expected_ids - set(result))
            raise ValueError(f"模型输出遗漏字幕 id：{missing[:20]}")
        return result

    @staticmethod
    def _validated_group_texts(
        rows: list[dict[str, Any]],
        expected_ids: set[int],
        *,
        allow_missing: bool = False,
    ) -> dict[int, str]:
        result: dict[int, str] = {}
        for row in rows:
            try:
                group_id = int(row["group_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("模型输出包含无效句组 id") from exc
            text = str(row.get("translated_text") or "").strip()
            if group_id not in expected_ids or not text or group_id in result:
                raise ValueError(f"模型输出的句组 {group_id} 无效、重复或为空")
            result[group_id] = text
        if not allow_missing and set(result) != expected_ids:
            missing = sorted(expected_ids - set(result))
            raise ValueError(f"模型输出遗漏句组 id：{missing[:20]}")
        return result


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

    def repair(
        self,
        youtube_cues: list[Cue],
        asr_cues: list[Cue] | None,
        domain: dict[str, Any],
    ) -> tuple[list[Cue], list[RepairRecord]]:
        self.runner.check_cancelled()
        youtube_transcript = subtitle_transcript(youtube_cues)
        asr_transcript = subtitle_transcript(asr_cues) if asr_cues else ""
        repair_input: dict[str, Any] = {
            "domain": domain,
            "downloaded_subtitle_transcript": youtube_transcript,
        }
        if asr_transcript:
            repair_input["qwen_asr_transcript"] = asr_transcript
        result = self._call_object(
            TRANSCRIPT_REPAIR_SYSTEM,
            json.dumps(repair_input, ensure_ascii=False),
            max_tokens=12_000,
        )
        corrected_transcript = str(result.get("corrected_transcript") or "").strip()
        if not corrected_transcript:
            raise ValueError("模型返回的校正文稿为空")
        timeline_cues = asr_cues or youtube_cues
        corrected = align_transcript_to_cues(
            timeline_cues,
            corrected_transcript,
            merge_empty_cues=True,
        )
        if not corrected:
            raise ValueError("校正文稿无法映射到字幕时间轴")
        youtube_aligned = align_transcript_to_cues(timeline_cues, youtube_transcript)
        corrected_by_id = {cue.index: cue.text for cue in corrected}
        youtube_by_id = {cue.index: cue.text for cue in youtube_aligned}
        records = [
            RepairRecord(
                cue.index,
                youtube_by_id.get(cue.index, ""),
                cue.text if asr_cues else "",
                corrected_by_id.get(cue.index, ""),
                corrected_by_id.get(cue.index, "") != cue.text,
            )
            for cue in timeline_cues
        ]
        timeline_name = "Qwen3-ASR" if asr_cues else "下载字幕"
        self.runner.logger(f"文稿清洗完成，已沿用 {timeline_name}时间轴。")
        return corrected, records

    def translate(
        self,
        cues: list[Cue],
        domain: dict[str, Any],
        *,
        artifact_base: Path | None = None,
        transcript_path: Path | None = None,
    ) -> list[Cue]:
        self.runner.check_cancelled()
        corrected_transcript = subtitle_transcript(cues)
        if not corrected_transcript:
            raise ValueError("校正文稿为空")
        guidance = _language_guidance(self.config.translation_language)
        sentence_groups, _ = _sentence_group_payload(cues)
        system = TRANSLATION_SYSTEM.replace(
            "{target_language}", self.config.translation_language
        ).replace("{language_guidance}", guidance)
        segmentation_system = SUBTITLE_SEGMENTATION_SYSTEM.replace(
            "{target_language}", self.config.translation_language
        ).replace("{language_guidance}", guidance)
        translated_transcript = ""
        rows: list[dict[str, Any]] = []
        try:
            group_texts: dict[int, str] = {}
            all_group_ids = {int(group["group_id"]) for group in sentence_groups}
            for attempt in range(3):
                missing_ids = all_group_ids - set(group_texts)
                if not missing_ids:
                    break
                requested_groups = [
                    group
                    for group in sentence_groups
                    if int(group["group_id"]) in missing_ids
                ]
                translation_request: dict[str, Any] = {
                    "domain": domain,
                    "target_language": self.config.translation_language,
                    "corrected_transcript": corrected_transcript,
                    "source_sentence_groups": requested_groups,
                }
                if attempt:
                    translation_request["retry_instruction"] = (
                        "只补译 source_sentence_groups 中尚未返回的句组；"
                        "严格保留 group_id，不要返回其他句组。"
                    )
                translation_rows = self._call_array(
                    system,
                    json.dumps(translation_request, ensure_ascii=False),
                    max_tokens=16_000,
                )
                rows = translation_rows
                returned = self._validated_group_texts(
                    translation_rows,
                    all_group_ids,
                    allow_missing=True,
                )
                group_texts.update(
                    (group_id, text)
                    for group_id, text in returned.items()
                    if group_id in missing_ids
                )
                remaining = all_group_ids - set(group_texts)
                if remaining:
                    self.runner.logger(
                        f"模型遗漏 {len(remaining)} 个翻译句组，正在补译…"
                    )
            missing_groups = sorted(all_group_ids - set(group_texts))
            if missing_groups:
                raise ValueError(f"模型输出遗漏句组 id：{missing_groups[:20]}")

            translated_transcript = "".join(
                group_texts[int(group["group_id"])] for group in sentence_groups
            )
            self.runner.logger(
                f"{self.config.translation_language} 完整文稿翻译完成，"
                "正在分批生成字幕时间轴…"
            )

            cue_by_id = {cue.index: cue for cue in cues}
            texts: dict[int, str] = {}
            multi_cue_groups: list[dict[str, Any]] = []
            for group in sentence_groups:
                group_id = int(group["group_id"])
                ids = [int(value) for value in group["ids"]]
                if len(ids) == 1:
                    texts[ids[0]] = group_texts[group_id]
                else:
                    multi_cue_groups.append(group)

            batches: list[list[dict[str, Any]]] = []
            current_batch: list[dict[str, Any]] = []
            current_size = 0
            batch_limit = self.config.subtitle_translation_batch_size
            for group in multi_cue_groups:
                group_size = len(group["ids"])
                if current_batch and current_size + group_size > batch_limit:
                    batches.append(current_batch)
                    current_batch = []
                    current_size = 0
                current_batch.append(group)
                current_size += group_size
            if current_batch:
                batches.append(current_batch)

            rows = []
            segmentation_unavailable = False
            for batch_number, batch_groups in enumerate(batches, 1):
                segmentation_groups = [
                    {
                        "source_parts": [
                            {
                                "text": cue_by_id[int(cue_id)].text,
                                "duration_ms": cue_by_id[int(cue_id)].duration_ms,
                            }
                            for cue_id in group["ids"]
                        ],
                        "translated_text": group_texts[int(group["group_id"])],
                    }
                    for group in batch_groups
                ]
                segmentation_request: dict[str, Any] = {
                    "target_language": self.config.translation_language,
                    "segmentation_groups": segmentation_groups,
                }
                batch_rows: list[dict[str, Any]] = []
                batch_texts: dict[int, str] = {}
                if segmentation_unavailable:
                    batch_texts = {
                        int(group["ids"][0]): group_texts[int(group["group_id"])]
                        for group in batch_groups
                    }
                    texts.update(batch_texts)
                    self.runner.logger(
                        f"字幕分段批次 {batch_number} 已沿用完整句组时间窗。"
                    )
                    self.runner.logger(
                        f"字幕时间轴分段进度：{batch_number}/{len(batches)}"
                    )
                    continue
                for attempt in range(2):
                    try:
                        batch_rows = self._call_array(
                            segmentation_system,
                            json.dumps(segmentation_request, ensure_ascii=False),
                            max_tokens=8_000,
                        )
                        rows.extend(batch_rows)
                        if len(batch_rows) != len(batch_groups):
                            raise ValueError(
                                "模型输出的句组数量与输入不一致"
                            )
                        batch_texts = {}
                        for row, group in zip(
                            batch_rows,
                            batch_groups,
                            strict=True,
                        ):
                            group_id = int(group["group_id"])
                            raw_parts = row.get("parts")
                            if not isinstance(raw_parts, list) or len(
                                raw_parts
                            ) != len(group["ids"]):
                                raise ValueError(
                                    f"字幕句组 {group_id} 的 parts 数量无效"
                                )
                            parts = [str(part or "").strip() for part in raw_parts]
                            if not any(parts):
                                raise ValueError(
                                    f"字幕句组 {group_id} 没有返回任何文本"
                                )
                            joined = "".join(parts)
                            if _without_whitespace(joined) != _without_whitespace(
                                group_texts[group_id]
                            ):
                                raise ValueError(
                                    f"字幕句组 {group_id} 改动或遗漏了翻译文稿"
                                )
                            for cue_id, part in zip(
                                group["ids"],
                                parts,
                                strict=True,
                            ):
                                if part:
                                    batch_texts[int(cue_id)] = part
                        break
                    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                        if attempt:
                            self.runner.logger(
                                f"字幕分段批次 {batch_number} 连续校验失败，"
                                "已按完整句组自动合并时间窗；后续批次将直接沿用。"
                            )
                            segmentation_unavailable = True
                            batch_texts = {
                                int(group["ids"][0]): group_texts[
                                    int(group["group_id"])
                                ]
                                for group in batch_groups
                            }
                            break
                        self.runner.logger(
                            f"字幕分段批次 {batch_number} 校验失败，正在重试…"
                        )
                        segmentation_request["previous_validation_error"] = str(exc)
                        segmentation_request["retry_instruction"] = (
                            "重新返回与 segmentation_groups 等长的 JSON 数组；"
                            "每项只含 parts，parts 与 source_parts 等长。允许用空字符串"
                            "表示合并位置，但非空部分拼接后必须与 translated_text 完全一致。"
                        )
                texts.update(batch_texts)
                self.runner.logger(
                    f"字幕时间轴分段进度：{batch_number}/{len(batches)}"
                )

            translated: list[Cue] = []
            for group in sentence_groups:
                group_ids = [int(cue_id) for cue_id in group["ids"]]
                returned_ids = [cue_id for cue_id in group_ids if cue_id in texts]
                if not returned_ids:
                    raise ValueError(
                        f"字幕句组 {group['group_id']} 没有可写入的字幕"
                    )
                positions = {cue_id: index for index, cue_id in enumerate(group_ids)}
                for index, cue_id in enumerate(returned_ids):
                    first_position = 0 if index == 0 else positions[cue_id]
                    last_position = (
                        len(group_ids) - 1
                        if index + 1 == len(returned_ids)
                        else positions[returned_ids[index + 1]] - 1
                    )
                    first_cue = cue_by_id[group_ids[first_position]]
                    last_cue = cue_by_id[group_ids[last_position]]
                    translated.append(
                        Cue(
                            len(translated) + 1,
                            first_cue.start_ms,
                            last_cue.end_ms,
                            texts[cue_id],
                        )
                    )
            if transcript_path is not None:
                transcript_path.parent.mkdir(parents=True, exist_ok=True)
                transcript_path.write_text(translated_transcript, encoding="utf-8")
            return translated
        except CancelledError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            if artifact_base is None:
                raise
            transcript_path = artifact_base.with_name(
                artifact_base.name + ".translation-debug.txt"
            )
            draft_path = artifact_base.with_name(
                artifact_base.name + ".segmentation-debug.json"
            )
            transcript_path.write_text(translated_transcript, encoding="utf-8")
            draft_path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise RuntimeError(
                f"{exc}\n翻译文稿：{transcript_path}\n分段草稿：{draft_path}"
            ) from exc

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
        self.runner.logger("正在判断主题并建立术语表…")
        domain = self.analyze_domain(job, youtube_cues, asr_cues)
        repairs: list[RepairRecord] = []
        if repair:
            if asr_cues:
                self.runner.logger("正在合并下载字幕与 Qwen3-ASR 文稿…")
            else:
                self.runner.logger("没有视频或 ASR 字幕，正在校正下载字幕…")
            corrected, repairs = self.repair(youtube_cues, asr_cues, domain)
            write_srt(job.corrected_subtitle_path, corrected)
        elif job.corrected_subtitle_path.is_file():
            corrected = read_srt(job.corrected_subtitle_path)
        else:
            corrected = youtube_cues
        translated_path = job.translated_subtitle_path(
            self.config.translation_language
        )
        if translate:
            translated = self.translate(
                corrected,
                domain,
                artifact_base=job.base_path,
                transcript_path=job.translated_transcript_path(
                    self.config.translation_language
                ),
            )
            write_srt(translated_path, translated)
        report = {
            "version": 4,
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
