from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..media import VideoJob
from ..sentences import SentenceUnit, build_sentence_units
from ..subtitles import Cue, subtitle_transcript


TRANSCRIPT_REPAIR_SYSTEM = """你是严格的视频对白校对专家。
结合 domain、glossary、下载字幕和完整 ASR 上下文，逐个校正 source_sentence_groups。
group_id 是稳定句子身份：每个必须且只能返回一次，禁止新增、丢失、合并或移动句子内容到另一组。
只改文本，不能改变时间戳；不得翻译、扩写、摘要或改变原意，不省略重复和停顿。
以声学句子为边界，下载字幕仅为文本证据。使用正常大小写和自然标点，不继承滚动字幕排版。
保留声效标签，不把标签解释成对白。只返回 JSON 数组：
[{"group_id": 1, "corrected_text": "..."}]。"""


@dataclass(frozen=True)
class RepairRecord:
    group_id: int
    first_cue: int
    last_cue: int
    source_text: str
    corrected_text: str
    changed: bool


def repair_units(
    workflow: Any,
    youtube_cues: list[Cue],
    asr_cues: list[Cue] | None,
    domain: dict[str, Any],
    *,
    source_units: list[SentenceUnit] | None = None,
) -> tuple[list[SentenceUnit], list[RepairRecord]]:
    """Repair sentence text without changing its acoustic identity or timing."""
    units = source_units if source_units is not None else build_sentence_units(
        asr_cues or youtube_cues,
        workflow.runner.logger,
    )
    corrected = workflow._transform_units(
        units,
        domain,
        TRANSCRIPT_REPAIR_SYSTEM,
        "corrected_text",
        {
            "downloaded_subtitle_transcript": subtitle_transcript(youtube_cues),
            "full_context_transcript": "\n".join(unit.text for unit in units),
        },
    )
    records = [
        RepairRecord(
            source.group_id,
            source.first_cue,
            source.last_cue,
            source.text,
            corrected_unit.text,
            corrected_unit.text != source.text,
        )
        for source, corrected_unit in zip(units, corrected, strict=True)
    ]
    workflow.runner.logger(
        f"句级校正完成：{len(corrected)} 句，身份和声学时间戳保持不变。"
    )
    return corrected, records


def run_repair_stage(workflow: Any, job: VideoJob) -> dict[str, Any]:
    """Run stage 2 only."""
    return workflow.process_job(job, repair=True, translate=False)
