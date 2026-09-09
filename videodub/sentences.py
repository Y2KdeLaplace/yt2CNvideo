"""Stable sentence identity and acoustic timing shared by text and speech stages."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .subtitles import Cue, SENTENCE_END_RE

MAX_SENTENCE_GAP_MS = 1000
MAX_SENTENCE_WINDOW_MS = 12000
MAX_OVERLAP_MS = 100
SOUND_RE = re.compile(r"\[[^\]]*\]|【[^】]*】")


def normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    return " ".join(text.split())


def spoken_text(text: str) -> str:
    text = normalized_text(SOUND_RE.sub(" ", normalized_text(text)))
    return text if any(unicodedata.category(c)[0] in "LN" for c in text) else ""


def text_kind(text: str) -> str:
    if spoken_text(text):
        return "spoken"
    return "sound" if SOUND_RE.search(normalized_text(text)) else "punctuation"


@dataclass(frozen=True)
class SentenceUnit:
    first_cue: int
    last_cue: int
    start_ms: int
    end_ms: int
    text: str
    group_id: int = 0
    kind: str = "spoken"


class TimelineError(ValueError):
    pass


def validate_timeline(cues: list[Cue], logger: Callable[[str], None] | None = None,
                      *, allow_punctuation: bool = False) -> None:
    errors: list[str] = []
    previous: Cue | None = None
    for cue in cues:
        issues: list[str] = []
        if cue.start_ms < 0 or cue.end_ms <= cue.start_ms:
            issues.append("invalid_duration")
        if previous is not None:
            if cue.start_ms < previous.start_ms or cue.end_ms < previous.end_ms:
                issues.append("non_monotonic_timeline")
            if previous.end_ms - cue.start_ms > MAX_OVERLAP_MS:
                issues.append("abnormal_overlap")
        kind = text_kind(cue.text)
        if cue.end_ms - cue.start_ms > MAX_SENTENCE_WINDOW_MS and kind == "spoken":
            issues.append("sentence_window_too_long")
        if kind == "punctuation":
            message = f'timeline QC cue {cue.index} text={cue.text!r} start={cue.start_ms} end={cue.end_ms}: punctuation_or_empty'
            if logger:
                logger(message)
            if not allow_punctuation:
                issues.append("punctuation_or_empty")
        if issues:
            message = f'timeline QC cue {cue.index} text={cue.text!r} start={cue.start_ms} end={cue.end_ms}: {", ".join(issues)}'
            errors.append(message)
            if logger:
                logger(message)
        previous = cue
    if errors:
        raise TimelineError("字幕时间轴不可靠，请检查或重新生成声学对齐；未修改时间戳。\n" + "\n".join(errors))


def _join_text(parts: list[Cue]) -> str:
    result = ""
    for cue in parts:
        text = " ".join(cue.text.split())
        if result and text and result[-1].isascii() and text[0].isascii() and text[0].isalnum():
            result += " "
        result += text
    return result


def build_sentence_units(cues: list[Cue], logger: Callable[[str], None] | None = None) -> list[SentenceUnit]:
    # Reject unreliable acoustic fragments instead of inventing their duration.
    validate_timeline(cues, logger, allow_punctuation=True)
    units: list[SentenceUnit] = []
    pending: list[Cue] = []

    def flush() -> None:
        if pending:
            units.append(SentenceUnit(pending[0].index, pending[-1].index,
                                      pending[0].start_ms, pending[-1].end_ms,
                                      _join_text(pending), len(units) + 1))
            pending.clear()

    for cue in cues:
        kind = text_kind(cue.text)
        if kind == "punctuation":
            # Preserve an adjacent sentence terminator, never its invented timing.
            if pending and cue.start_ms - pending[-1].end_ms <= MAX_SENTENCE_GAP_MS:
                previous = pending[-1]
                pending[-1] = Cue(previous.index, previous.start_ms, previous.end_ms,
                                  previous.text + cue.text.strip())
                if SENTENCE_END_RE.search(pending[-1].text):
                    flush()
            continue
        if kind == "sound":
            flush()
            units.append(SentenceUnit(cue.index, cue.index, cue.start_ms, cue.end_ms,
                                      cue.text, len(units) + 1, "sound"))
            continue
        if pending:
            gap = cue.start_ms - pending[-1].end_ms
            window = cue.end_ms - pending[0].start_ms
            if gap > MAX_SENTENCE_GAP_MS:
                if logger:
                    logger(f'timeline QC cue {cue.index} text={cue.text!r} start={cue.start_ms} end={cue.end_ms}: sentence_gap={gap}ms; 保留独立句界')
                flush()
            elif window > MAX_SENTENCE_WINDOW_MS:
                # Only an actual pause or punctuation can end an overlong group.
                if gap >= 250 or re.search(r"[,;:，；：]$", pending[-1].text):
                    flush()
                else:
                    raise TimelineError(f'timeline QC cue {pending[0].index}–{cue.index} text={_join_text(pending + [cue])!r} start={pending[0].start_ms} end={cue.end_ms}: sentence_window_too_long; 缺少可靠句界')
        pending.append(cue)
        if SENTENCE_END_RE.search(cue.text):
            flush()
    flush()
    validate_units(units)
    return units


def validate_units(units: list[SentenceUnit]) -> None:
    ids = [u.group_id for u in units]
    if ids != list(range(1, len(units) + 1)):
        raise ValueError("SentenceUnit group_id 必须唯一且连续")
    for unit in units:
        if any(type(value) is not int for value in (
            unit.group_id, unit.first_cue, unit.last_cue, unit.start_ms, unit.end_ms
        )) or not isinstance(unit.text, str):
            raise ValueError("SentenceUnit 身份、时间戳必须为整数，文本必须为字符串")
        if unit.first_cue < 1 or unit.last_cue < unit.first_cue:
            raise ValueError(f"SentenceUnit {unit.group_id} cue 范围无效")
        if unit.kind not in {"spoken", "sound"} or text_kind(unit.text) != unit.kind:
            raise ValueError(f"SentenceUnit {unit.group_id} 文本类型无效：{unit.text!r}")
    validate_timeline([Cue(u.first_cue, u.start_ms, u.end_ms, u.text) for u in units])


def display_cues(units: list[SentenceUnit]) -> list[Cue]:
    """Presentation boundary; no display-derived timing is fed back into speech."""
    validate_units(units)
    return [Cue(i, u.start_ms, u.end_ms, u.text) for i, u in enumerate(units, 1)]


def sentence_path(subtitle_path: Path) -> Path:
    return subtitle_path.with_suffix(".sentences.json")


def write_units(subtitle_path: Path, units: list[SentenceUnit]) -> None:
    validate_units(units)
    target = sentence_path(subtitle_path)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"version": 1, "units": [asdict(u) for u in units]}, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def read_units(subtitle_path: Path) -> list[SentenceUnit]:
    path = sentence_path(subtitle_path)
    if not path.is_file():
        raise RuntimeError(f"缺少句级数据：{path.name}。请重新执行句级校正/翻译，不能从显示 SRT 猜测配音句界。")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError(f"不支持的句级数据版本：{path}")
    units = [SentenceUnit(**item) for item in data["units"]]
    validate_units(units)
    return units
