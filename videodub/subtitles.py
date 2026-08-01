from __future__ import annotations

import html
import json
import re
from bisect import bisect_left
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


TIMESTAMP_RE = re.compile(
    r"(?P<sh>\d{1,3}):(?P<sm>\d{2}):(?P<ss>\d{2})[,.](?P<sms>\d{3})"
    r"\s*-->\s*"
    r"(?P<eh>\d{1,3}):(?P<em>\d{2}):(?P<es>\d{2})[,.](?P<ems>\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
SENTENCE_END_RE = re.compile(r"[.!?。！？][\"'”’》〉】』〕〗〙〛）)\]]*$")


@dataclass
class Cue:
    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return max(1, self.end_ms - self.start_ms)


def _to_ms(hours: str, minutes: str, seconds: str, milliseconds: str) -> int:
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(milliseconds)
    )


def _clean_text(text: str) -> str:
    text = html.unescape(TAG_RE.sub("", text))
    text = text.replace("\ufeff", "").replace("\u200b", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def parse_srt_text(content: str) -> list[Cue]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    cues: list[Cue] = []
    for block in re.split(r"\n{2,}", normalized):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        timestamp_pos = next(
            (i for i, line in enumerate(lines) if TIMESTAMP_RE.search(line)),
            None,
        )
        if timestamp_pos is None:
            continue
        match = TIMESTAMP_RE.search(lines[timestamp_pos])
        assert match is not None
        text = _clean_text("\n".join(lines[timestamp_pos + 1 :]))
        if not text:
            continue
        cues.append(
            Cue(
                index=len(cues) + 1,
                start_ms=_to_ms(
                    match["sh"], match["sm"], match["ss"], match["sms"]
                ),
                end_ms=_to_ms(match["eh"], match["em"], match["es"], match["ems"]),
                text=text,
            )
        )
    return cues


def read_srt(path: str | Path) -> list[Cue]:
    path = Path(path)
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return parse_srt_text(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别字幕文件编码：{path}")


def _timestamp(milliseconds: int) -> str:
    milliseconds = max(0, int(milliseconds))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def write_srt(path: str | Path, cues: list[Cue]) -> None:
    blocks = []
    for number, cue in enumerate(cues, 1):
        blocks.append(
            f"{number}\n{_timestamp(cue.start_ms)} --> {_timestamp(cue.end_ms)}\n"
            f"{cue.text.strip()}"
        )
    Path(path).write_text("\n\n".join(blocks) + "\n", encoding="utf-8-sig")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _join_tokens(tokens: list[str]) -> str:
    text = ""
    no_space_before = set(",.;:!?%、。，！？；：)]}〉》】』〕〗〙〛")
    no_space_after = set("([{〈《【「〔〖〘〚")
    for token in tokens:
        if not text:
            text = token
        elif token[0] in no_space_before or text[-1] in no_space_after:
            text += token
        elif "\u4e00" <= token[0] <= "\u9fff" and "\u4e00" <= text[-1] <= "\u9fff":
            text += token
        else:
            text += " " + token
    return text.strip()


def subtitle_transcript(cues: list[Cue]) -> str:
    """Return subtitle text without IDs/timestamps or rolling-caption repeats."""
    transcript_tokens: list[str] = []
    for cue in cues:
        current = _tokens(cue.text)
        normalized = [token.casefold() for token in current]
        existing = [token.casefold() for token in transcript_tokens]
        overlap = 0
        for size in range(min(len(existing), len(normalized), 80), 0, -1):
            if existing[-size:] == normalized[:size]:
                overlap = size
                break
        transcript_tokens.extend(current[overlap:])
    return _join_tokens(transcript_tokens)


def align_transcript_to_cues(source_cues: list[Cue], transcript: str) -> list[Cue]:
    """Map a corrected same-language transcript onto an existing SRT timeline."""
    source_tokens: list[str] = []
    boundaries: list[int] = []
    for cue in source_cues:
        current = _tokens(cue.text)
        normalized = [token.casefold() for token in current]
        existing = [token.casefold() for token in source_tokens]
        overlap = 0
        for size in range(min(len(existing), len(normalized), 80), 0, -1):
            if existing[-size:] == normalized[:size]:
                overlap = size
                break
        source_tokens.extend(current[overlap:])
        boundaries.append(len(source_tokens))
    target_tokens = _tokens(transcript)
    if not source_tokens or not target_tokens:
        return []

    matcher = SequenceMatcher(
        None,
        [token.casefold() for token in source_tokens],
        [token.casefold() for token in target_tokens],
        autojunk=False,
    )
    anchor_map = {0: 0, len(source_tokens): len(target_tokens)}
    for match in matcher.get_matching_blocks():
        if match.size:
            anchor_map[match.a] = match.b
            anchor_map[match.a + match.size] = match.b + match.size
    anchor_source = sorted(anchor_map)

    def target_boundary(source_boundary: int) -> int:
        position = bisect_left(anchor_source, source_boundary)
        if position < len(anchor_source) and anchor_source[position] == source_boundary:
            return anchor_map[source_boundary]
        left = anchor_source[max(0, position - 1)]
        right = anchor_source[min(position, len(anchor_source) - 1)]
        if left == right:
            return anchor_map[left]
        ratio = (source_boundary - left) / (right - left)
        return round(anchor_map[left] + ratio * (anchor_map[right] - anchor_map[left]))

    mapped = [0]
    for boundary in boundaries:
        mapped.append(max(mapped[-1], min(len(target_tokens), target_boundary(boundary))))
    mapped[-1] = len(target_tokens)

    aligned: list[Cue] = []
    for cue, start, end in zip(source_cues, mapped[:-1], mapped[1:], strict=True):
        text = _join_tokens(target_tokens[start:end])
        if text:
            aligned.append(Cue(cue.index, cue.start_ms, cue.end_ms, text))
    return aligned


def semantic_cues(
    cues: list[Cue],
    *,
    max_duration_ms: int = 12_000,
    max_characters: int = 240,
) -> list[Cue]:
    """Join subtitle fragments into sentence-like units while retaining time ranges."""
    grouped: list[Cue] = []
    pending: list[Cue] = []
    for cue in cues:
        pending.append(cue)
        text = " ".join(item.text.strip() for item in pending).strip()
        duration = pending[-1].end_ms - pending[0].start_ms
        if (
            SENTENCE_END_RE.search(text)
            or duration >= max_duration_ms
            or len(text) >= max_characters
        ):
            grouped.append(
                Cue(len(grouped) + 1, pending[0].start_ms, pending[-1].end_ms, text)
            )
            pending = []
    if pending:
        grouped.append(
            Cue(
                len(grouped) + 1,
                pending[0].start_ms,
                pending[-1].end_ms,
                " ".join(item.text.strip() for item in pending).strip(),
            )
        )
    return grouped


def extract_json_array(text: str) -> list[dict]:
    fenced = JSON_FENCE_RE.search(text)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("模型返回内容中没有 JSON 数组")
        parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("模型返回的 JSON 顶层不是数组")
    return parsed


def find_source_subtitle(video_path: Path) -> Path | None:
    stem = video_path.stem
    candidates = [
        item
        for item in video_path.parent.iterdir()
        if item.is_file()
        and item.suffix.lower() == ".srt"
        and item.stem.startswith(stem)
        and not item.name.endswith(
            (
                ".asr.srt",
                ".zh-CN.srt",
                ".corrected.srt",
                ".speech.zh-CN.srt",
            )
        )
    ]
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        if ".en-orig." in name:
            rank = 0
        elif re.search(r"\.en(?:[-_.]|$)", name):
            rank = 1
        elif not re.search(r"\.zh(?:[-_.]|$)", name):
            rank = 2
        else:
            rank = 3
        return rank, name

    return sorted(candidates, key=score)[0]
