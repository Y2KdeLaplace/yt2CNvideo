from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig
from .media import VideoJob
from .openai_compatible import ChatResult, OpenAICompatibleClient
from .runner import ProcessRunner
from .subtitles import Cue, extract_json_array, find_source_subtitle, read_srt, write_srt


DOMAIN_SYSTEM = """你是视频内容分析与术语规划专家。
根据标题、简介和字幕样本判断视频所属领域，并为后续英文字幕校正和简体中文翻译建立术语表。
只返回 JSON 对象，不要使用 Markdown。不要臆造字幕中没有依据的专有名词。"""

DETECTION_SYSTEM = """你是英文字幕质量审查员。输入来自 YouTube 字幕。
你的任务只是定位很可能存在语音识别错误、专有名词错误、明显语法断裂或上下文不通的字幕。
正常的口语、省略、重复、填充词和不完美断句不应标记。宁可少报，不要把风格优化当作错误。
只返回 JSON 数组，不要修复，不要输出未被要求的字幕。"""

REPAIR_SYSTEM = """你是谨慎的英文字幕校对专家。
结合领域信息、上下文和按字幕时间截取的视频画面，只修复有充分依据的识别错误。
截图可能包含幻灯片标题、公式、术语、人物或屏幕文字，也可能与口语无关。
不得扩写、润色、翻译或改变原意；证据不足时保持原文。
只返回 JSON 数组，不要使用 Markdown。"""

TRANSLATION_SYSTEM = """你是专业的简体中文字幕译者。
根据视频领域和术语表翻译英文字幕。保持每个 id 独立，不合并、不拆分、不遗漏。
译文要准确、自然、简洁，适合字幕显示；术语、人名、公式和缩写保持全片一致。
只返回 JSON 数组，不要使用 Markdown。"""


@dataclass(frozen=True)
class Suspect:
    cue_id: int
    confidence: float
    reason: str


@dataclass(frozen=True)
class EvidenceWindow:
    number: int
    cue_ids: tuple[int, ...]
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Repair:
    cue_id: int
    original: str
    corrected: str
    changed: bool
    confidence: float
    evidence: str


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if "```" in candidate:
        start = candidate.find("```")
        first_newline = candidate.find("\n", start)
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


def build_evidence_windows(
    cues: list[Cue],
    suspects: list[Suspect],
    *,
    margin_ms: int = 1500,
    merge_gap_ms: int = 2500,
    max_cues: int = 8,
) -> list[EvidenceWindow]:
    by_id = {cue.index: cue for cue in cues}
    selected = [by_id[item.cue_id] for item in suspects if item.cue_id in by_id]
    selected.sort(key=lambda cue: cue.start_ms)
    if not selected:
        return []
    groups: list[list[Cue]] = [[selected[0]]]
    for cue in selected[1:]:
        current = groups[-1]
        if (
            cue.start_ms - current[-1].end_ms <= merge_gap_ms
            and len(current) < max_cues
        ):
            current.append(cue)
        else:
            groups.append([cue])
    return [
        EvidenceWindow(
            number=number,
            cue_ids=tuple(cue.index for cue in group),
            start_ms=max(0, group[0].start_ms - margin_ms),
            end_ms=group[-1].end_ms + margin_ms,
        )
        for number, group in enumerate(groups, 1)
    ]


class SubtitleRepairWorkflow:
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
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(2):
            result = self.client.chat(system, prompt, max_tokens=3000)
            self._record_usage(result)
            try:
                return _extract_json_object(result.text)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                prompt += (
                    "\n\n上一次输出无法解析。请严格只返回一个有效 JSON 对象，"
                    f"不要添加 Markdown。解析错误：{exc}"
                )
        raise RuntimeError(f"模型连续返回无效 JSON：{last_error}")

    def _call_array(
        self,
        system: str,
        prompt: str,
        *,
        image_paths: list[Path] | None = None,
        max_tokens: int = 5000,
    ) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        for _attempt in range(2):
            result = self.client.chat(
                system,
                prompt,
                image_paths=image_paths,
                max_tokens=max_tokens,
            )
            self._record_usage(result)
            try:
                return extract_json_array(result.text)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                prompt += (
                    "\n\n上一次输出无法解析。请严格只返回有效 JSON 数组，"
                    f"不要添加 Markdown。解析错误：{exc}"
                )
        raise RuntimeError(f"模型连续返回无效 JSON：{last_error}")

    def analyze_domain(
        self,
        job: VideoJob,
        cues: list[Cue],
    ) -> dict[str, Any]:
        metadata = _read_metadata(job)
        prompt = (
            "请返回以下结构：\n"
            '{"domain":"主要领域","subdomains":["子领域"],'
            '"summary":"内容概述","glossary":['
            '{"term":"英文术语","preferred_zh":"建议中文","notes":"消歧说明"}]}\n\n'
            f"标题：{metadata['title']}\n"
            f"简介：{metadata['description']}\n"
            "字幕样本：\n"
            + json.dumps(_cue_payload(_sample_cues(cues)), ensure_ascii=False)
        )
        brief = self._call_object(DOMAIN_SYSTEM, prompt)
        brief.setdefault("domain", "未确定")
        brief.setdefault("subdomains", [])
        brief.setdefault("summary", "")
        brief.setdefault("glossary", [])
        if not isinstance(brief["glossary"], list):
            brief["glossary"] = []
        brief["glossary"] = brief["glossary"][:60]
        return brief

    def detect_suspects(
        self,
        cues: list[Cue],
        domain: dict[str, Any],
    ) -> list[Suspect]:
        size = self.config.subtitle_detection_batch_size
        radius = self.config.subtitle_context_radius
        suspects: dict[int, Suspect] = {}
        total = math.ceil(len(cues) / size)
        for batch_number, start in enumerate(range(0, len(cues), size), 1):
            self.runner.check_cancelled()
            targets = cues[start : start + size]
            context = cues[max(0, start - radius) : min(len(cues), start + size + radius)]
            target_ids = [cue.index for cue in targets]
            prompt = (
                '返回格式：[{"id":字幕ID,"confidence":0到1,'
                '"reason":"为什么很可能有错"}]。没有可疑项时返回 []。\n'
                f"只能返回这些目标 ID：{target_ids}\n"
                "领域信息："
                + json.dumps(domain, ensure_ascii=False)
                + "\n字幕（含少量只供理解的前后文）：\n"
                + json.dumps(_cue_payload(context), ensure_ascii=False)
            )
            items = self._call_array(
                DETECTION_SYSTEM,
                prompt,
                max_tokens=3000,
            )
            allowed = set(target_ids)
            for item in items:
                try:
                    cue_id = int(item["id"])
                    confidence = float(item.get("confidence", 0.5))
                    reason = str(item.get("reason", "")).strip()
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    cue_id in allowed
                    and confidence >= self.config.subtitle_suspect_threshold
                ):
                    previous = suspects.get(cue_id)
                    candidate = Suspect(cue_id, confidence, reason)
                    if previous is None or candidate.confidence > previous.confidence:
                        suspects[cue_id] = candidate
            self.runner.logger(f"可疑字幕筛查：{batch_number}/{total}")
        return [suspects[key] for key in sorted(suspects)]

    def _extract_screenshots(
        self,
        job: VideoJob,
        window: EvidenceWindow,
    ) -> list[Path]:
        evidence_root = job.video_path.parent / ".subtitle-evidence"
        folder = evidence_root / f"window-{window.number:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        count = self.config.subtitle_screenshot_count
        if count == 1:
            positions = [(window.start_ms + window.end_ms) / 2]
        else:
            span = max(1, window.end_ms - window.start_ms)
            positions = [
                window.start_ms + span * index / (count - 1)
                for index in range(count)
            ]
        paths: list[Path] = []
        for index, position_ms in enumerate(positions, 1):
            output = folder / f"frame-{index:02d}-{round(position_ms)}ms.jpg"
            if self.config.overwrite or not output.exists():
                self.runner.run(
                    [
                        self.config.ffmpeg_path,
                        "-y",
                        "-ss",
                        f"{position_ms / 1000:.3f}",
                        "-i",
                        job.video_path,
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale=1280:-2:force_original_aspect_ratio=decrease",
                        "-q:v",
                        "3",
                        output,
                    ],
                    quiet=True,
                )
            paths.append(output)
        return paths

    def repair_suspects(
        self,
        job: VideoJob,
        cues: list[Cue],
        domain: dict[str, Any],
        suspects: list[Suspect],
    ) -> tuple[list[Cue], list[Repair]]:
        by_id = {cue.index: cue for cue in cues}
        cue_position = {cue.index: index for index, cue in enumerate(cues)}
        repairs: dict[int, Repair] = {}
        windows = build_evidence_windows(cues, suspects)
        suspect_by_id = {item.cue_id: item for item in suspects}
        for window_number, window in enumerate(windows, 1):
            self.runner.check_cancelled()
            target_ids = list(window.cue_ids)
            positions = [cue_position[item] for item in target_ids]
            context = cues[
                max(0, min(positions) - self.config.subtitle_context_radius) :
                min(len(cues), max(positions) + self.config.subtitle_context_radius + 1)
            ]
            image_paths: list[Path] = []
            if self.config.subtitle_use_vision:
                image_paths = self._extract_screenshots(job, window)
            prompt = (
                '返回格式：[{"id":字幕ID,"corrected_text":"校正后的英文",'
                '"changed":true或false,"confidence":0到1,'
                '"evidence":"简短说明依据"}]。\n'
                f"必须且只能返回这些 ID：{target_ids}\n"
                f"截图按时间顺序取自 {window.start_ms / 1000:.3f}s 到 "
                f"{window.end_ms / 1000:.3f}s。\n"
                "领域和术语："
                + json.dumps(domain, ensure_ascii=False)
                + "\n筛查原因："
                + json.dumps(
                    [asdict(suspect_by_id[item]) for item in target_ids],
                    ensure_ascii=False,
                )
                + "\n字幕上下文："
                + json.dumps(_cue_payload(context), ensure_ascii=False)
            )
            try:
                items = self._call_array(
                    REPAIR_SYSTEM,
                    prompt,
                    image_paths=image_paths or None,
                    max_tokens=4000,
                )
            except RuntimeError as exc:
                if image_paths:
                    raise RuntimeError(
                        "截图辅助修复请求失败。请确认当前模型支持图片输入，"
                        "且兼容接口接受 image_url；"
                        "否则取消“图像使用”。\n"
                        f"原始错误：{exc}"
                    ) from exc
                raise
            returned: set[int] = set()
            for item in items:
                try:
                    cue_id = int(item["id"])
                    corrected = str(item["corrected_text"]).strip()
                except (KeyError, TypeError, ValueError):
                    continue
                if cue_id not in target_ids or not corrected:
                    continue
                original = by_id[cue_id].text
                changed = bool(item.get("changed", corrected != original))
                repair = Repair(
                    cue_id=cue_id,
                    original=original,
                    corrected=corrected if changed else original,
                    changed=changed and corrected != original,
                    confidence=float(item.get("confidence", 0.5)),
                    evidence=str(item.get("evidence", "")).strip(),
                )
                repairs[cue_id] = repair
                returned.add(cue_id)
            for cue_id in target_ids:
                if cue_id not in returned:
                    original = by_id[cue_id].text
                    repairs[cue_id] = Repair(
                        cue_id, original, original, False, 0.0, "模型未返回，保留原文"
                    )
            self.runner.logger(
                f"证据复核与英文修复：{window_number}/{len(windows)}"
            )
        corrected_cues = [
            Cue(
                cue.index,
                cue.start_ms,
                cue.end_ms,
                repairs[cue.index].corrected if cue.index in repairs else cue.text,
            )
            for cue in cues
        ]
        return corrected_cues, [repairs[key] for key in sorted(repairs)]

    def translate(
        self,
        cues: list[Cue],
        domain: dict[str, Any],
    ) -> list[Cue]:
        size = self.config.subtitle_translation_batch_size
        radius = self.config.subtitle_context_radius
        translated: dict[int, str] = {}
        total = math.ceil(len(cues) / size)
        compact_domain = {
            "domain": domain.get("domain"),
            "subdomains": domain.get("subdomains"),
            "summary": domain.get("summary"),
            "glossary": (domain.get("glossary") or [])[:60],
        }
        for batch_number, start in enumerate(range(0, len(cues), size), 1):
            self.runner.check_cancelled()
            targets = cues[start : start + size]
            context = cues[max(0, start - radius) : min(len(cues), start + size + radius)]
            target_ids = [cue.index for cue in targets]
            prompt = (
                '返回格式：[{"id":字幕ID,"zh":"简体中文译文"}]。\n'
                f"必须且只能返回这些目标 ID：{target_ids}\n"
                "领域和术语表："
                + json.dumps(compact_domain, ensure_ascii=False)
                + "\n字幕（额外前后文只供理解，不要翻译非目标 ID）：\n"
                + json.dumps(_cue_payload(context), ensure_ascii=False)
            )
            items = self._call_array(
                TRANSLATION_SYSTEM,
                prompt,
                max_tokens=6000,
            )
            for attempt in range(2):
                for item in items:
                    try:
                        cue_id = int(item["id"])
                        text = str(item["zh"]).strip()
                    except (KeyError, TypeError, ValueError):
                        continue
                    if cue_id in target_ids and text:
                        translated[cue_id] = text
                missing = [
                    cue_id for cue_id in target_ids if cue_id not in translated
                ]
                if not missing or attempt == 1:
                    break
                missing_set = set(missing)
                missing_cues = [
                    cue for cue in targets if cue.index in missing_set
                ]
                retry_prompt = (
                    '只补全上次遗漏的字幕。返回格式：[{"id":字幕ID,'
                    '"zh":"简体中文译文"}]。\n'
                    f"必须且只能返回这些 ID：{missing}\n"
                    "领域和术语表："
                    + json.dumps(compact_domain, ensure_ascii=False)
                    + "\n待补全字幕："
                    + json.dumps(_cue_payload(missing_cues), ensure_ascii=False)
                )
                items = self._call_array(
                    TRANSLATION_SYSTEM,
                    retry_prompt,
                    max_tokens=3000,
                )
            missing = [cue_id for cue_id in target_ids if cue_id not in translated]
            if missing:
                raise RuntimeError(
                    f"翻译模型重试后仍遗漏字幕 ID：{missing}"
                )
            self.runner.logger(f"中文字幕翻译：{batch_number}/{total}")
        return [
            Cue(cue.index, cue.start_ms, cue.end_ms, translated[cue.index])
            for cue in cues
        ]

    def process_job(self, job: VideoJob) -> dict[str, Any]:
        if (
            job.corrected_subtitle_path.exists()
            and job.chinese_subtitle_path.exists()
            and not self.config.overwrite
        ):
            self.runner.logger(f"跳过已有字幕结果：{job.title}")
            return {"video": str(job.video_path), "skipped": True}
        source = find_source_subtitle(job.video_path)
        if source is None:
            raise RuntimeError(f"没有找到英文字幕：{job.video_path.name}")
        cues = read_srt(source)
        if not cues:
            raise RuntimeError(f"字幕文件为空：{source}")
        prompt_tokens_before = self.prompt_tokens
        completion_tokens_before = self.completion_tokens
        self.runner.logger(f"开始处理：{job.title}（{len(cues)} 条字幕）")
        domain = self.analyze_domain(job, cues)
        self.runner.logger(f"推断领域：{domain.get('domain', '未确定')}")
        suspects = self.detect_suspects(cues, domain)
        self.runner.logger(f"发现 {len(suspects)} 条高可能错误字幕")
        corrected, repairs = self.repair_suspects(
            job, cues, domain, suspects
        )
        write_srt(job.corrected_subtitle_path, corrected)
        chinese = self.translate(corrected, domain)
        write_srt(job.chinese_subtitle_path, chinese)
        report = {
            "version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "video": str(job.video_path),
            "source_subtitle": str(source),
            "corrected_subtitle": str(job.corrected_subtitle_path),
            "chinese_subtitle": str(job.chinese_subtitle_path),
            "domain": domain,
            "suspects": [asdict(item) for item in suspects],
            "repairs": [asdict(item) for item in repairs],
            "usage": {
                "prompt_tokens": self.prompt_tokens - prompt_tokens_before,
                "completion_tokens": (
                    self.completion_tokens - completion_tokens_before
                ),
            },
            "settings": {
                "model": self.config.subtitle_model,
                "use_images": self.config.subtitle_use_vision,
                "suspect_threshold": self.config.subtitle_suspect_threshold,
            },
        }
        report_path = job.base_path.with_name(
            job.base_path.name + ".subtitle-report.json"
        )
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.runner.logger(f"英文校正字幕：{job.corrected_subtitle_path.name}")
        self.runner.logger(f"中文字幕：{job.chinese_subtitle_path.name}")
        self.runner.logger(f"修复报告：{report_path.name}")
        return report
