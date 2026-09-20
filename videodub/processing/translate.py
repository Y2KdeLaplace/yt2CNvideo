from __future__ import annotations

from pathlib import Path
from typing import Any

from ..media import VideoJob
from ..sentences import SentenceUnit


TRANSLATION_SYSTEM = """你是专业的视频文稿译者和校对者。
把无时间戳的完整校正文稿翻译为 {target_language}。先理解全文上下文，再逐个翻译 source_sentence_groups；这些句组按顺序拼接就是完整文稿，此阶段不做字幕分段或定时。
不得概括成讲义或摘要，不得省略推理、例子、剧情信息及有意义的重复；保留人物重复、停顿、笑点与口头节奏。术语、变量、单位和符号前后一致。
针对科普、课程或技术内容，准确保留概念关系和推导。遇到数学公式、概率、函数或计算的口语表达时，可以直接整理成紧凑的 ASCII 公式，不要使用 LaTeX。公式本身保持 ASCII，解释文字使用目标语言。例如英文“one minus e to the negative r of t k times delta”中的公式应写成“1 - exp(-r(t_k)*delta)”。
针对电影、剧集或生活对白，优先保留人物口吻、称谓、关系、情绪、潜台词、幽默和语境；使用目标语言中的自然口语，不要改写成科普讲解或书面总结，也不要无故弱化对剧情有作用的粗口、停顿或重复。
动作表达必须是可以直接说出口的自然口语，避免机械词典式翻译；不要改成解释说明。保留声效标签，不将声效变成对白。
目标语言要求：{language_guidance}
每个输入 group_id 必须且只能返回一次，不得遗漏、合并或新增 group_id。只返回 JSON 数组，每项格式为 {"group_id": 1, "translated_text": "..."}，不要使用 Markdown。"""


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


def translate_units(
    workflow: Any,
    units: list[SentenceUnit],
    domain: dict[str, Any],
    *,
    transcript_path: Path | None = None,
) -> list[SentenceUnit]:
    """Translate complete sentence groups while preserving their identities."""
    system = TRANSLATION_SYSTEM.replace(
        "{target_language}",
        workflow.config.translation_language,
    ).replace(
        "{language_guidance}",
        _language_guidance(workflow.config.translation_language),
    )
    translated = workflow._transform_units(
        units,
        domain,
        system,
        "translated_text",
        {"complete_corrected_transcript": "\n".join(unit.text for unit in units)},
    )
    if transcript_path is not None:
        transcript_path.write_text(
            "\n".join(unit.text for unit in translated) + "\n",
            encoding="utf-8",
        )
    return translated


def run_translate_stage(
    workflow: Any,
    job: VideoJob,
    *,
    repair_first: bool = False,
) -> dict[str, Any]:
    """Run stage 3, optionally sharing one pass with the requested repair."""
    return workflow.process_job(job, repair=repair_first, translate=True)
