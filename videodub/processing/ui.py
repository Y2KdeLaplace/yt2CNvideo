from __future__ import annotations

from pathlib import Path
from tkinter import ttk
from typing import Any

from ..config import SUPPORTED_LANGUAGES
from ..media import VideoJob


PROCESS_COLUMNS = (
    "video",
    "downloaded",
    "asr",
    "corrected",
    "translated",
    "dubbed",
)


def job_status_values(
    job: VideoJob,
    source: Path | None,
    translation_language: str,
    tts_language: str,
    output_dir: str,
) -> tuple[str, str, str, str, str, str]:
    translated = job.translated_subtitle_path(translation_language)
    dubbed = (
        job.dubbed_video_path(tts_language, output_dir).is_file()
        or job.dubbed_audio_path(tts_language, output_dir).is_file()
    )
    return (
        job.title,
        "●" if source else "—",
        "●" if job.asr_subtitle_path.is_file() else "—",
        "●" if job.corrected_subtitle_path.is_file() else "—",
        "●" if translated.is_file() else "—",
        "●" if dubbed else "—",
    )


def build_processing_section(app: Any, parent: ttk.Frame) -> None:
    """Build the four-stage processing UI against the host application."""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)
    selection = ttk.LabelFrame(parent, text="选择视频", padding=8)
    selection.grid(row=0, column=0, sticky="nsew")
    app.refresh_jobs_button = ttk.Button(
        selection,
        text="刷新",
        command=app._refresh_jobs,
        style="Main.TButton",
        takefocus=False,
    )
    app.refresh_jobs_button.pack(anchor="e", pady=(0, 5))
    app.job_tree = ttk.Treeview(
        selection,
        columns=PROCESS_COLUMNS,
        show="headings",
        selectmode="extended",
        height=9,
    )
    labels = {
        "video": "视频/字幕名",
        "downloaded": "下载字幕",
        "asr": "识别字幕",
        "corrected": "修复字幕",
        "translated": "翻译字幕",
        "dubbed": "配音完成",
    }
    for key in PROCESS_COLUMNS:
        app.job_tree.heading(key, text=labels[key])
        width = 420 if key == "video" else 100
        anchor = "w" if key == "video" else "center"
        app.job_tree.column(key, width=width, anchor=anchor)
    scroll = ttk.Scrollbar(selection, orient="vertical", command=app.job_tree.yview)
    app.job_tree.configure(yscrollcommand=scroll.set)
    app.job_tree.bind("<Button-1>", app._clear_job_selection_on_blank)
    app.job_tree.bind("<Button-3>", app._show_tree_menu)
    app.job_tree.bind("<Button-2>", app._show_tree_menu)
    app.job_tree.bind("<Control-Button-1>", app._toggle_job_selection)
    app.job_tree.bind("<Command-Button-1>", app._toggle_job_selection)
    app.job_tree.bind("<Control-a>", app._select_all_jobs)
    app.job_tree.bind("<Control-A>", app._select_all_jobs)
    app.job_tree.bind("<Command-a>", app._select_all_jobs)
    app.job_tree.bind("<Command-A>", app._select_all_jobs)
    app.job_tree.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    controls = ttk.Frame(parent)
    controls.grid(row=1, column=0, sticky="ew", pady=(10, 0))
    app.extract_check = ttk.Checkbutton(
        controls,
        text="提取",
        variable=app.stage_asr,
        command=app._update_stage_language_states,
        style="Stage.TCheckbutton",
    )
    app.extract_check.pack(side="left")
    app.asr_language_combo = ttk.Combobox(
        controls,
        textvariable=app.asr_language,
        values=tuple(SUPPORTED_LANGUAGES),
        state="readonly",
        width=7,
    )
    app.asr_language_combo.pack(side="left", padx=(5, 14))
    ttk.Checkbutton(
        controls,
        text="修复",
        variable=app.stage_repair,
        style="Stage.TCheckbutton",
    ).pack(side="left", padx=(0, 14))
    ttk.Checkbutton(
        controls,
        text="翻译",
        variable=app.stage_translate,
        command=app._update_stage_language_states,
        style="Stage.TCheckbutton",
    ).pack(side="left")
    app.translation_language_combo = ttk.Combobox(
        controls,
        textvariable=app.translation_language,
        values=tuple(SUPPORTED_LANGUAGES),
        state="readonly",
        width=7,
    )
    app.translation_language_combo.pack(side="left", padx=(5, 14))
    ttk.Checkbutton(
        controls,
        text="配音",
        variable=app.stage_dub,
        command=app._update_stage_language_states,
        style="Stage.TCheckbutton",
    ).pack(side="left")
    app.tts_language_combo = ttk.Combobox(
        controls,
        textvariable=app.tts_language,
        values=tuple(SUPPORTED_LANGUAGES),
        state="readonly",
        width=7,
    )
    app.tts_language_combo.pack(side="left", padx=(5, 14))
    ttk.Checkbutton(
        controls,
        text="并行处理",
        variable=app.parallel_enabled,
        command=app._update_parallel_state,
        style="Stage.TCheckbutton",
    ).pack(side="left", padx=(4, 7))
    app.parallel_entry = ttk.Entry(
        controls,
        textvariable=app.parallel_count,
        width=5,
        state="disabled",
        justify="center",
    )
    app.parallel_entry.pack(side="left")
    app.process_button = ttk.Button(
        controls,
        text="运行",
        command=app._start_processing,
        style="Main.TButton",
    )
    app.process_button.pack(side="right")
    app._update_stage_language_states()
    app._refresh_jobs()
