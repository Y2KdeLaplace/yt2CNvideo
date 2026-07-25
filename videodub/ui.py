from __future__ import annotations

import ctypes
import os
import queue
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from videodub.config import AppConfig, api_key_from_runtime, load_config, save_config
from videodub.downloader import download
from videodub.media import VideoJob, discover_video_jobs
from videodub.platform_utils import open_in_file_manager
from videodub.runner import CancelledError, ProcessRunner
from videodub.subtitle_workflow import SubtitleRepairWorkflow
from videodub.subtitles import find_source_subtitle
from videodub.tts import EDGE_VOICES, dub_video


class VideoDubApp(tk.Tk):
    """YouTube 下载、字幕修复与翻译 UI。"""

    def __init__(self) -> None:
        super().__init__()
        self.title("YouTube 视频中文化工具")
        self.geometry("1000x820")
        self.minsize(880, 740)
        self.config_data = load_config()
        self.ui_font = tkfont.nametofont("TkDefaultFont").actual()["family"]
        self.mono_font = tkfont.nametofont("TkFixedFont").actual()["family"]
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.current_task = ""
        self.subtitle_jobs: list[VideoJob] = []
        self.dub_jobs: list[VideoJob] = []
        self.runner = ProcessRunner(lambda line: self.events.put(("log", line)))

        self.download_dir = tk.StringVar(value=self.config_data.download_dir)
        self.link_type = tk.StringVar(value=self.config_data.link_type)
        self.download_subtitles = tk.BooleanVar(
            value=self.config_data.download_subtitles
        )
        self.subtitle_languages = tk.StringVar(
            value=self.config_data.subtitle_languages
        )
        self.overwrite = tk.BooleanVar(value=self.config_data.overwrite)
        self.subtitle_api_base_url = tk.StringVar(
            value=self.config_data.subtitle_api_base_url
        )
        self.subtitle_api_key = tk.StringVar()
        self.subtitle_model = tk.StringVar(
            value=self.config_data.subtitle_model
        )
        self.subtitle_use_vision = tk.BooleanVar(
            value=self.config_data.subtitle_use_vision
        )
        self.output_dir = tk.StringVar(value=self.config_data.output_dir)
        self.tts_provider = tk.StringVar(value=self.config_data.tts_provider)
        self.tts_voice = tk.StringVar(value=self.config_data.tts_voice)
        self.tts_rate = tk.IntVar(value=self.config_data.tts_rate)
        self.piper_model_path = tk.StringVar(
            value=self.config_data.piper_model_path
        )
        self.audio_mode = tk.StringVar(value=self.config_data.audio_mode)
        self.original_volume = tk.DoubleVar(
            value=self.config_data.original_volume
        )
        self.embed_subtitles = tk.BooleanVar(
            value=self.config_data.embed_subtitles
        )
        self.status = tk.StringVar(value="正在检查工具…")

        self._configure_style()
        self._build_ui()
        self.after(100, self._drain_events)
        self.after(150, self._check_tools)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        self.option_add("*Font", (self.ui_font, 10))
        style.configure("Title.TLabel", font=(self.ui_font, 20, "bold"))
        style.configure("Hint.TLabel", foreground="#596579")
        style.configure("Primary.TButton", font=(self.ui_font, 11, "bold"))
        style.configure("TLabelframe.Label", font=(self.ui_font, 11, "bold"))
        style.configure("TButton", padding=(12, 7))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=18)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="YouTube 视频中文化工具", style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            outer,
            text="先下载视频与 YouTube 英文字幕，再用大模型校正字幕并翻译为中文。",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(4, 14))

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        task_tab = ttk.Frame(self.notebook, padding=10)
        subtitle_tab = ttk.Frame(self.notebook, padding=10)
        dub_tab = ttk.Frame(self.notebook, padding=10)
        log_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(task_tab, text="  1. 下载任务  ")
        self.notebook.add(subtitle_tab, text="  2. 字幕修复与翻译  ")
        self.notebook.add(dub_tab, text="  3. 中文配音  ")
        self.notebook.add(log_tab, text="  运行日志  ")

        folder_box = ttk.LabelFrame(task_tab, text="下载位置", padding=12)
        folder_box.pack(fill="x")
        ttk.Label(folder_box, text="保存到").grid(row=0, column=0, sticky="w")
        ttk.Entry(folder_box, textvariable=self.download_dir).grid(
            row=0, column=1, sticky="ew", padx=10
        )
        ttk.Button(folder_box, text="浏览…", command=self._browse_folder).grid(
            row=0, column=2
        )
        ttk.Button(folder_box, text="打开文件夹", command=self._open_folder).grid(
            row=0, column=3, padx=(8, 0)
        )
        folder_box.columnconfigure(1, weight=1)

        link_box = ttk.LabelFrame(task_tab, text="YouTube 链接", padding=12)
        link_box.pack(fill="both", expand=True, pady=12)
        ttk.Label(
            link_box,
            text="粘贴一个或多个链接，每行一个。请选择这批链接的类型。",
            style="Hint.TLabel",
        ).pack(anchor="w")
        controls = ttk.Frame(link_box)
        controls.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Label(controls, text="链接类型：").pack(side="left")
        ttk.Radiobutton(
            controls,
            text="单个视频",
            variable=self.link_type,
            value="single",
        ).pack(side="left", padx=(6, 16))
        ttk.Radiobutton(
            controls,
            text="YouTube 播放列表（playlist?list=…）",
            variable=self.link_type,
            value="playlist",
        ).pack(side="left")
        ttk.Checkbutton(
            controls,
            text="覆盖已有文件",
            variable=self.overwrite,
        ).pack(side="right")
        self.url_text = tk.Text(
            link_box,
            height=3,
            wrap="word",
            relief="solid",
            borderwidth=1,
            font=(self.ui_font, 10),
        )
        self.url_text.pack(fill="both", expand=True, pady=(8, 0))

        options = ttk.LabelFrame(task_tab, text="字幕下载选项", padding=12)
        options.pack(fill="x")
        ttk.Label(options, text="字幕语言").grid(row=0, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.subtitle_languages).grid(
            row=0, column=1, sticky="ew", padx=10
        )
        ttk.Label(
            options,
            text="默认只请求英文原字幕和英文字幕，以减少 YouTube HTTP 429 限流。",
            style="Hint.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            options,
            text="下载 YouTube 字幕（字幕失败或限流时，自动跳过字幕并继续下载视频）",
            variable=self.download_subtitles,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        options.columnconfigure(1, weight=1)

        actions = ttk.Frame(task_tab, padding=(0, 14, 0, 0))
        actions.pack(fill="x")
        self.download_button = ttk.Button(
            actions,
            text="开始下载",
            style="Primary.TButton",
            command=self._start_download,
        )
        self.download_button.pack(side="left")
        self.stop_button = ttk.Button(
            actions,
            text="停止",
            state="disabled",
            command=self._stop_download,
        )
        self.stop_button.pack(side="left", padx=10)
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=160)
        self.progress.pack(side="left", padx=(4, 10))
        ttk.Label(actions, textvariable=self.status, anchor="w").pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(actions, text="清空日志", command=self._clear_log).pack(side="right")

        log_box = ttk.LabelFrame(log_tab, text="运行日志", padding=8)
        log_box.pack(fill="both", expand=True)
        self.log = tk.Text(
            log_box,
            height=8,
            wrap="word",
            state="disabled",
            background="#111827",
            foreground="#e5e7eb",
            insertbackground="white",
            font=(self.mono_font, 9),
        )
        self.log.pack(fill="both", expand=True)

        self._build_subtitle_tab(subtitle_tab)
        self._build_dub_tab(dub_tab)

        # Reserve the task controls at the bottom before the URL box expands.
        link_box.pack_forget()
        options.pack_forget()
        actions.pack_forget()
        actions.pack(side="bottom", fill="x")
        options.pack(side="bottom", fill="x", pady=(10, 0))
        link_box.pack(fill="both", expand=True, pady=12)
        self._refresh_subtitle_jobs()
        self._refresh_dub_jobs()

    def _build_subtitle_tab(self, tab: ttk.Frame) -> None:
        files_box = ttk.LabelFrame(tab, text="选择已下载的视频", padding=10)
        files_box.pack(fill="both", expand=True)
        ttk.Label(
            files_box,
            text="这里只显示同时存在视频文件和英文字幕的项目；可按 Ctrl/Shift 多选。",
            style="Hint.TLabel",
        ).pack(anchor="w")
        list_frame = ttk.Frame(files_box)
        list_frame.pack(fill="both", expand=True, pady=(7, 0))
        self.subtitle_listbox = tk.Listbox(
            list_frame,
            height=6,
            selectmode="extended",
            activestyle="dotbox",
            exportselection=False,
            font=(self.ui_font, 9),
        )
        subtitle_scroll = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.subtitle_listbox.yview
        )
        self.subtitle_listbox.configure(yscrollcommand=subtitle_scroll.set)
        self.subtitle_listbox.pack(side="left", fill="both", expand=True)
        subtitle_scroll.pack(side="right", fill="y")
        file_actions = ttk.Frame(files_box)
        file_actions.pack(fill="x", pady=(7, 0))
        ttk.Button(
            file_actions, text="刷新列表", command=self._refresh_subtitle_jobs
        ).pack(side="left")
        ttk.Button(
            file_actions, text="全选", command=self._select_all_subtitle_jobs
        ).pack(side="left", padx=8)
        ttk.Label(
            file_actions,
            text="输入目录与“下载位置”相同",
            style="Hint.TLabel",
        ).pack(side="right")

        api_box = ttk.LabelFrame(tab, text="OpenAI 兼容 API", padding=10)
        api_box.pack(fill="x", pady=(10, 0))
        ttk.Label(api_box, text="API 地址").grid(row=0, column=0, sticky="w")
        ttk.Entry(api_box, textvariable=self.subtitle_api_base_url).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=(10, 0)
        )
        ttk.Label(api_box, text="API Key").grid(
            row=1, column=0, sticky="w", pady=(7, 0)
        )
        ttk.Entry(api_box, textvariable=self.subtitle_api_key, show="*").grid(
            row=1, column=1, sticky="ew", padx=(10, 14), pady=(7, 0)
        )
        ttk.Label(api_box, text="模型名称").grid(
            row=1, column=2, sticky="w", pady=(7, 0)
        )
        ttk.Entry(api_box, textvariable=self.subtitle_model).grid(
            row=1, column=3, sticky="ew", padx=(10, 0), pady=(7, 0)
        )
        ttk.Label(
            api_box,
            text=(
                "地址可填公网、localhost 或局域网，例如 http://127.0.0.1:8000/v1。"
                "同一个模型负责全部文本任务，并可按下方选项接收截图；API Key 不会保存。"
            ),
            style="Hint.TLabel",
            wraplength=850,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))
        api_box.columnconfigure(1, weight=1)
        api_box.columnconfigure(3, weight=1)

        settings_box = ttk.LabelFrame(tab, text="处理设置", padding=10)
        settings_box.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(
            settings_box,
            text="图像使用（把疑点时间段的截图发送给同一个模型）",
            variable=self.subtitle_use_vision,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            settings_box,
            text=(
                "默认所有任务都使用上方模型。开启图像时，该模型和兼容接口必须支持"
                " image_url；关闭后只发送文字。批量、上下文、疑点阈值和截图数量"
                "由程序自动管理。"
            ),
            style="Hint.TLabel",
            wraplength=790,
        ).grid(row=1, column=0, columnspan=8, sticky="w", pady=(3, 8))
        ttk.Checkbutton(
            settings_box,
            text="覆盖已有结果",
            variable=self.overwrite,
        ).grid(row=0, column=7, sticky="e")
        settings_box.columnconfigure(7, weight=1)

        actions = ttk.Frame(tab, padding=(0, 12, 0, 0))
        actions.pack(fill="x")
        self.subtitle_button = ttk.Button(
            actions,
            text="开始修复并翻译",
            style="Primary.TButton",
            command=self._start_subtitle_workflow,
        )
        self.subtitle_button.pack(side="left")
        self.subtitle_stop_button = ttk.Button(
            actions,
            text="停止",
            state="disabled",
            command=self._stop_download,
        )
        self.subtitle_stop_button.pack(side="left", padx=10)
        self.subtitle_progress = ttk.Progressbar(
            actions, mode="indeterminate", length=160
        )
        self.subtitle_progress.pack(side="left", padx=(4, 10))
        ttk.Label(actions, textvariable=self.status, anchor="w").pack(
            side="left", fill="x", expand=True
        )

    def _build_dub_tab(self, tab: ttk.Frame) -> None:
        files_box = ttk.LabelFrame(tab, text="选择带中文字幕的视频", padding=10)
        files_box.pack(fill="both", expand=True)
        ttk.Label(
            files_box,
            text="这里只显示已经生成 .zh-CN.srt 中文字幕的视频。",
            style="Hint.TLabel",
        ).pack(anchor="w")
        list_frame = ttk.Frame(files_box)
        list_frame.pack(fill="both", expand=True, pady=(7, 0))
        self.dub_listbox = tk.Listbox(
            list_frame,
            height=6,
            selectmode="extended",
            activestyle="dotbox",
            exportselection=False,
            font=(self.ui_font, 9),
        )
        dub_scroll = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.dub_listbox.yview
        )
        self.dub_listbox.configure(yscrollcommand=dub_scroll.set)
        self.dub_listbox.pack(side="left", fill="both", expand=True)
        dub_scroll.pack(side="right", fill="y")
        file_actions = ttk.Frame(files_box)
        file_actions.pack(fill="x", pady=(7, 0))
        ttk.Button(
            file_actions, text="刷新列表", command=self._refresh_dub_jobs
        ).pack(side="left")
        ttk.Button(
            file_actions, text="全选", command=self._select_all_dub_jobs
        ).pack(side="left", padx=8)

        output_box = ttk.LabelFrame(tab, text="输出位置", padding=10)
        output_box.pack(fill="x", pady=(10, 0))
        ttk.Entry(output_box, textvariable=self.output_dir).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(
            output_box, text="浏览…", command=self._browse_output_folder
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            output_box, text="打开文件夹", command=self._open_output_folder
        ).pack(side="left", padx=(8, 0))

        voice_box = ttk.LabelFrame(tab, text="中文声音", padding=10)
        voice_box.pack(fill="x", pady=(10, 0))
        provider_row = ttk.Frame(voice_box)
        provider_row.pack(fill="x")
        ttk.Radiobutton(
            provider_row,
            text="Edge TTS（在线、自然、免费，推荐）",
            variable=self.tts_provider,
            value="edge",
            command=self._update_tts_provider,
        ).pack(side="left")
        ttk.Radiobutton(
            provider_row,
            text="Piper（离线、需下载模型）",
            variable=self.tts_provider,
            value="piper",
            command=self._update_tts_provider,
        ).pack(side="left", padx=(24, 0))

        self.edge_options = ttk.Frame(voice_box)
        ttk.Label(self.edge_options, text="声音").pack(side="left")
        ttk.Combobox(
            self.edge_options,
            textvariable=self.tts_voice,
            values=EDGE_VOICES,
            width=28,
        ).pack(side="left", padx=(8, 18))
        ttk.Label(self.edge_options, text="语速").pack(side="left")
        ttk.Spinbox(
            self.edge_options,
            textvariable=self.tts_rate,
            from_=-50,
            to=50,
            increment=5,
            width=6,
        ).pack(side="left", padx=(8, 3))
        ttk.Label(self.edge_options, text="%").pack(side="left")
        self.edge_install_button = ttk.Button(
            self.edge_options,
            text="安装/更新 Edge TTS",
            command=self._install_edge_tts,
        )
        self.edge_install_button.pack(side="right")

        self.piper_options = ttk.Frame(voice_box)
        ttk.Label(self.piper_options, text="中文模型").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(
            self.piper_options, textvariable=self.piper_model_path
        ).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(
            self.piper_options,
            text="浏览…",
            command=self._browse_piper_model,
        ).grid(row=0, column=2)
        self.piper_options.columnconfigure(1, weight=1)

        audio_box = ttk.LabelFrame(tab, text="视频声音", padding=10)
        audio_box.pack(fill="x", pady=(10, 0))
        ttk.Radiobutton(
            audio_box,
            text="替换原音轨",
            variable=self.audio_mode,
            value="replace",
        ).pack(side="left")
        ttk.Radiobutton(
            audio_box,
            text="保留少量原声并混合",
            variable=self.audio_mode,
            value="mix",
        ).pack(side="left", padx=(18, 0))
        ttk.Label(audio_box, text="原声音量").pack(side="left", padx=(18, 6))
        ttk.Spinbox(
            audio_box,
            textvariable=self.original_volume,
            from_=0.0,
            to=1.0,
            increment=0.05,
            width=6,
        ).pack(side="left")
        ttk.Checkbutton(
            audio_box,
            text="把中文字幕嵌入视频",
            variable=self.embed_subtitles,
        ).pack(side="right")

        actions = ttk.Frame(tab, padding=(0, 12, 0, 0))
        actions.pack(fill="x")
        self.dub_button = ttk.Button(
            actions,
            text="生成中文配音视频",
            style="Primary.TButton",
            command=self._start_dubbing,
        )
        self.dub_button.pack(side="left")
        self.dub_stop_button = ttk.Button(
            actions,
            text="停止",
            state="disabled",
            command=self._stop_download,
        )
        self.dub_stop_button.pack(side="left", padx=10)
        self.dub_progress = ttk.Progressbar(
            actions, mode="indeterminate", length=160
        )
        self.dub_progress.pack(side="left", padx=(4, 10))
        ttk.Label(actions, textvariable=self.status, anchor="w").pack(
            side="left", fill="x", expand=True
        )
        self._update_tts_provider()

    def _check_tools(self) -> None:
        problems = self.config_data.validate_core()
        if problems:
            self.status.set("工具检查失败")
            self._append_log("\n".join(problems))
            messagebox.showerror(
                "工具路径错误",
                "\n".join(problems)
                + "\n\n请安装 yt-dlp 和 ffmpeg，或把它们加入系统 PATH。",
                parent=self,
            )
        else:
            self.status.set(
                f"工具正常：yt-dlp {Path(self.config_data.yt_dlp_path).name}"
            )
            self._append_log("yt-dlp、ffmpeg、ffprobe 检查通过，可以开始下载。")

    def _browse_folder(self) -> None:
        selected = filedialog.askdirectory(
            initialdir=self.download_dir.get() or None,
            parent=self,
        )
        if selected:
            self.download_dir.set(selected)
            self._refresh_subtitle_jobs()
            self._refresh_dub_jobs()

    def _open_folder(self) -> None:
        path = Path(self.download_dir.get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        try:
            open_in_file_manager(path)
        except Exception as exc:
            messagebox.showerror("无法打开文件夹", str(exc), parent=self)

    def _browse_output_folder(self) -> None:
        selected = filedialog.askdirectory(
            initialdir=self.output_dir.get() or None,
            parent=self,
        )
        if selected:
            self.output_dir.set(selected)

    def _open_output_folder(self) -> None:
        path = Path(self.output_dir.get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        try:
            open_in_file_manager(path)
        except Exception as exc:
            messagebox.showerror("无法打开文件夹", str(exc), parent=self)

    def _browse_piper_model(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self,
            filetypes=(("ONNX 模型", "*.onnx"), ("所有文件", "*.*")),
        )
        if selected:
            self.piper_model_path.set(selected)

    def _update_tts_provider(self) -> None:
        self.edge_options.pack_forget()
        self.piper_options.pack_forget()
        if self.tts_provider.get() == "piper":
            self.piper_options.pack(fill="x", pady=(10, 0))
        else:
            self.edge_options.pack(fill="x", pady=(10, 0))

    def _urls(self) -> list[str]:
        return [
            line.strip()
            for line in self.url_text.get("1.0", "end").splitlines()
            if line.strip()
        ]

    def _refresh_subtitle_jobs(self) -> None:
        root = self.download_dir.get().strip()
        jobs = discover_video_jobs(root) if root else []
        self.subtitle_jobs = [
            job for job in jobs if find_source_subtitle(job.video_path) is not None
        ]
        self.subtitle_listbox.delete(0, "end")
        for job in self.subtitle_jobs:
            source = find_source_subtitle(job.video_path)
            subtitle_name = source.name if source else "无字幕"
            self.subtitle_listbox.insert(
                "end", f"{job.title}    [{subtitle_name}]"
            )
        if self.subtitle_jobs:
            self.subtitle_listbox.selection_set(0)

    def _select_all_subtitle_jobs(self) -> None:
        if self.subtitle_jobs:
            self.subtitle_listbox.selection_set(0, "end")

    def _selected_subtitle_jobs(self) -> list[VideoJob]:
        return [
            self.subtitle_jobs[index]
            for index in self.subtitle_listbox.curselection()
            if index < len(self.subtitle_jobs)
        ]

    def _refresh_dub_jobs(self) -> None:
        root = self.download_dir.get().strip()
        jobs = discover_video_jobs(root) if root else []
        self.dub_jobs = [
            job for job in jobs if job.chinese_subtitle_path.is_file()
        ]
        self.dub_listbox.delete(0, "end")
        for job in self.dub_jobs:
            self.dub_listbox.insert("end", job.title)
        if self.dub_jobs:
            self.dub_listbox.selection_set(0)

    def _select_all_dub_jobs(self) -> None:
        if self.dub_jobs:
            self.dub_listbox.selection_set(0, "end")

    def _selected_dub_jobs(self) -> list[VideoJob]:
        return [
            self.dub_jobs[index]
            for index in self.dub_listbox.curselection()
            if index < len(self.dub_jobs)
        ]

    def _make_config(self) -> AppConfig:
        config = load_config()
        config.download_dir = self.download_dir.get()
        config.link_type = self.link_type.get()
        config.download_subtitles = bool(self.download_subtitles.get())
        config.subtitle_languages = self.subtitle_languages.get().strip()
        config.overwrite = bool(self.overwrite.get())
        config.normalize()
        config.ensure_directories()
        problems = config.validate_core()
        if problems:
            raise ValueError("\n".join(problems))
        save_config(config)
        return config

    def _make_subtitle_config(self) -> AppConfig:
        config = self._make_config()
        config.subtitle_api_base_url = self.subtitle_api_base_url.get().strip()
        config.subtitle_model = self.subtitle_model.get().strip()
        config.subtitle_use_vision = bool(self.subtitle_use_vision.get())
        if not config.subtitle_api_base_url:
            raise ValueError("请填写 OpenAI 兼容 API 地址。")
        if not config.subtitle_model:
            raise ValueError("请填写模型名称。")
        config.normalize()
        save_config(config)
        return config

    def _make_tts_config(self) -> AppConfig:
        config = self._make_config()
        config.output_dir = self.output_dir.get()
        config.tts_provider = self.tts_provider.get()
        config.tts_voice = self.tts_voice.get().strip()
        config.tts_rate = int(self.tts_rate.get())
        config.piper_model_path = self.piper_model_path.get().strip()
        config.audio_mode = self.audio_mode.get()
        config.original_volume = float(self.original_volume.get())
        config.embed_subtitles = bool(self.embed_subtitles.get())
        if config.tts_provider == "edge" and not config.tts_voice:
            raise ValueError("请选择或填写 Edge TTS 中文声音。")
        if config.tts_provider == "piper" and not config.piper_model_path:
            raise ValueError("请选择 Piper 中文 ONNX 模型。")
        config.normalize()
        config.ensure_directories()
        save_config(config)
        return config

    def _set_busy(self, task: str) -> None:
        self.current_task = task
        self.download_button.configure(state="disabled")
        self.subtitle_button.configure(state="disabled")
        self.dub_button.configure(state="disabled")
        self.edge_install_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.subtitle_stop_button.configure(state="disabled")
        self.dub_stop_button.configure(state="disabled")
        if task == "download":
            self.stop_button.configure(state="normal")
            self.progress.start(12)
        elif task == "subtitle":
            self.subtitle_stop_button.configure(state="normal")
            self.subtitle_progress.start(12)
        else:
            self.dub_stop_button.configure(state="normal")
            self.dub_progress.start(12)

    def _start_download(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        urls = self._urls()
        if not urls:
            messagebox.showwarning("缺少链接", "请先粘贴 YouTube 链接。", parent=self)
            return
        try:
            config = self._make_config()
        except Exception as exc:
            messagebox.showerror("设置错误", str(exc), parent=self)
            return

        self.runner.reset()
        self._set_busy("download")
        self.status.set("正在下载…")
        self._append_log("\n" + "=" * 60)
        self.worker = threading.Thread(
            target=self._download_worker,
            args=(config, urls),
            daemon=True,
        )
        self.worker.start()

    def _install_edge_tts(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.runner.reset()
        self._set_busy("install_edge")
        self.status.set("正在安装 Edge TTS…")
        self._append_log("\n" + "=" * 60)
        self.worker = threading.Thread(
            target=self._install_edge_worker,
            daemon=True,
        )
        self.worker.start()

    def _start_dubbing(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        jobs = self._selected_dub_jobs()
        if not jobs:
            messagebox.showwarning(
                "没有选择视频",
                "请先完成中文字幕翻译，再刷新并选择视频。",
                parent=self,
            )
            return
        try:
            config = self._make_tts_config()
        except Exception as exc:
            messagebox.showerror("设置错误", str(exc), parent=self)
            return

        self.runner.reset()
        self._set_busy("dub")
        self.status.set("正在生成中文配音…")
        self._append_log("\n" + "=" * 60)
        self._append_log(
            "配音流程：中文字幕 → 中文语音 → 按字幕时间对齐 → 替换/混合音轨"
        )
        self.worker = threading.Thread(
            target=self._dub_worker,
            args=(config, jobs),
            daemon=True,
        )
        self.worker.start()

    def _start_subtitle_workflow(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        jobs = self._selected_subtitle_jobs()
        if not jobs:
            messagebox.showwarning(
                "没有选择视频",
                "请先刷新列表，并选择至少一个带英文字幕的视频。",
                parent=self,
            )
            return
        try:
            config = self._make_subtitle_config()
        except Exception as exc:
            messagebox.showerror("设置错误", str(exc), parent=self)
            return

        self.runner.reset()
        self._set_busy("subtitle")
        self.status.set("正在分析字幕…")
        self._append_log("\n" + "=" * 60)
        self._append_log(
            "字幕流程：YouTube 英文字幕 → 领域分析 → 疑点定位"
            " → 可选截图复核 → 英文校正 → 中文翻译"
        )
        api_key = api_key_from_runtime(self.subtitle_api_key.get())
        self.worker = threading.Thread(
            target=self._subtitle_worker,
            args=(config, jobs, api_key),
            daemon=True,
        )
        self.worker.start()

    def _download_worker(self, config: AppConfig, urls: list[str]) -> None:
        try:
            count = 0
            for number, url in enumerate(urls, 1):
                self.events.put(("status", f"正在处理链接 {number}/{len(urls)}"))
                count += len(download(config, self.runner, url))
            self.events.put(("done", f"下载完成，共找到 {count} 个视频。"))
        except CancelledError:
            self.events.put(("cancelled", "下载已停止"))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _subtitle_worker(
        self,
        config: AppConfig,
        jobs: list[VideoJob],
        api_key: str,
    ) -> None:
        try:
            workflow = SubtitleRepairWorkflow(
                config,
                self.runner,
                api_key=api_key,
            )
            for number, job in enumerate(jobs, 1):
                self.events.put(
                    ("status", f"正在处理字幕 {number}/{len(jobs)}：{job.title}")
                )
                workflow.process_job(job)
            self.events.put(
                ("done", f"字幕修复与翻译完成，共处理 {len(jobs)} 个视频。")
            )
        except CancelledError:
            self.events.put(("cancelled", "字幕任务已停止"))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _install_edge_worker(self) -> None:
        try:
            self.runner.run(
                [sys.executable, "-m", "pip", "install", "-U", "edge-tts"]
            )
            self.events.put(("done", "Edge TTS 安装完成，可以开始配音。"))
        except CancelledError:
            self.events.put(("cancelled", "安装已停止"))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _dub_worker(self, config: AppConfig, jobs: list[VideoJob]) -> None:
        try:
            outputs: list[Path] = []
            for number, job in enumerate(jobs, 1):
                self.events.put(
                    ("status", f"正在配音 {number}/{len(jobs)}：{job.title}")
                )
                output = dub_video(config, self.runner, job)
                outputs.append(output)
                self.runner.logger(f"中文配音视频：{output}")
            self.events.put(
                ("done", f"中文配音完成，共生成 {len(outputs)} 个视频。")
            )
        except CancelledError:
            self.events.put(("cancelled", "配音任务已停止"))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _stop_download(self) -> None:
        self.status.set("正在停止…")
        self.runner.cancel()

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain_events(self) -> None:
        try:
            while True:
                event, text = self.events.get_nowait()
                if event == "log":
                    self._append_log(text)
                elif event == "status":
                    self.status.set(text)
                elif event == "done":
                    finished_task = self.current_task
                    self._finish(text)
                    if finished_task == "download":
                        self._refresh_subtitle_jobs()
                    elif finished_task == "subtitle":
                        self._refresh_dub_jobs()
                    messagebox.showinfo("任务完成", text, parent=self)
                elif event == "cancelled":
                    self._finish(text)
                elif event == "error":
                    self._append_log("错误：" + text)
                    self._finish("任务失败")
                    messagebox.showerror("任务失败", text, parent=self)
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _finish(self, text: str) -> None:
        self.progress.stop()
        self.subtitle_progress.stop()
        self.dub_progress.stop()
        self.download_button.configure(state="normal")
        self.subtitle_button.configure(state="normal")
        self.dub_button.configure(state="normal")
        self.edge_install_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.subtitle_stop_button.configure(state="disabled")
        self.dub_stop_button.configure(state="disabled")
        self.current_task = ""
        self.status.set(text)

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno(
                "任务仍在进行",
                "关闭窗口会停止当前任务，确定关闭吗？",
                parent=self,
            ):
                return
            self.runner.cancel()
        self.destroy()


def main() -> None:
    if os.name == "nt":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    VideoDubApp().mainloop()


if __name__ == "__main__":
    main()
