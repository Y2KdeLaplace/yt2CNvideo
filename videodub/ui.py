from __future__ import annotations

import ctypes
import os
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from videodub.config import (
    AppConfig,
    api_key_from_runtime,
    load_config,
    migrate_legacy_layout,
    migrate_work_directory,
    save_config,
)
from videodub.downloader import download
from videodub.media import VideoJob, discover_video_jobs
from videodub.platform_utils import open_in_file_manager
from videodub.qwen_speech import (
    QwenServiceInfo,
    check_qwen_service,
    extract_asr_subtitle,
)
from videodub.runner import CancelledError, ProcessRunner
from videodub.subtitle_workflow import SpeechSubtitleWorkflow, SubtitleRepairWorkflow
from videodub.subtitles import find_source_subtitle
from videodub.tts import dub_video


class VideoDubApp(tk.Tk):
    """YouTube 下载、字幕提取、字幕修复翻译与中文配音 UI。"""

    LOG_KEYS = ("download", "asr", "subtitle", "dub")

    def __init__(self) -> None:
        super().__init__()
        self.title("YouTube 视频中文化工具")
        self.geometry("1040x900")
        self.minsize(920, 780)
        self.config_data = load_config()
        migrate_legacy_layout(self.config_data.work_dir)
        self.config_data.ensure_directories()
        self.ui_font = tkfont.nametofont("TkDefaultFont").actual()["family"]
        self.mono_font = tkfont.nametofont("TkFixedFont").actual()["family"]
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.current_task = ""
        self.logs: dict[str, tk.Text] = {}
        self.start_buttons: dict[str, ttk.Button] = {}
        self.stop_buttons: dict[str, ttk.Button] = {}
        self.runner = ProcessRunner(lambda line: self.events.put(("log", line)))

        self.asr_jobs: list[VideoJob] = []
        self.subtitle_jobs: list[VideoJob] = []
        self.dub_jobs: list[VideoJob] = []

        self.work_dir = tk.StringVar(value=self.config_data.work_dir)
        self.link_type = tk.StringVar(value=self.config_data.link_type)
        self.subtitle_languages = tk.StringVar(
            value=self.config_data.subtitle_languages
        )
        self.qwen_asr_enabled = tk.BooleanVar(
            value=self.config_data.qwen_asr_enabled
        )
        self.qwen_tts_enabled = tk.BooleanVar(
            value=self.config_data.qwen_tts_enabled
        )
        self.qwen_asr_model = tk.StringVar(value="正在检测服务…")
        self.qwen_tts_model = tk.StringVar(value="正在检测服务…")
        self.subtitle_api_base_url = tk.StringVar(
            value=self.config_data.subtitle_api_base_url
        )
        self.subtitle_api_key = tk.StringVar()
        self.subtitle_model = tk.StringVar(value=self.config_data.subtitle_model)
        self.save_model_info = tk.BooleanVar(
            value=self.config_data.save_model_info
        )
        self.status = tk.StringVar(value="正在检查工具…")
        self._service_checks_running: set[str] = set()

        self._configure_style()
        self._build_ui()
        self.after(100, self._drain_events)
        self.after(150, self._check_tools)
        self.after(250, lambda: self._check_qwen_service_async("asr"))
        self.after(300, lambda: self._check_qwen_service_async("tts"))
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        self.option_add("*Font", (self.ui_font, 10))
        style.configure("Hint.TLabel", foreground="#6b7280")
        style.configure("TLabelframe.Label", font=(self.ui_font, 11, "bold"))
        style.configure("TButton", padding=(12, 7), font=(self.ui_font, 10))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)

        work_box = ttk.LabelFrame(outer, text="工作路径", padding=10)
        work_box.pack(fill="x", pady=(0, 12))
        ttk.Entry(work_box, textvariable=self.work_dir).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(work_box, text="浏览…", command=self._browse_work_folder).grid(
            row=0, column=1, padx=(8, 0)
        )
        ttk.Button(work_box, text="打开文件夹", command=self._open_work_folder).grid(
            row=0, column=2, padx=(8, 0)
        )
        work_box.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        self.download_tab = ttk.Frame(self.notebook, padding=10)
        self.asr_tab = ttk.Frame(self.notebook, padding=10)
        self.subtitle_tab = ttk.Frame(self.notebook, padding=10)
        self.dub_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.download_tab, text="  YouTube 视频下载  ")
        self.notebook.add(self.asr_tab, text="  字幕提取  ")
        self.notebook.add(self.subtitle_tab, text="  字幕修复与翻译  ")
        self.notebook.add(self.dub_tab, text="  中文配音  ")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_download_tab(self.download_tab)
        self._build_asr_tab(self.asr_tab)
        self._build_subtitle_tab(self.subtitle_tab)
        self._build_dub_tab(self.dub_tab)
        self._refresh_all_jobs()

    def _build_log(self, tab: ttk.Frame, key: str) -> None:
        log_box = ttk.LabelFrame(tab, text="运行日志", padding=7)
        log_box.pack(side="bottom", fill="both", pady=(10, 0))
        log = tk.Text(
            log_box,
            height=7,
            wrap="word",
            state="disabled",
            background="#111827",
            foreground="#e5e7eb",
            insertbackground="white",
            font=(self.mono_font, 9),
        )
        scroll = ttk.Scrollbar(log_box, orient="vertical", command=log.yview)
        log.configure(yscrollcommand=scroll.set)
        log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.logs[key] = log

    def _build_actions(
        self,
        tab: ttk.Frame,
        key: str,
        action_text: str,
        command: object,
    ) -> None:
        actions = ttk.Frame(tab, padding=(0, 10, 0, 0))
        actions.pack(side="bottom", fill="x")
        ttk.Label(actions, textvariable=self.status, anchor="w").pack(
            side="left", fill="x", expand=True
        )
        stop = ttk.Button(
            actions,
            text="停止",
            state="disabled",
            command=self._stop_task,
        )
        start = ttk.Button(actions, text=action_text, command=command)
        start.pack(side="right")
        stop.pack(side="right", padx=(0, 8))
        self.start_buttons[key] = start
        self.stop_buttons[key] = stop

    def _build_download_tab(self, tab: ttk.Frame) -> None:
        self._build_log(tab, "download")
        self._build_actions(tab, "download", "下载", self._start_download)

        link_box = ttk.LabelFrame(tab, text="YouTube 链接", padding=12)
        link_box.pack(fill="both", expand=True)
        self.url_text = tk.Text(
            link_box,
            height=6,
            wrap="word",
            relief="solid",
            borderwidth=1,
            font=(self.ui_font, 10),
        )
        self.url_text.pack(fill="both", expand=True)
        self.url_placeholder = tk.Label(
            self.url_text,
            text="粘贴一个或多个链接，每行一个",
            foreground="#9ca3af",
            background="white",
            anchor="nw",
            padx=2,
            pady=2,
        )
        self.url_placeholder.place(x=4, y=4)
        self.url_placeholder.bind("<Button-1>", lambda _event: self.url_text.focus_set())
        self.url_text.bind("<KeyRelease>", self._update_url_placeholder)

        controls = ttk.Frame(link_box)
        controls.pack(fill="x", pady=(10, 0))
        ttk.Label(controls, text="链接类型：").pack(side="left")
        ttk.Radiobutton(
            controls,
            text="单个视频",
            variable=self.link_type,
            value="single",
        ).pack(side="left", padx=(6, 16))
        ttk.Radiobutton(
            controls,
            text="YouTube 播放列表",
            variable=self.link_type,
            value="playlist",
        ).pack(side="left")

        options = ttk.LabelFrame(tab, text="字幕下载选项", padding=12)
        options.pack(fill="x", pady=(10, 0))
        ttk.Label(options, text="字幕语言").grid(row=0, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.subtitle_languages).grid(
            row=0, column=1, sticky="ew", padx=(10, 0)
        )
        options.columnconfigure(1, weight=1)

    def _build_video_list(
        self,
        parent: ttk.LabelFrame,
        *,
        refresh_command: object,
        select_command: object,
    ) -> tk.Listbox:
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill="both", expand=True)
        listbox = tk.Listbox(
            list_frame,
            height=6,
            selectmode="extended",
            activestyle="dotbox",
            exportselection=False,
            font=(self.ui_font, 9),
        )
        scroll = ttk.Scrollbar(
            list_frame, orient="vertical", command=listbox.yview
        )
        listbox.configure(yscrollcommand=scroll.set)
        listbox.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        buttons = ttk.Frame(parent)
        buttons.pack(fill="x", pady=(7, 0))
        ttk.Button(buttons, text="刷新列表", command=refresh_command).pack(side="left")
        ttk.Button(buttons, text="全选", command=select_command).pack(
            side="left", padx=8
        )
        return listbox

    def _build_asr_tab(self, tab: ttk.Frame) -> None:
        self._build_log(tab, "asr")
        self._build_actions(tab, "asr", "字幕提取", self._start_asr)

        files = ttk.LabelFrame(tab, text="选择视频", padding=10)
        files.pack(fill="both", expand=True)
        self.asr_listbox = self._build_video_list(
            files,
            refresh_command=self._refresh_asr_jobs,
            select_command=self._select_all_asr_jobs,
        )

        service = ttk.LabelFrame(tab, text="Qwen3-ASR 字幕提取", padding=10)
        service.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(
            service,
            text="启用",
            variable=self.qwen_asr_enabled,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(service, text="模型路径").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(
            service,
            textvariable=self.qwen_asr_model,
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(8, 0))
        service.columnconfigure(1, weight=1)

    def _build_model_box(self, tab: ttk.Frame, title: str) -> None:
        api = ttk.LabelFrame(tab, text=title, padding=10)
        api.pack(fill="x", pady=(10, 0))
        ttk.Label(api, text="API 地址").grid(row=0, column=0, sticky="w")
        ttk.Entry(api, textvariable=self.subtitle_api_base_url).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=(10, 0)
        )
        ttk.Label(api, text="API Key").grid(
            row=1, column=0, sticky="w", pady=(7, 0)
        )
        ttk.Entry(api, textvariable=self.subtitle_api_key, show="*").grid(
            row=1, column=1, sticky="ew", padx=(10, 14), pady=(7, 0)
        )
        ttk.Label(api, text="模型名称").grid(
            row=1, column=2, sticky="w", pady=(7, 0)
        )
        ttk.Entry(api, textvariable=self.subtitle_model).grid(
            row=1, column=3, sticky="ew", padx=(10, 0), pady=(7, 0)
        )
        ttk.Checkbutton(
            api,
            text="保存信息（API Key 不写入磁盘）",
            variable=self.save_model_info,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))
        api.columnconfigure(1, weight=1)
        api.columnconfigure(3, weight=1)

    def _build_subtitle_tab(self, tab: ttk.Frame) -> None:
        self._build_log(tab, "subtitle")
        self._build_actions(
            tab,
            "subtitle",
            "字幕修复与翻译",
            self._start_subtitle_workflow,
        )

        files = ttk.LabelFrame(tab, text="选择配对字幕", padding=10)
        files.pack(fill="both", expand=True)
        self.subtitle_listbox = self._build_video_list(
            files,
            refresh_command=self._refresh_subtitle_jobs,
            select_command=self._select_all_subtitle_jobs,
        )
        self._build_model_box(tab, "修复字幕模型")

    def _build_dub_tab(self, tab: ttk.Frame) -> None:
        self._build_log(tab, "dub")
        self._build_actions(tab, "dub", "中文配音", self._start_dubbing)

        files = ttk.LabelFrame(tab, text="选择中文字幕", padding=10)
        files.pack(fill="both", expand=True)
        self.dub_listbox = self._build_video_list(
            files,
            refresh_command=self._refresh_dub_jobs,
            select_command=self._select_all_dub_jobs,
        )
        self._build_model_box(tab, "配音文本模型")

        service = ttk.LabelFrame(tab, text="Qwen3-TTS 语音生成", padding=10)
        service.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(
            service,
            text="启用（未启用时使用 Edge TTS）",
            variable=self.qwen_tts_enabled,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(service, text="模型路径").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(
            service,
            textvariable=self.qwen_tts_model,
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(8, 0))
        service.columnconfigure(1, weight=1)

    def _update_url_placeholder(self, _event: object | None = None) -> None:
        if self.url_text.get("1.0", "end-1c"):
            self.url_placeholder.place_forget()
        else:
            self.url_placeholder.place(x=4, y=4)

    def _check_tools(self) -> None:
        problems = self.config_data.validate_core()
        if problems:
            self.status.set("工具检查失败")
            self._append_log("download", "\n".join(problems))
            messagebox.showerror(
                "工具路径错误",
                "\n".join(problems)
                + "\n\n请安装 yt-dlp 和 ffmpeg，或把它们加入系统 PATH。",
                parent=self,
            )
        else:
            self.status.set("工具检查通过")
            self._append_log(
                "download",
                "yt-dlp、ffmpeg、ffprobe 检查通过，可以开始下载。",
            )

    def _on_tab_changed(self, _event: object) -> None:
        index = self.notebook.index(self.notebook.select())
        if index == 1:
            self._check_qwen_service_async("asr")
        elif index == 3:
            self._check_qwen_service_async("tts")

    def _check_qwen_service_async(self, service_type: str) -> None:
        if service_type in self._service_checks_running:
            return
        self._service_checks_running.add(service_type)
        base_url = (
            self.config_data.qwen_asr_base_url
            if service_type == "asr"
            else self.config_data.qwen_tts_base_url
        )

        def worker() -> None:
            self.events.put(
                (
                    f"{service_type}_service",
                    check_qwen_service(base_url, service_type),
                )
            )

        threading.Thread(target=worker, daemon=True).start()

    def _apply_service_info(
        self,
        service_type: str,
        info: QwenServiceInfo,
    ) -> None:
        self._service_checks_running.discard(service_type)
        model_var = self.qwen_asr_model if service_type == "asr" else self.qwen_tts_model
        model_var.set(info.display)
        if info.available:
            self._append_log(
                service_type if service_type == "asr" else "dub",
                f"检测到 Qwen3-{service_type.upper()}：{info.display}",
            )

    def _browse_work_folder(self) -> None:
        selected = filedialog.askdirectory(
            initialdir=self.work_dir.get() or None,
            parent=self,
        )
        if not selected:
            return
        try:
            migrate_work_directory(self.work_dir.get(), selected)
            self.work_dir.set(str(Path(selected).resolve()))
            config = self._make_config()
            self.config_data = config
            self._append_log(
                "download",
                f"工作目录已迁移到：{config.work_dir}",
            )
            self._refresh_all_jobs()
        except Exception as exc:
            messagebox.showerror("迁移工作目录失败", str(exc), parent=self)

    def _open_work_folder(self) -> None:
        path = Path(self.work_dir.get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        (path / "output").mkdir(exist_ok=True)
        try:
            open_in_file_manager(path)
        except Exception as exc:
            messagebox.showerror("无法打开文件夹", str(exc), parent=self)

    def _jobs(self) -> list[VideoJob]:
        root = self.work_dir.get().strip()
        if not root:
            return []
        return discover_video_jobs(root, Path(root) / "output")

    def _refresh_all_jobs(self) -> None:
        self._refresh_asr_jobs()
        self._refresh_subtitle_jobs()
        self._refresh_dub_jobs()

    def _refresh_asr_jobs(self) -> None:
        self.asr_jobs = self._jobs()
        self.asr_listbox.delete(0, "end")
        for job in self.asr_jobs:
            suffix = "已有 ASR 字幕" if job.asr_subtitle_path.is_file() else "待提取"
            self.asr_listbox.insert("end", f"{job.title}    [{suffix}]")
        if self.asr_jobs:
            self.asr_listbox.selection_set(0)

    def _refresh_subtitle_jobs(self) -> None:
        self.subtitle_jobs = [
            job
            for job in self._jobs()
            if find_source_subtitle(job.video_path) is not None
            and job.asr_subtitle_path.is_file()
        ]
        self.subtitle_listbox.delete(0, "end")
        for job in self.subtitle_jobs:
            source = find_source_subtitle(job.video_path)
            self.subtitle_listbox.insert(
                "end",
                f"{job.title}    [{source.name if source else ''} + "
                f"{job.asr_subtitle_path.name}]",
            )
        if self.subtitle_jobs:
            self.subtitle_listbox.selection_set(0)

    def _refresh_dub_jobs(self) -> None:
        self.dub_jobs = [
            job for job in self._jobs() if job.chinese_subtitle_path.is_file()
        ]
        self.dub_listbox.delete(0, "end")
        for job in self.dub_jobs:
            self.dub_listbox.insert(
                "end",
                f"{job.title}    [{job.chinese_subtitle_path.name}]",
            )
        if self.dub_jobs:
            self.dub_listbox.selection_set(0)

    @staticmethod
    def _select_all(listbox: tk.Listbox, jobs: list[VideoJob]) -> None:
        if jobs:
            listbox.selection_set(0, "end")

    def _select_all_asr_jobs(self) -> None:
        self._select_all(self.asr_listbox, self.asr_jobs)

    def _select_all_subtitle_jobs(self) -> None:
        self._select_all(self.subtitle_listbox, self.subtitle_jobs)

    def _select_all_dub_jobs(self) -> None:
        self._select_all(self.dub_listbox, self.dub_jobs)

    @staticmethod
    def _selected(listbox: tk.Listbox, jobs: list[VideoJob]) -> list[VideoJob]:
        return [
            jobs[index]
            for index in listbox.curselection()
            if index < len(jobs)
        ]

    def _urls(self) -> list[str]:
        return [
            line.strip()
            for line in self.url_text.get("1.0", "end").splitlines()
            if line.strip()
        ]

    def _make_config(self) -> AppConfig:
        config = load_config()
        config.work_dir = self.work_dir.get()
        config.link_type = self.link_type.get()
        config.subtitle_languages = self.subtitle_languages.get().strip()
        config.qwen_asr_enabled = bool(self.qwen_asr_enabled.get())
        config.qwen_tts_enabled = bool(self.qwen_tts_enabled.get())
        config.save_model_info = bool(self.save_model_info.get())
        config.normalize()
        config.ensure_directories()
        problems = config.validate_core()
        if problems:
            raise ValueError("\n".join(problems))
        save_config(config)
        self.config_data = config
        return config

    def _make_model_config(self) -> AppConfig:
        config = self._make_config()
        config.subtitle_api_base_url = self.subtitle_api_base_url.get().strip()
        config.subtitle_model = self.subtitle_model.get().strip()
        config.save_model_info = bool(self.save_model_info.get())
        if not config.subtitle_api_base_url:
            raise ValueError("请填写 OpenAI 兼容 API 地址。")
        if not config.subtitle_model:
            raise ValueError("请填写模型名称。")
        config.normalize()
        if config.save_model_info:
            save_config(config)
            self.config_data = config
        return config

    def _begin_task(self, task: str, status: str) -> None:
        self.runner.reset()
        self.current_task = task
        for button in self.start_buttons.values():
            button.configure(state="disabled")
        for key, button in self.stop_buttons.items():
            button.configure(state="normal" if key == task else "disabled")
        self.status.set(status)
        self._append_log(task, "=" * 64)

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
        self._begin_task("download", "正在下载…")
        self.worker = threading.Thread(
            target=self._download_worker,
            args=(config, urls),
            daemon=True,
        )
        self.worker.start()

    def _start_asr(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        jobs = self._selected(self.asr_listbox, self.asr_jobs)
        if not jobs:
            messagebox.showwarning("没有选择视频", "请选择至少一个视频。", parent=self)
            return
        if not self.qwen_asr_enabled.get():
            messagebox.showwarning(
                "字幕提取未启用",
                "请先启用 Qwen3-ASR 字幕提取。",
                parent=self,
            )
            return
        try:
            config = self._make_config()
        except Exception as exc:
            messagebox.showerror("设置错误", str(exc), parent=self)
            return
        self._begin_task("asr", "正在提取字幕…")
        self.worker = threading.Thread(
            target=self._asr_worker,
            args=(config, jobs),
            daemon=True,
        )
        self.worker.start()

    def _start_subtitle_workflow(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        jobs = self._selected(self.subtitle_listbox, self.subtitle_jobs)
        if not jobs:
            messagebox.showwarning(
                "没有配对字幕",
                "请先完成下载和 Qwen3-ASR 字幕提取，再刷新列表。",
                parent=self,
            )
            return
        try:
            config = self._make_model_config()
        except Exception as exc:
            messagebox.showerror("设置错误", str(exc), parent=self)
            return
        self._begin_task("subtitle", "正在比对、修复并翻译字幕…")
        api_key = api_key_from_runtime(self.subtitle_api_key.get())
        self.worker = threading.Thread(
            target=self._subtitle_worker,
            args=(config, jobs, api_key),
            daemon=True,
        )
        self.worker.start()

    def _start_dubbing(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        jobs = self._selected(self.dub_listbox, self.dub_jobs)
        if not jobs:
            messagebox.showwarning(
                "没有选择中文字幕",
                "请先完成字幕修复与翻译，再刷新列表。",
                parent=self,
            )
            return
        try:
            config = self._make_model_config()
            config.tts_provider = "qwen" if self.qwen_tts_enabled.get() else "edge"
        except Exception as exc:
            messagebox.showerror("设置错误", str(exc), parent=self)
            return
        self._begin_task("dub", "正在改写讲解字幕并生成中文配音…")
        api_key = api_key_from_runtime(self.subtitle_api_key.get())
        self.worker = threading.Thread(
            target=self._dub_worker,
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

    def _asr_worker(self, config: AppConfig, jobs: list[VideoJob]) -> None:
        try:
            for number, job in enumerate(jobs, 1):
                self.events.put(
                    ("status", f"正在提取字幕 {number}/{len(jobs)}：{job.title}")
                )
                extract_asr_subtitle(config, self.runner, job)
            self.events.put(("done", f"字幕提取完成，共处理 {len(jobs)} 个视频。"))
        except CancelledError:
            self.events.put(("cancelled", "字幕提取已停止"))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _subtitle_worker(
        self,
        config: AppConfig,
        jobs: list[VideoJob],
        api_key: str,
    ) -> None:
        try:
            workflow = SubtitleRepairWorkflow(config, self.runner, api_key=api_key)
            for number, job in enumerate(jobs, 1):
                self.events.put(
                    (
                        "status",
                        f"正在处理字幕 {number}/{len(jobs)}：{job.title}",
                    )
                )
                workflow.process_job(job)
            self.events.put(
                ("done", f"字幕修复与翻译完成，共处理 {len(jobs)} 个视频。")
            )
        except CancelledError:
            self.events.put(("cancelled", "字幕任务已停止"))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _dub_worker(
        self,
        config: AppConfig,
        jobs: list[VideoJob],
        api_key: str,
    ) -> None:
        try:
            speech_workflow = SpeechSubtitleWorkflow(
                config,
                self.runner,
                api_key=api_key,
            )
            outputs: list[Path] = []
            for number, job in enumerate(jobs, 1):
                self.events.put(
                    ("status", f"正在配音 {number}/{len(jobs)}：{job.title}")
                )
                speech_path = speech_workflow.process_job(job)
                output = dub_video(
                    config,
                    self.runner,
                    job,
                    speech_subtitle_path=speech_path,
                )
                outputs.append(output)
                self.runner.logger(f"中文配音视频：{output}")
            self.events.put(
                ("done", f"中文配音完成，共生成 {len(outputs)} 个视频。")
            )
        except CancelledError:
            self.events.put(("cancelled", "配音任务已停止"))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _stop_task(self) -> None:
        self.status.set("正在停止…")
        self.runner.cancel()

    def _append_log(self, key: str, text: str) -> None:
        log = self.logs.get(key)
        if log is None:
            return
        log.configure(state="normal")
        log.insert("end", text + "\n")
        log.see("end")
        log.configure(state="disabled")

    def _drain_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                text = str(payload)
                if event == "log":
                    self._append_log(self.current_task or "download", text)
                elif event == "status":
                    self.status.set(text)
                elif event == "asr_service":
                    self._apply_service_info("asr", payload)  # type: ignore[arg-type]
                elif event == "tts_service":
                    self._apply_service_info("tts", payload)  # type: ignore[arg-type]
                elif event == "done":
                    finished_task = self.current_task
                    self._finish(text)
                    self._refresh_all_jobs()
                    messagebox.showinfo("任务完成", text, parent=self)
                    if finished_task == "asr":
                        self.notebook.select(self.subtitle_tab)
                elif event == "cancelled":
                    self._finish(text)
                elif event == "error":
                    self._append_log(self.current_task or "download", "错误：" + text)
                    self._finish("任务失败")
                    messagebox.showerror("任务失败", text, parent=self)
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _finish(self, text: str) -> None:
        for button in self.start_buttons.values():
            button.configure(state="normal")
        for button in self.stop_buttons.values():
            button.configure(state="disabled")
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
