from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from videodub import __version__
from videodub.config import (
    AppConfig,
    PROJECT_ROOT,
    api_key_from_runtime,
    encrypt_api_key,
    load_config,
    migrate_legacy_layout,
    migrate_work_directory,
    save_config,
)
from videodub.downloader import (
    cleanup_new_download_directories,
    download,
    snapshot_download_directories,
)
from videodub.media import VideoJob, discover_video_jobs
from videodub.model_manager import (
    InstalledModel,
    choice_by_label,
    install_model,
    list_installed_models,
    model_choices,
    read_installed_model,
    uninstall_model,
)
from videodub.model_runtime import ManagedModelService
from videodub.platform_utils import open_in_file_manager
from videodub.qwen_speech import extract_asr_subtitle
from videodub.runner import CancelledError, ProcessRunner
from videodub.subtitle_workflow import SpeechSubtitleWorkflow, SubtitleRepairWorkflow
from videodub.subtitles import find_source_subtitle
from videodub.tts import dub_video


GITHUB_REPOSITORY = "Y2KdeLaplace/yt2bilibili"


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for item in value.lstrip("vV").split("."):
        digits = "".join(character for character in item if character.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


class ToolTip:
    def __init__(self, widget: tk.Widget, text_fn) -> None:
        self.widget = widget
        self.text_fn = text_fn
        self.window: tk.Toplevel | None = None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def show(self, _event: object = None) -> None:
        text = self.text_fn()
        if not text:
            return
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.geometry(f"+{self.widget.winfo_rootx()+12}+{self.widget.winfo_rooty()+30}")
        ttk.Label(self.window, text=text, padding=(7, 4), relief="solid").pack()

    def hide(self, _event: object = None) -> None:
        if self.window:
            self.window.destroy()
            self.window = None


class VideoDubApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("YouTube 视频中文化工具")
        self.geometry("1080x820")
        self.minsize(900, 700)
        self.config_data = load_config()
        migrate_legacy_layout(self.config_data.work_dir)
        self.config_data.ensure_directories()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.runner = ProcessRunner(lambda line: self.events.put(("log", line)))
        self.active_runners: list[ProcessRunner] = []
        self.runners_lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self.current_task = ""
        self.jobs: list[VideoJob] = []
        self.session_api_key = api_key_from_runtime(config=self.config_data)
        self.ui_font = tkfont.nametofont("TkDefaultFont").actual()["family"]
        self.mono_font = tkfont.nametofont("TkFixedFont").actual()["family"]

        self.work_dir = tk.StringVar(value=self.config_data.work_dir)
        self.link_type = tk.StringVar(value=self.config_data.link_type)
        self.subtitle_languages = tk.StringVar(value=self.config_data.subtitle_languages)
        self.stage_asr = tk.BooleanVar(value=True)
        self.stage_repair = tk.BooleanVar(value=True)
        self.stage_translate = tk.BooleanVar(value=True)
        self.stage_dub = tk.BooleanVar(value=True)
        self.parallel_enabled = tk.BooleanVar(value=False)
        self.parallel_count = tk.StringVar(value="2")

        self._configure_style()
        self._set_icon()
        self._build_ui()
        self.after(100, self._drain_events)
        self.after(180, self._check_tools)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        self.option_add("*Font", (self.ui_font, 10))
        style.configure("TLabelframe.Label", font=(self.ui_font, 11, "bold"))
        style.configure("TButton", font=(self.ui_font, 11), padding=(6, 2))
        style.configure("Toolbutton.TButton", font=(self.ui_font, 11), padding=(5, 2))
        style.configure("Stage.TCheckbutton", font=(self.ui_font, 11))
        style.configure("Treeview", rowheight=27)
        style.configure("Treeview.Heading", font=(self.ui_font, 10, "bold"))

    def _set_icon(self) -> None:
        icon = PROJECT_ROOT / "assets" / "app-icon.png"
        if icon.is_file():
            try:
                self._icon_image = tk.PhotoImage(file=icon)
                self.iconphoto(True, self._icon_image)
            except tk.TclError:
                self._icon_image = None

    def _center_dialog(
        self,
        dialog: tk.Toplevel,
        width: int,
        height: int,
    ) -> None:
        dialog.update_idletasks()
        x = max(0, (dialog.winfo_screenwidth() - width) // 2)
        y = max(0, (dialog.winfo_screenheight() - height) // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 9))
        ttk.Label(header, text="工作路径").pack(side="left")
        ttk.Entry(header, textvariable=self.work_dir, state="readonly").pack(
            side="left", fill="x", expand=True, padx=(9, 7)
        )
        ttk.Button(header, text="选择", command=self._browse_work_folder).pack(side="left")
        ttk.Button(header, text="打开", command=self._open_work_folder).pack(
            side="left", padx=(6, 12)
        )
        model_menu = tk.Menu(self, tearoff=False)
        model_menu.add_command(label="语言模型", command=self._show_language_model)
        model_menu.add_command(label="语音识别模型", command=lambda: self._show_model_dialog("asr"))
        model_menu.add_command(label="语音生成模型", command=lambda: self._show_model_dialog("tts"))
        ttk.Menubutton(header, text="模型", menu=model_menu).pack(side="left")
        about_menu = tk.Menu(self, tearoff=False)
        about_menu.add_command(label="更新", command=self._check_update)
        about_menu.add_command(label="版本", command=self._show_version)
        ttk.Menubutton(header, text="关于", menu=about_menu).pack(side="left", padx=(6, 0))

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        self.download_tab = ttk.Frame(self.notebook, padding=12)
        self.process_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.download_tab, text="视频下载")
        self.notebook.add(self.process_tab, text="处理")
        self.notebook.bind("<<NotebookTabChanged>>", lambda _e: self._refresh_jobs())
        self._build_download_tab()
        self._build_process_tab()

        log_box = ttk.LabelFrame(outer, text="运行日志", padding=7)
        log_box.pack(fill="both", pady=(10, 0))
        self.log = tk.Text(
            log_box,
            height=8,
            wrap="word",
            state="disabled",
            font=(self.mono_font, 9),
            background="#111827",
            foreground="#e5e7eb",
            insertbackground="#e5e7eb",
        )
        scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _build_download_tab(self) -> None:
        link_box = ttk.LabelFrame(self.download_tab, text="链接", padding=10)
        link_box.pack(fill="x")
        self.url_text = tk.Text(link_box, height=8, wrap="word")
        self.url_text.pack(fill="x")
        self.url_placeholder = ttk.Label(
            self.url_text,
            text="粘贴一个或多个 YouTube 视频或播放列表链接，每行一个",
            foreground="#808080",
        )
        self.url_placeholder.place(x=7, y=6)
        self.url_placeholder.bind("<Button-1>", lambda _e: self.url_text.focus_set())
        self.url_text.bind("<KeyRelease>", self._update_url_placeholder)
        self.url_text.bind("<FocusIn>", self._update_url_placeholder)
        self.url_text.bind("<Button-3>", self._show_url_menu)
        self.url_text.bind("<Button-2>", self._show_url_menu)

        settings = ttk.LabelFrame(self.download_tab, text="下载设置", padding=10)
        settings.pack(fill="x", pady=(10, 0))
        ttk.Radiobutton(settings, text="单个视频", variable=self.link_type, value="single").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Radiobutton(settings, text="播放列表", variable=self.link_type, value="playlist").grid(
            row=0, column=1, sticky="w", padx=(18, 0)
        )
        ttk.Label(settings, text="字幕语言").grid(row=1, column=0, sticky="w", pady=(9, 0))
        ttk.Entry(settings, textvariable=self.subtitle_languages).grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=(10, 0), pady=(9, 0)
        )
        settings.columnconfigure(3, weight=1)
        controls = ttk.Frame(self.download_tab)
        controls.pack(fill="x", pady=(10, 0))
        self.download_button = ttk.Button(controls, text="下载", command=self._start_download)
        self.download_button.pack(side="right")

    def _build_process_tab(self) -> None:
        selection = ttk.LabelFrame(self.process_tab, text="选择视频", padding=8)
        selection.pack(fill="x")
        ttk.Button(selection, text="刷新", command=self._refresh_jobs).pack(
            anchor="e", pady=(0, 5)
        )
        columns = ("video", "downloaded", "asr", "corrected", "chinese")
        self.job_tree = ttk.Treeview(
            selection,
            columns=columns,
            show="headings",
            selectmode="extended",
            height=10,
        )
        labels = {
            "video": "视频名",
            "downloaded": "下载字幕",
            "asr": "识别字幕",
            "corrected": "修复字幕",
            "chinese": "中文字幕",
        }
        for key in columns:
            self.job_tree.heading(key, text=labels[key])
            self.job_tree.column(key, width=105 if key != "video" else 470, anchor="center" if key != "video" else "w")
        scroll = ttk.Scrollbar(selection, orient="vertical", command=self.job_tree.yview)
        self.job_tree.configure(yscrollcommand=scroll.set)
        self.job_tree.bind("<Button-3>", self._show_tree_menu)
        self.job_tree.bind("<Button-2>", self._show_tree_menu)
        self.job_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        controls = ttk.Frame(self.process_tab)
        controls.pack(fill="x", pady=(10, 0))
        for text, variable in (
            ("提取", self.stage_asr),
            ("修复", self.stage_repair),
            ("翻译", self.stage_translate),
            ("配音", self.stage_dub),
        ):
            ttk.Checkbutton(
                controls,
                text=text,
                variable=variable,
                style="Stage.TCheckbutton",
            ).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(
            controls,
            text="并行处理",
            variable=self.parallel_enabled,
            command=self._update_parallel_state,
            style="Stage.TCheckbutton",
        ).pack(side="left", padx=(4, 7))
        self.parallel_entry = ttk.Entry(
            controls,
            textvariable=self.parallel_count,
            width=5,
            state="disabled",
            justify="center",
        )
        self.parallel_entry.pack(side="left")
        self.process_button = ttk.Button(controls, text="运行", command=self._start_processing)
        self.process_button.pack(side="right")
        self._refresh_jobs()

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _separator(self, title: str) -> None:
        self._append_log(f"== {title} " + "=" * max(4, 50 - len(title)))

    def _check_tools(self) -> None:
        problems = self.config_data.validate_core()
        if problems:
            self._append_log("\n".join(problems))
        else:
            self._append_log("yt-dlp、ffmpeg、ffprobe 检查通过。")

    def _update_url_placeholder(self, _event: object = None) -> None:
        if self.url_text.get("1.0", "end-1c").strip():
            self.url_placeholder.place_forget()
        else:
            self.url_placeholder.place(x=7, y=6)

    def _show_url_menu(self, event: tk.Event) -> str:
        self.url_text.focus_set()
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(
            label="粘贴",
            command=self._paste_url,
        )
        menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _paste_url(self) -> None:
        self.url_text.event_generate("<<Paste>>")
        self.after(1, self._update_url_placeholder)

    def _show_tree_menu(self, event: tk.Event) -> str:
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(
            label="全选",
            command=lambda: self.job_tree.selection_set(
                self.job_tree.get_children()
            ),
        )
        menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _update_parallel_state(self) -> None:
        self.parallel_entry.configure(
            state="normal" if self.parallel_enabled.get() else "disabled"
        )

    def _browse_work_folder(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.work_dir.get(), parent=self)
        if not selected:
            return
        try:
            migrate_work_directory(self.work_dir.get(), selected)
            self.config_data.work_dir = str(Path(selected).resolve())
            self._persist_config()
            self.work_dir.set(self.config_data.work_dir)
            self._append_log(f"工作路径已迁移：{self.config_data.work_dir}")
            self._refresh_jobs()
        except Exception as exc:
            messagebox.showerror("迁移失败", str(exc), parent=self)

    def _open_work_folder(self) -> None:
        try:
            open_in_file_manager(self.work_dir.get())
        except Exception as exc:
            messagebox.showerror("无法打开", str(exc), parent=self)

    def _sync_basic_config(self) -> AppConfig:
        self.config_data.work_dir = self.work_dir.get()
        self.config_data.link_type = self.link_type.get()
        self.config_data.subtitle_languages = self.subtitle_languages.get().strip()
        self.config_data.normalize()
        self.config_data.ensure_directories()
        problems = self.config_data.validate_core()
        if problems:
            raise ValueError("\n".join(problems))
        self._persist_config()
        return self.config_data

    def _persist_config(self) -> None:
        if self.config_data.save_model_info:
            save_config(self.config_data)
            return
        save_config(
            replace(
                self.config_data,
                subtitle_api_base_url="",
                subtitle_model="",
                subtitle_api_key_encrypted="",
            )
        )

    def _jobs(self) -> list[VideoJob]:
        return discover_video_jobs(self.config_data.work_dir, self.config_data.output_dir)

    def _refresh_jobs(self) -> None:
        if not hasattr(self, "job_tree"):
            return
        selected_paths = {
            str(self.jobs[int(item)].video_path)
            for item in self.job_tree.selection()
            if item.isdigit() and int(item) < len(self.jobs)
        }
        self.jobs = self._jobs()
        self.job_tree.delete(*self.job_tree.get_children())
        for index, job in enumerate(self.jobs):
            values = (
                job.title,
                "●" if find_source_subtitle(job.video_path) else "—",
                "●" if job.asr_subtitle_path.is_file() else "—",
                "●" if job.corrected_subtitle_path.is_file() else "—",
                "●" if job.chinese_subtitle_path.is_file() else "—",
            )
            item = self.job_tree.insert("", "end", iid=str(index), values=values)
            if str(job.video_path) in selected_paths:
                self.job_tree.selection_add(item)
        if self.jobs and not self.job_tree.selection():
            self.job_tree.selection_set("0")

    def _selected_jobs(self) -> list[VideoJob]:
        return [self.jobs[int(item)] for item in self.job_tree.selection() if item.isdigit()]

    def _set_running(self, running: bool, task: str = "") -> None:
        self.current_task = task if running else ""
        if running and task == "download":
            self.download_button.configure(
                text="停止",
                command=self._stop,
                state="normal",
            )
            self.process_button.configure(state="disabled")
        elif running and task == "process":
            self.process_button.configure(
                text="停止",
                command=self._stop,
                state="normal",
            )
            self.download_button.configure(state="disabled")
        else:
            self.download_button.configure(
                text="下载",
                command=self._start_download,
                state="normal",
            )
            self.process_button.configure(
                text="运行",
                command=self._start_processing,
                state="normal",
            )

    def _start_download(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        urls = [line.strip() for line in self.url_text.get("1.0", "end").splitlines() if line.strip()]
        if not urls:
            messagebox.showwarning("缺少链接", "请先输入 YouTube 链接。", parent=self)
            return
        try:
            config = self._sync_basic_config()
        except Exception as exc:
            messagebox.showerror("设置错误", str(exc), parent=self)
            return
        self._separator("YouTube 视频下载")
        self.runner.reset()
        self._set_running(True, "download")
        self.worker = threading.Thread(target=self._download_worker, args=(config, urls), daemon=True)
        self.worker.start()

    def _download_worker(self, config: AppConfig, urls: list[str]) -> None:
        try:
            for url in urls:
                self.runner.check_cancelled()
                before = snapshot_download_directories(config)
                try:
                    download(config, self.runner, url)
                except Exception:
                    cleanup_new_download_directories(
                        config,
                        before,
                        self.runner,
                    )
                    raise
            self.events.put(("done", "下载完成"))
        except CancelledError:
            self.events.put(("done", "下载已停止"))
        except Exception as exc:
            self.events.put(("error", exc))

    def _start_processing(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        jobs = self._selected_jobs()
        stages = (
            self.stage_asr.get(),
            self.stage_repair.get(),
            self.stage_translate.get(),
            self.stage_dub.get(),
        )
        if not jobs:
            messagebox.showwarning("没有选择视频", "请至少选择一个视频。", parent=self)
            return
        if not any(stages):
            messagebox.showwarning("没有任务", "请至少选择一个处理步骤。", parent=self)
            return
        try:
            config = self._sync_basic_config()
            if (stages[1] or stages[2] or stages[3]) and (
                not config.subtitle_api_base_url or not config.subtitle_model
            ):
                raise ValueError("请先在“模型 → 语言模型”中完成配置。")
            if stages[0] and not config.asr_model_path:
                raise ValueError("请先在“模型 → 语音识别模型”中安装并选择模型。")
            parallel = 1
            if self.parallel_enabled.get():
                raw_parallel = self.parallel_count.get().strip()
                if not raw_parallel.isdigit() or int(raw_parallel) <= 1:
                    raise ValueError("并行处理数量必须是大于 1 的正整数。")
                parallel = int(raw_parallel)
        except Exception as exc:
            messagebox.showerror("设置错误", str(exc), parent=self)
            return
        self._separator("字幕与中文配音")
        self.runner.reset()
        self._set_running(True, "process")
        self.worker = threading.Thread(
            target=self._processing_worker,
            args=(config, jobs, stages, parallel),
            daemon=True,
        )
        self.worker.start()

    def _processing_worker(
        self,
        config: AppConfig,
        jobs: list[VideoJob],
        stages: tuple[bool, bool, bool, bool],
        parallel: int,
    ) -> None:
        try:
            worker_count = min(parallel, len(jobs))
            if worker_count == 1:
                self._process_one_job(config, jobs[0], stages, 0)
                for index, job in enumerate(jobs[1:], 1):
                    self._process_one_job(config, job, stages, index)
            else:
                self.events.put(("log", f"并行处理：{worker_count} 个视频"))
                with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="videodub",
                ) as executor:
                    futures = [
                        executor.submit(
                            self._process_one_job,
                            config,
                            job,
                            stages,
                            index,
                        )
                        for index, job in enumerate(jobs)
                    ]
                    try:
                        for future in as_completed(futures):
                            future.result()
                    except Exception:
                        self._cancel_active_runners()
                        for future in futures:
                            future.cancel()
                        raise
            self.events.put(("done", "处理完成"))
        except CancelledError:
            self._cancel_active_runners()
            self.events.put(("done", "处理已停止"))
        except Exception as exc:
            self._cancel_active_runners()
            self.events.put(("error", exc))

    def _process_one_job(
        self,
        config: AppConfig,
        job: VideoJob,
        stages: tuple[bool, bool, bool, bool],
        slot: int,
    ) -> None:
        extract, repair, translate, dubbing = stages
        runner = ProcessRunner(
            lambda line: self.events.put(("log", f"[{job.title}] {line}"))
        )
        with self.runners_lock:
            self.active_runners.append(runner)
        try:
            runner.reset()
            with ExitStack() as stack:
                asr_url = ""
                if extract:
                    asr_service = stack.enter_context(
                        ManagedModelService(
                            config,
                            runner,
                            "asr",
                            port=12000 + slot * 2,
                        )
                    )
                    asr_url = asr_service.base_url
                    extract_asr_subtitle(
                        config,
                        runner,
                        job,
                        base_url=asr_url,
                    )
                if repair or translate:
                    SubtitleRepairWorkflow(
                        config,
                        runner,
                        api_key=self.session_api_key
                        or api_key_from_runtime(config=config),
                    ).process_job(job, repair=repair, translate=translate)
                if dubbing:
                    tts_url = ""
                    if config.tts_provider == "qwen":
                        tts_service = stack.enter_context(
                            ManagedModelService(
                                config,
                                runner,
                                "tts",
                                port=12001 + slot * 2,
                            )
                        )
                        tts_url = tts_service.base_url
                    speech = SpeechSubtitleWorkflow(
                        config,
                        runner,
                        api_key=self.session_api_key
                        or api_key_from_runtime(config=config),
                    ).process_job(job)
                    output = dub_video(
                        config,
                        runner,
                        job,
                        speech_subtitle_path=speech,
                        qwen_base_url=tts_url,
                    )
                    runner.logger(f"中文配音视频：{output}")
        finally:
            with self.runners_lock:
                if runner in self.active_runners:
                    self.active_runners.remove(runner)

    def _stop(self) -> None:
        self.runner.cancel()
        self._cancel_active_runners()
        self._append_log("正在停止当前任务…")

    def _cancel_active_runners(self) -> None:
        with self.runners_lock:
            runners = list(self.active_runners)
        for runner in runners:
            runner.cancel()

    def _show_language_model(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("语言模型")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill="both", expand=True)
        base = tk.StringVar(value=self.config_data.subtitle_api_base_url)
        key = tk.StringVar(value=self.session_api_key)
        model = tk.StringVar(value=self.config_data.subtitle_model)
        save = tk.BooleanVar(value=self.config_data.save_model_info)
        for row, (label, variable, show) in enumerate(
            (("API 地址", base, ""), ("API Key", key, "•"), ("模型名", model, ""))
        ):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Entry(frame, textvariable=variable, show=show, width=56).grid(
                row=row, column=1, sticky="ew", padx=(10, 0), pady=6
            )
        ttk.Checkbutton(frame, text="保存信息", variable=save).grid(
            row=3, column=1, sticky="w", padx=(10, 0), pady=(7, 10)
        )

        def commit() -> None:
            if not base.get().strip() or not model.get().strip():
                messagebox.showwarning("信息不完整", "请填写 API 地址和模型。", parent=dialog)
                return
            self.config_data.subtitle_api_base_url = base.get().strip()
            self.config_data.subtitle_model = model.get().strip()
            self.config_data.save_model_info = save.get()
            self.session_api_key = key.get().strip()
            self.config_data.subtitle_api_key_encrypted = (
                encrypt_api_key(self.session_api_key) if save.get() else ""
            )
            self._persist_config()
            dialog.destroy()

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="保存", command=commit).pack(side="left")
        self._center_dialog(dialog, 620, 250)

    def _show_model_dialog(self, kind: str) -> None:
        title = "语音识别模型" if kind == "asr" else "语音生成模型"
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.transient(self)
        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill="both", expand=True)
        choices = model_choices(kind)
        selected_model = tk.StringVar()
        install_choice = tk.StringVar(value=choices[0].label if choices else "")
        custom = tk.StringVar()
        installed_map: dict[str, InstalledModel] = {}
        locked = tk.BooleanVar(value=False)

        ttk.Label(frame, text="模型选择").grid(row=0, column=0, sticky="w")
        selected_combo = ttk.Combobox(
            frame,
            textvariable=selected_model,
            state="readonly",
        )
        selected_combo.grid(row=0, column=1, sticky="ew", padx=(10, 8))
        lock_button = ttk.Button(frame)
        lock_button.grid(row=0, column=2, sticky="e")
        ToolTip(
            selected_combo,
            lambda: installed_map.get(selected_model.get()).path
            if selected_model.get() in installed_map
            else "",
        )

        ttk.Label(frame, text="模型安装").grid(
            row=1,
            column=0,
            sticky="w",
            pady=(12, 0),
        )
        install_combo = ttk.Combobox(
            frame,
            textvariable=install_choice,
            values=[item.label for item in choices],
            state="readonly",
        )
        install_combo.grid(
            row=1,
            column=1,
            columnspan=2,
            sticky="ew",
            padx=(10, 0),
            pady=(12, 0),
        )
        custom_entry = ttk.Entry(frame, textvariable=custom, state="disabled")
        custom_entry.grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(10, 8),
            pady=(8, 0),
        )
        install_actions = ttk.Frame(frame)
        install_actions.grid(row=2, column=2, sticky="e", pady=(8, 0))
        install_button = ttk.Button(install_actions, text="安装")
        install_button.pack(side="left")
        uninstall_button = ttk.Button(install_actions, text="卸载")
        uninstall_button.pack(side="left", padx=(6, 0))

        log = tk.Text(
            frame,
            height=14,
            state="disabled",
            wrap="word",
            font=(self.mono_font, 9),
            background="#111827",
            foreground="#e5e7eb",
        )
        log.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(3, weight=1)

        def current_path() -> str:
            return (
                self.config_data.asr_model_path
                if kind == "asr"
                else self.config_data.tts_model_path
            )

        def refresh_lock_state() -> None:
            has_selection = selected_model.get() in installed_map
            selected_combo.configure(
                state="disabled" if locked.get() else "readonly"
            )
            lock_button.configure(
                text="解锁选择" if locked.get() else "锁定选择",
                state="normal" if has_selection else "disabled",
            )
            uninstall_button.configure(
                state="normal" if has_selection else "disabled"
            )

        def reload_installed(
            prefer_path: str = "",
            *,
            keep_locked: bool = True,
        ) -> None:
            installed_map.clear()
            labels: list[str] = []
            for item in list_installed_models(kind):
                label = f"{item.repo_id}（已安装）"
                labels.append(label)
                installed_map[label] = item
            selected_combo.configure(values=labels)
            target = prefer_path or current_path()
            matched = ""
            for label, item in installed_map.items():
                if Path(item.path) == Path(target):
                    matched = label
                    break
            selected_model.set(matched or (labels[0] if labels else ""))
            locked.set(bool(matched) if keep_locked else False)
            refresh_lock_state()

        def model_log(text: str) -> None:
            def write() -> None:
                if not dialog.winfo_exists():
                    return
                log.configure(state="normal")
                log.insert("end", text.rstrip() + "\n")
                log.see("end")
                log.configure(state="disabled")

            self.after(0, write)

        def update_custom(_event: object = None) -> None:
            choice = choice_by_label(kind, install_choice.get())
            custom_entry.configure(
                state="normal" if choice.key == "other" else "disabled"
            )

        install_combo.bind("<<ComboboxSelected>>", update_custom)
        selected_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: refresh_lock_state(),
        )

        def toggle_lock() -> None:
            if locked.get():
                locked.set(False)
                refresh_lock_state()
                return
            installed = installed_map.get(selected_model.get())
            if not installed:
                return
            if kind == "asr":
                self.config_data.asr_backend = installed.backend
                self.config_data.asr_model_id = installed.repo_id
                self.config_data.asr_model_path = installed.path
            else:
                self.config_data.tts_backend = installed.backend
                self.config_data.tts_model_id = installed.repo_id
                self.config_data.tts_model_path = installed.path
                self.config_data.tts_codec_path = installed.codec_path
                self.config_data.tts_provider = "qwen"
            self._persist_config()
            locked.set(True)
            refresh_lock_state()

        def install_selected() -> None:
            choice = choice_by_label(kind, install_choice.get())
            install_button.configure(state="disabled")

            def worker() -> None:
                local_runner = ProcessRunner(model_log)
                try:
                    installed = install_model(self.config_data, kind, choice, custom.get(), local_runner)
                    self.after(
                        0,
                        lambda: reload_installed(
                            installed.path,
                            keep_locked=False,
                        ),
                    )
                except Exception as exc:
                    model_log(f"安装失败：{exc}")
                finally:
                    self.after(
                        0,
                        lambda: install_button.configure(state="normal")
                        if dialog.winfo_exists()
                        else None,
                    )

            threading.Thread(target=worker, daemon=True).start()

        def uninstall_selected() -> None:
            installed = installed_map.get(selected_model.get())
            if not installed:
                return
            if not messagebox.askyesno("确认卸载", f"删除模型 {installed.repo_id}？", parent=dialog):
                return
            try:
                uninstall_model(installed, ProcessRunner(model_log))
                if kind == "asr" and Path(self.config_data.asr_model_path) == Path(installed.path):
                    self.config_data.asr_backend = ""
                    self.config_data.asr_model_id = ""
                    self.config_data.asr_model_path = ""
                elif kind == "tts" and Path(self.config_data.tts_model_path) == Path(installed.path):
                    self.config_data.tts_backend = ""
                    self.config_data.tts_model_id = ""
                    self.config_data.tts_model_path = ""
                    self.config_data.tts_codec_path = ""
                    self.config_data.tts_provider = "edge"
                self._persist_config()
                reload_installed()
            except Exception as exc:
                messagebox.showerror("卸载失败", str(exc), parent=dialog)

        lock_button.configure(command=toggle_lock)
        install_button.configure(command=install_selected)
        uninstall_button.configure(command=uninstall_selected)
        update_custom()
        reload_installed()
        self._center_dialog(dialog, 780, 510)

    def _show_version(self) -> None:
        messagebox.showinfo(
            "版本",
            f"YouTube 视频中文化工具\n版本 {__version__}\n\nGitHub：{GITHUB_REPOSITORY}",
            parent=self,
        )

    def _check_update(self) -> None:
        self._separator("检查更新")

        def worker() -> None:
            try:
                headers = {"User-Agent": f"youtube-video-localizer/{__version__}"}
                try:
                    request = urllib.request.Request(
                        f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/latest",
                        headers=headers,
                    )
                    with urllib.request.urlopen(request, timeout=15) as response:
                        data = json.loads(response.read().decode("utf-8"))
                    latest = str(data.get("tag_name") or "").lstrip("v")
                    url = str(data.get("html_url") or "")
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
                    request = urllib.request.Request(
                        f"https://api.github.com/repos/{GITHUB_REPOSITORY}/tags?per_page=1",
                        headers=headers,
                    )
                    with urllib.request.urlopen(request, timeout=15) as response:
                        tags = json.loads(response.read().decode("utf-8"))
                    latest = str(tags[0].get("name") or "").lstrip("v") if tags else ""
                    url = f"https://github.com/{GITHUB_REPOSITORY}/tags"
                if not latest:
                    raise RuntimeError("GitHub 尚未发布版本标签")
                self.events.put(("update", (latest, url)))
            except (OSError, ValueError, urllib.error.URLError) as exc:
                self.events.put(("error", RuntimeError(f"检查更新失败：{exc}")))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    self._append_log(str(payload))
                    self._set_running(False)
                    self._refresh_jobs()
                elif kind == "error":
                    self._append_log(f"错误：{payload}")
                    self._set_running(False)
                    messagebox.showerror("任务失败", str(payload), parent=self)
                elif kind == "update":
                    latest, url = payload
                    if latest and _version_tuple(latest) > _version_tuple(__version__):
                        messagebox.showinfo(
                            "发现新版本",
                            f"当前版本：{__version__}\n最新版本：{latest}\n{url}",
                            parent=self,
                        )
                    else:
                        messagebox.showinfo("更新", f"当前已是最新版本 {__version__}。", parent=self)
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("退出", "任务仍在运行，停止并退出？", parent=self):
                return
            self.runner.cancel()
        try:
            self._persist_config()
        finally:
            self.destroy()


def main() -> None:
    VideoDubApp().mainloop()


if __name__ == "__main__":
    main()
