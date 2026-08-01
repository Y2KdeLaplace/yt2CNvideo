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
    SUPPORTED_LANGUAGES,
    api_key_from_runtime,
    configure_cache_directory,
    encrypt_api_key,
    load_language_model_info,
    load_config,
    migrate_cache_directory,
    migrate_work_directory,
    save_config,
    save_language_model_info,
)
from videodub.downloader import (
    cleanup_new_download_directories,
    download,
    snapshot_download_directories,
)
from videodub.media import VideoJob, discover_video_jobs
from videodub.model_manager import (
    InstalledModel,
    ModelFileOption,
    choice_by_label,
    install_model,
    list_huggingface_gguf_options,
    list_installed_models,
    model_choices,
    read_installed_model,
    uninstall_model,
)
from videodub.model_runtime import ManagedModelService
from videodub.platform_utils import open_in_file_manager
from videodub.qwen_speech import (
    TTS_VOICE_PRESETS,
    extract_asr_subtitle,
    resolve_tts_reference,
)
from videodub.runner import CancelledError, ProcessRunner
from videodub.subtitle_workflow import SpeechSubtitleWorkflow, SubtitleRepairWorkflow
from videodub.subtitles import find_source_subtitle
from videodub.tts import dub_video


GITHUB_REPOSITORY = "Y2KdeLaplace/yt2CNvideo"


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
        # Keep the initial top-left placement hidden until the final geometry
        # is known, especially on macOS where Tk may paint before construction.
        self.withdraw()
        self.title("scip - YouTube 视频中文化工具")
        self.geometry("1080x738")
        self.minsize(900, 630)
        self.config_data = load_config()
        load_language_model_info(self.config_data)
        configure_cache_directory(self.config_data.cache_dir)
        self.config_data.ensure_directories()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.runner = ProcessRunner(lambda line: self.events.put(("log", line)))
        self.active_runners: list[ProcessRunner] = []
        self.model_download_runner: ProcessRunner | None = None
        self.runners_lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self.current_task = ""
        self.jobs: list[VideoJob] = []
        self.model_lock_state = {"asr": False, "tts": False}
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
        language_labels = {value: label for label, value in SUPPORTED_LANGUAGES.items()}
        self.asr_language = tk.StringVar(
            value=language_labels[self.config_data.asr_language]
        )
        self.translation_language = tk.StringVar(
            value=language_labels[self.config_data.translation_language]
        )
        self.tts_language = tk.StringVar(
            value=language_labels[self.config_data.tts_language]
        )
        self.parallel_enabled = tk.BooleanVar(value=False)
        self.parallel_count = tk.StringVar(value="2")

        self._configure_style()
        self._set_icon()
        self._build_ui()
        self._center_main_window()
        self.deiconify()
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
        style.configure("Main.TButton", font=(self.ui_font, 10), padding=(5, 1))
        style.configure("Main.TMenubutton", font=(self.ui_font, 10), padding=(5, 1))
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

    def _center_main_window(self) -> None:
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = max(0, (self.winfo_screenwidth() - width) // 2)
        y = max(0, (self.winfo_screenheight() - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 9))
        ttk.Label(header, text="工作路径").pack(side="left")
        ttk.Entry(header, textvariable=self.work_dir, state="readonly").pack(
            side="left", fill="x", expand=True, padx=(9, 7)
        )
        ttk.Button(
            header,
            text="选择",
            command=self._browse_work_folder,
            style="Main.TButton",
        ).pack(side="left")
        ttk.Button(
            header,
            text="打开",
            command=self._open_work_folder,
            style="Main.TButton",
        ).pack(
            side="left", padx=(6, 12)
        )
        model_menu = tk.Menu(self, tearoff=False)
        model_menu.add_command(label="语言模型", command=self._show_language_model)
        model_menu.add_command(label="语音识别模型", command=lambda: self._show_model_dialog("asr"))
        model_menu.add_command(label="语音生成模型", command=lambda: self._show_model_dialog("tts"))
        ttk.Menubutton(
            header,
            text="模型",
            menu=model_menu,
            style="Main.TMenubutton",
        ).pack(side="left")
        about_menu = tk.Menu(self, tearoff=False)
        about_menu.add_command(label="更新", command=self._check_update)
        about_menu.add_command(label="版本", command=self._show_version)
        cache_menu = tk.Menu(about_menu, tearoff=False)
        cache_menu.add_command(label="设置缓存目录", command=self._set_cache_directory)
        cache_menu.add_command(label="打开缓存目录", command=self._open_cache_directory)
        about_menu.add_cascade(label="缓存目录", menu=cache_menu)
        ttk.Menubutton(
            header,
            text="关于",
            menu=about_menu,
            style="Main.TMenubutton",
        ).pack(side="left", padx=(6, 0))

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="x")
        self.download_tab = ttk.Frame(self.notebook, padding=12)
        self.process_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.download_tab, text="视频下载")
        self.notebook.add(self.process_tab, text="处理")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._build_download_tab()
        self._build_process_tab()
        self.after_idle(self._resize_notebook_to_current_tab)

        log_box = ttk.LabelFrame(outer, text="运行日志", padding=7)
        log_box.pack(fill="both", expand=True, pady=(10, 0))
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

    def _on_tab_changed(self, _event: object = None) -> None:
        self._refresh_jobs()
        if (
            hasattr(self, "job_tree")
            and self.notebook.select() == str(self.process_tab)
        ):
            self.job_tree.selection_remove(*self.job_tree.selection())
        self.after_idle(self._resize_notebook_to_current_tab)
        # Do not leave the first control (the Refresh button on macOS) as the
        # key-window default when entering the processing page.
        self.after_idle(self.focus_set)

    def _resize_notebook_to_current_tab(self) -> None:
        selected = self.notebook.select()
        if not selected:
            return
        tab = self.nametowidget(selected)
        tab.update_idletasks()
        self.notebook.configure(height=tab.winfo_reqheight())

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
        self.download_button = ttk.Button(
            controls,
            text="下载",
            command=self._start_download,
            style="Main.TButton",
        )
        self.download_button.pack(side="right")

    def _build_process_tab(self) -> None:
        selection = ttk.LabelFrame(self.process_tab, text="选择视频", padding=8)
        selection.pack(fill="x")
        ttk.Button(
            selection,
            text="刷新",
            command=self._refresh_jobs,
            style="Main.TButton",
            takefocus=False,
        ).pack(
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
            "chinese": "翻译字幕",
        }
        for key in columns:
            self.job_tree.heading(key, text=labels[key])
            self.job_tree.column(key, width=105 if key != "video" else 470, anchor="center" if key != "video" else "w")
        scroll = ttk.Scrollbar(selection, orient="vertical", command=self.job_tree.yview)
        self.job_tree.configure(yscrollcommand=scroll.set)
        self.job_tree.bind("<Button-1>", self._clear_job_selection_on_blank)
        self.job_tree.bind("<Button-3>", self._show_tree_menu)
        self.job_tree.bind("<Button-2>", self._show_tree_menu)
        self.job_tree.bind("<Control-Button-1>", self._toggle_job_selection)
        self.job_tree.bind("<Command-Button-1>", self._toggle_job_selection)
        self.job_tree.bind("<Control-a>", self._select_all_jobs)
        self.job_tree.bind("<Control-A>", self._select_all_jobs)
        self.job_tree.bind("<Command-a>", self._select_all_jobs)
        self.job_tree.bind("<Command-A>", self._select_all_jobs)
        self.job_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        controls = ttk.Frame(self.process_tab)
        controls.pack(fill="x", pady=(10, 0))
        self.extract_check = ttk.Checkbutton(
            controls,
            text="提取",
            variable=self.stage_asr,
            command=self._update_stage_language_states,
            style="Stage.TCheckbutton",
        )
        self.extract_check.pack(side="left")
        self.asr_language_combo = ttk.Combobox(
            controls,
            textvariable=self.asr_language,
            values=tuple(SUPPORTED_LANGUAGES),
            state="readonly",
            width=7,
        )
        self.asr_language_combo.pack(side="left", padx=(5, 14))
        ttk.Checkbutton(
            controls,
            text="修复",
            variable=self.stage_repair,
            style="Stage.TCheckbutton",
        ).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(
            controls,
            text="翻译",
            variable=self.stage_translate,
            command=self._update_stage_language_states,
            style="Stage.TCheckbutton",
        ).pack(side="left")
        self.translation_language_combo = ttk.Combobox(
            controls,
            textvariable=self.translation_language,
            values=tuple(SUPPORTED_LANGUAGES),
            state="readonly",
            width=7,
        )
        self.translation_language_combo.pack(side="left", padx=(5, 14))
        ttk.Checkbutton(
            controls,
            text="配音",
            variable=self.stage_dub,
            command=self._update_stage_language_states,
            style="Stage.TCheckbutton",
        ).pack(side="left")
        self.tts_language_combo = ttk.Combobox(
            controls,
            textvariable=self.tts_language,
            values=tuple(SUPPORTED_LANGUAGES),
            state="readonly",
            width=7,
        )
        self.tts_language_combo.pack(side="left", padx=(5, 14))
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
        self.process_button = ttk.Button(
            controls,
            text="运行",
            command=self._start_processing,
            style="Main.TButton",
        )
        self.process_button.pack(side="right")
        self._update_stage_language_states()
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

    def _update_stage_language_states(self) -> None:
        for widget, enabled in (
            (self.asr_language_combo, self.stage_asr.get()),
            (self.translation_language_combo, self.stage_translate.get()),
            (self.tts_language_combo, self.stage_dub.get()),
        ):
            widget.configure(state="readonly" if enabled else "disabled")

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
        self.config_data.asr_language = SUPPORTED_LANGUAGES[self.asr_language.get()]
        self.config_data.translation_language = SUPPORTED_LANGUAGES[
            self.translation_language.get()
        ]
        self.config_data.tts_language = SUPPORTED_LANGUAGES[self.tts_language.get()]
        self.config_data.normalize()
        self.config_data.ensure_directories()
        problems = self.config_data.validate_core()
        if problems:
            raise ValueError("\n".join(problems))
        self._persist_config()
        return self.config_data

    def _persist_config(self) -> None:
        save_language_model_info(self.config_data)
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

    def _selected_jobs(self) -> list[VideoJob]:
        return [self.jobs[int(item)] for item in self.job_tree.selection() if item.isdigit()]

    def _clear_job_selection_on_blank(self, event: tk.Event) -> str | None:
        if self.job_tree.identify_row(event.y):
            return None
        self.job_tree.selection_remove(*self.job_tree.selection())
        return "break"

    def _toggle_job_selection(self, event: tk.Event) -> str:
        item = self.job_tree.identify_row(event.y)
        if not item:
            return "break"
        if item in self.job_tree.selection():
            self.job_tree.selection_remove(item)
        else:
            self.job_tree.selection_add(item)
            self.job_tree.focus(item)
        return "break"

    def _select_all_jobs(self, _event: tk.Event | None = None) -> str:
        if self.job_tree.selection():
            self.job_tree.selection_set(self.job_tree.get_children())
        return "break"

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
            if stages[0] and (
                not self.model_lock_state["asr"] or not config.asr_model_path
            ):
                raise ValueError("请先在“模型 → 语音识别模型”中选择并锁定模型。")
            if stages[3] and (
                not self.model_lock_state["tts"] or not config.tts_model_path
            ):
                raise ValueError("请先在“模型 → 语音生成模型”中选择并锁定模型。")
            if stages[3]:
                tts_model = read_installed_model(config.tts_model_path)
                if tts_model and tts_model.variant == "base":
                    resolve_tts_reference(config)
            parallel = 1
            if self.parallel_enabled.get():
                raw_parallel = self.parallel_count.get().strip()
                if not raw_parallel.isdigit() or int(raw_parallel) <= 1:
                    raise ValueError("并行处理数量必须是大于 1 的正整数。")
                parallel = int(raw_parallel)
        except Exception as exc:
            messagebox.showerror("设置错误", str(exc), parent=self)
            return
        self._separator("字幕与配音")
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
                        language=config.asr_language,
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
                    runner.logger(f"配音视频：{output}")
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
        ttk.Checkbutton(
            frame,
            text="保存信息",
            variable=save,
        ).grid(
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
        dialog.update_idletasks()
        self._center_dialog(
            dialog,
            dialog.winfo_reqwidth(),
            dialog.winfo_reqheight(),
        )

    def _show_model_dialog(self, kind: str) -> None:
        title = "语音识别模型" if kind == "asr" else "语音生成模型"
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.transient(self)
        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill="both", expand=True)
        choices = model_choices(kind)
        selected_model = tk.StringVar()
        installed_map: dict[str, InstalledModel] = {}
        locked = tk.BooleanVar(value=self.model_lock_state[kind])
        download_state: dict[str, object] = {
            "busy": False,
            "runner": None,
            "installer": None,
        }
        reference_audio = tk.StringVar(
            value=self.config_data.tts_reference_audio
        )
        reference_text_file = tk.StringVar(
            value=self.config_data.tts_reference_text_file
        )
        voice_preset = tk.StringVar(value=self.config_data.tts_voice_preset)
        use_custom_voice = tk.BooleanVar(
            value=self.config_data.tts_use_custom_voice
        )

        ttk.Label(frame, text="模型选择").grid(row=0, column=0, sticky="w")
        selected_combo = ttk.Combobox(
            frame,
            textvariable=selected_model,
            state="readonly",
        )
        selected_combo.grid(row=0, column=1, sticky="ew", padx=(10, 8))
        install_open_button = ttk.Button(frame, text="管理")
        install_open_button.grid(row=0, column=2, sticky="e", padx=(0, 8))
        lock_button = ttk.Button(frame)
        lock_button.grid(row=0, column=3, sticky="e")
        ToolTip(
            selected_combo,
            lambda: installed_map.get(selected_model.get()).path
            if locked.get() and selected_model.get() in installed_map
            else "",
        )

        reference_frame = ttk.Frame(frame)
        reference_frame.grid(
            row=1,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(10, 0),
        )
        ttk.Label(reference_frame, text="预设声音").grid(
            row=0,
            column=0,
            sticky="w",
        )
        voice_preset_combo = ttk.Combobox(
            reference_frame,
            textvariable=voice_preset,
            values=tuple(TTS_VOICE_PRESETS),
            state="readonly",
        )
        voice_preset_combo.grid(row=0, column=1, sticky="ew", padx=(10, 8))
        custom_voice_check = ttk.Checkbutton(
            reference_frame,
            text="使用自定义声音",
            variable=use_custom_voice,
        )
        custom_voice_check.grid(row=0, column=2, sticky="e")

        custom_reference_frame = ttk.Frame(reference_frame)
        custom_reference_frame.grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(9, 0),
        )
        ttk.Label(custom_reference_frame, text="参考音频 WAV").grid(
            row=0,
            column=0,
            sticky="w",
        )
        reference_entry = ttk.Entry(
            custom_reference_frame,
            textvariable=reference_audio,
            state="readonly",
        )
        reference_entry.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(10, 8),
        )
        reference_browse = ttk.Button(custom_reference_frame, text="选择")
        reference_browse.grid(row=0, column=2, sticky="e")
        ttk.Label(custom_reference_frame, text="对应文本文件").grid(
            row=1,
            column=0,
            sticky="w",
            pady=(8, 0),
        )
        reference_text_entry = ttk.Entry(
            custom_reference_frame,
            textvariable=reference_text_file,
            state="readonly",
        )
        reference_text_entry.grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(10, 8),
            pady=(8, 0),
        )
        reference_text_browse = ttk.Button(
            custom_reference_frame,
            text="选择",
        )
        reference_text_browse.grid(row=1, column=2, sticky="e", pady=(8, 0))
        custom_reference_frame.columnconfigure(1, weight=1)
        reference_frame.columnconfigure(1, weight=1)
        reference_frame.grid_remove()

        log = tk.Text(
            frame,
            height=14,
            state="disabled",
            wrap="word",
            font=(self.mono_font, 9),
            background="#111827",
            foreground="#e5e7eb",
        )
        log.grid(row=2, column=0, columnspan=4, sticky="nsew", pady=(12, 0))
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)

        def current_path() -> str:
            return (
                self.config_data.asr_model_path
                if kind == "asr"
                else self.config_data.tts_model_path
            )

        def selected_installed() -> InstalledModel | None:
            return installed_map.get(selected_model.get())

        def apply_installed(installed: InstalledModel) -> None:
            if kind == "asr":
                self.config_data.asr_backend = installed.backend
                self.config_data.asr_model_id = installed.repo_id
                self.config_data.asr_model_path = installed.path
            else:
                self.config_data.tts_backend = installed.backend
                self.config_data.tts_model_id = installed.repo_id
                self.config_data.tts_model_path = installed.path
                self.config_data.tts_codec_path = installed.codec_path
            self._persist_config()

        def save_reference() -> None:
            if kind != "tts":
                return
            self.config_data.tts_reference_audio = (
                reference_audio.get().strip()
            )
            self.config_data.tts_reference_text_file = (
                reference_text_file.get().strip()
            )
            self.config_data.tts_voice_preset = voice_preset.get().strip()
            self.config_data.tts_use_custom_voice = use_custom_voice.get()
            self.config_data.tts_reference_text = ""
            self._persist_config()

        def refresh_custom_voice_state() -> None:
            voice_preset_combo.configure(
                state="disabled" if use_custom_voice.get() else "readonly"
            )
            if use_custom_voice.get():
                custom_reference_frame.grid()
            else:
                custom_reference_frame.grid_remove()
            save_reference()

        def refresh_reference_state() -> None:
            installed = selected_installed()
            show = (
                kind == "tts"
                and locked.get()
                and installed is not None
                and installed.variant == "base"
            )
            if show:
                reference_frame.grid()
                refresh_custom_voice_state()
            else:
                reference_frame.grid_remove()

        def refresh_lock_state() -> None:
            has_selection = selected_installed() is not None
            selected_combo.configure(
                state="disabled" if locked.get() else "readonly"
            )
            install_open_button.configure(
                state="disabled" if locked.get() else "normal"
            )
            lock_button.configure(
                text="解锁" if locked.get() else "锁定",
                state="normal" if has_selection else "disabled",
            )
            refresh_reference_state()

        def reload_installed(
            prefer_path: str = "",
            *,
            keep_locked: bool = True,
        ) -> None:
            installed_map.clear()
            labels: list[str] = []
            for item in list_installed_models(kind):
                source = {
                    "modelscope": "ModelScope",
                    "huggingface": "Hugging Face",
                }.get(item.source, item.source)
                label = f"{item.repo_id}（{source}）"
                if item.backend == "gguf" and Path(item.path).is_file():
                    label += f" / {Path(item.path).name}"
                labels.append(label)
                installed_map[label] = item
            selected_combo.configure(values=labels)
            target = prefer_path or current_path()
            target_id = (
                self.config_data.asr_model_id
                if kind == "asr"
                else self.config_data.tts_model_id
            )
            matched = ""
            for label, item in installed_map.items():
                if Path(item.path) == Path(target):
                    matched = label
                    break
            if not matched and target_id:
                for label, item in installed_map.items():
                    if item.repo_id == target_id:
                        matched = label
                        break
            selected_model.set(matched or (labels[0] if labels else ""))
            locked.set(bool(matched) if keep_locked else False)
            self.model_lock_state[kind] = locked.get()
            if matched and keep_locked:
                apply_installed(installed_map[matched])
            refresh_lock_state()

        progress_line_active = False

        def write_model_log(text: str, *, force_progress: bool = False) -> None:
            def write() -> None:
                nonlocal progress_line_active
                if not dialog.winfo_exists():
                    return
                message = text.rstrip()
                is_progress = force_progress or message.startswith("下载进度（")
                log.configure(state="normal")
                if is_progress and progress_line_active:
                    log.delete("model_progress_start", "end-1c")
                elif is_progress:
                    log.mark_set("model_progress_start", "end-1c")
                    log.mark_gravity("model_progress_start", "left")
                elif progress_line_active:
                    log.mark_unset("model_progress_start")
                log.insert("end", message + "\n")
                progress_line_active = is_progress
                log.see("end")
                log.configure(state="disabled")

            self.after(0, write)

        def model_log(text: str) -> None:
            write_model_log(text)

        def model_progress_log(text: str) -> None:
            write_model_log(text, force_progress=True)

        selected_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: refresh_lock_state(),
        )

        def toggle_lock() -> None:
            if locked.get():
                save_reference()
                locked.set(False)
                self.model_lock_state[kind] = False
                refresh_lock_state()
                return
            installed = selected_installed()
            if not installed:
                return
            apply_installed(installed)
            locked.set(True)
            self.model_lock_state[kind] = True
            refresh_lock_state()

        def clear_uninstalled(installed: InstalledModel) -> None:
            if kind == "asr" and Path(
                self.config_data.asr_model_path
            ) == Path(installed.path):
                self.config_data.asr_backend = ""
                self.config_data.asr_model_id = ""
                self.config_data.asr_model_path = ""
                self.model_lock_state["asr"] = False
            elif kind == "tts" and Path(
                self.config_data.tts_model_path
            ) == Path(installed.path):
                self.config_data.tts_backend = ""
                self.config_data.tts_model_id = ""
                self.config_data.tts_model_path = ""
                self.config_data.tts_codec_path = ""
                self.model_lock_state["tts"] = False
            self._persist_config()

        def browse_reference() -> None:
            selected = filedialog.askopenfilename(
                parent=dialog,
                title="选择参考音频",
                filetypes=(("WAV 音频", "*.wav"), ("所有文件", "*.*")),
            )
            if selected:
                reference_audio.set(selected)
                save_reference()

        def browse_reference_text() -> None:
            selected = filedialog.askopenfilename(
                parent=dialog,
                title="选择对应文本文件",
                filetypes=(("所有文件", "*.*"),),
            )
            if selected:
                reference_text_file.set(selected)
                save_reference()

        def show_install_dialog() -> None:
            installer = tk.Toplevel(dialog)
            installer.title(f"{title}下载")
            installer.transient(dialog)
            installer.grab_set()
            download_state["installer"] = installer
            install_frame = ttk.Frame(installer, padding=14)
            install_frame.pack(fill="both", expand=True)
            install_choice = tk.StringVar(
                value=choices[0].label if choices else ""
            )
            custom = tk.StringVar()

            ttk.Label(install_frame, text="模型").grid(
                row=0,
                column=0,
                sticky="w",
            )
            install_combo = ttk.Combobox(
                install_frame,
                textvariable=install_choice,
                values=[item.label for item in choices],
                state="readonly",
            )
            install_combo.grid(
                row=0,
                column=1,
                sticky="ew",
                padx=(10, 0),
            )
            ttk.Label(install_frame, text="模型名称").grid(
                row=1,
                column=0,
                sticky="w",
                pady=(8, 0),
            )
            custom_entry = ttk.Entry(
                install_frame,
                textvariable=custom,
                state="disabled",
            )
            custom_entry.grid(
                row=1,
                column=1,
                sticky="ew",
                padx=(10, 0),
                pady=(8, 0),
            )
            ttk.Label(install_frame, text="已安装模型").grid(
                row=2,
                column=0,
                columnspan=2,
                sticky="w",
                pady=(12, 4),
            )
            installed_tree = ttk.Treeview(
                install_frame,
                columns=("model", "backend", "path"),
                show="headings",
                height=6,
            )
            installed_tree.heading("model", text="模型")
            installed_tree.heading("backend", text="类型")
            installed_tree.heading("path", text="位置")
            installed_tree.column("model", width=230, stretch=False)
            installed_tree.column("backend", width=75, stretch=False)
            installed_tree.column("path", width=330, stretch=True)
            installed_tree.grid(
                row=3,
                column=0,
                columnspan=2,
                sticky="nsew",
            )
            actions = ttk.Frame(install_frame)
            actions.grid(
                row=4,
                column=0,
                columnspan=2,
                sticky="e",
                pady=(10, 0),
            )
            download_button = ttk.Button(actions, text="安装")
            download_button.pack(side="left")
            uninstall_button = ttk.Button(actions, text="卸载")
            uninstall_button.pack(side="left", padx=(6, 0))
            install_frame.columnconfigure(1, weight=1)
            install_frame.rowconfigure(3, weight=1)
            installed_rows: dict[str, InstalledModel] = {}

            def refresh_installed_table(prefer_path: str = "") -> None:
                installed_rows.clear()
                installed_tree.delete(*installed_tree.get_children())
                for index, item in enumerate(list_installed_models(kind)):
                    row_id = str(index)
                    installed_rows[row_id] = item
                    installed_tree.insert(
                        "",
                        "end",
                        iid=row_id,
                        values=(item.repo_id, item.backend, item.path),
                    )
                    if prefer_path and Path(item.path) == Path(prefer_path):
                        installed_tree.selection_set(row_id)

            def selected_table_model() -> InstalledModel | None:
                selected = installed_tree.selection()
                return installed_rows.get(selected[0]) if selected else None

            def update_custom(_event: object = None) -> None:
                choice = choice_by_label(kind, install_choice.get())
                custom_entry.configure(
                    state="normal" if choice.key == "other" else "disabled"
                )

            def set_download_finished(runner: ProcessRunner) -> None:
                if download_state.get("runner") is runner:
                    download_state["busy"] = False
                    download_state["runner"] = None
                if self.model_download_runner is runner:
                    self.model_download_runner = None

            def restore_buttons() -> None:
                if not installer.winfo_exists():
                    return
                download_button.configure(state="normal")
                uninstall_button.configure(
                    state=(
                        "normal"
                        if selected_table_model()
                        else "disabled"
                    )
                )

            def finish_download(
                installed: InstalledModel,
                runner: ProcessRunner,
            ) -> None:
                set_download_finished(runner)
                if not dialog.winfo_exists():
                    return
                reload_installed(installed.path, keep_locked=False)
                refresh_installed_table(installed.path)
                if installer.winfo_exists():
                    installer.destroy()

            def fail_download(
                error: Exception,
                runner: ProcessRunner,
            ) -> None:
                set_download_finished(runner)
                model_log(f"下载失败：{error}")
                restore_buttons()

            def run_download(
                choice,
                repo_id: str,
                selected_files: tuple[str, ...],
                runner: ProcessRunner,
            ) -> None:
                def worker() -> None:
                    try:
                        installed = install_model(
                            self.config_data,
                            kind,
                            choice,
                            repo_id if choice.key == "other" else "",
                            runner,
                            selected_files,
                        )
                        self.after(
                            0,
                            lambda: finish_download(installed, runner),
                        )
                    except Exception as exc:
                        self.after(
                            0,
                            lambda error=exc: fail_download(error, runner),
                        )

                threading.Thread(target=worker, daemon=True).start()

            def show_file_selection(
                options: tuple[ModelFileOption, ...],
                on_selected,
                on_cancel,
            ) -> None:
                chooser = tk.Toplevel(installer)
                chooser.title("选择模型版本")
                chooser.transient(installer)
                chooser.grab_set()
                chooser_frame = ttk.Frame(chooser, padding=14)
                chooser_frame.pack(fill="both", expand=True)
                selected_label = tk.StringVar(value=options[0].label)
                ttk.Label(
                    chooser_frame,
                    text="检测到多个 GGUF 版本，请选择要下载的模型：",
                ).pack(anchor="w")
                option_map = {item.label: item for item in options}
                version_combo = ttk.Combobox(
                    chooser_frame,
                    textvariable=selected_label,
                    values=list(option_map),
                    state="readonly",
                    width=68,
                )
                version_combo.pack(fill="x", pady=(10, 12))
                chooser_actions = ttk.Frame(chooser_frame)
                chooser_actions.pack(anchor="e")

                def close_chooser() -> None:
                    chooser.grab_release()
                    chooser.destroy()
                    if installer.winfo_exists():
                        installer.grab_set()

                def choose() -> None:
                    option = option_map[selected_label.get()]
                    close_chooser()
                    on_selected(option.files)

                def cancel() -> None:
                    close_chooser()
                    on_cancel()

                ttk.Button(
                    chooser_actions,
                    text="下载",
                    command=choose,
                ).pack(side="left")
                ttk.Button(
                    chooser_actions,
                    text="取消",
                    command=cancel,
                ).pack(side="left", padx=(6, 0))
                chooser.protocol("WM_DELETE_WINDOW", cancel)
                chooser.update_idletasks()
                self._center_dialog(
                    chooser,
                    max(620, chooser.winfo_reqwidth()),
                    chooser.winfo_reqheight(),
                )

            def download_selected() -> None:
                choice = choice_by_label(kind, install_choice.get())
                repo_id = (
                    custom.get().strip()
                    if choice.key == "other"
                    else choice.repo_id
                )
                if not repo_id or "/" not in repo_id:
                    messagebox.showerror(
                        "模型名称无效",
                        "请输入有效的模型名称，例如 owner/model。",
                        parent=installer,
                    )
                    return
                download_button.configure(state="disabled")
                uninstall_button.configure(state="disabled")
                runner = ProcessRunner(
                    model_log,
                    progress_logger=model_progress_log,
                )
                download_state["busy"] = True
                download_state["runner"] = runner
                self.model_download_runner = runner

                def begin(selected_files: tuple[str, ...]) -> None:
                    if not installer.winfo_exists():
                        runner.cancel()
                        set_download_finished(runner)
                        return
                    run_download(choice, repo_id, selected_files, runner)

                should_check_files = (
                    choice.source == "huggingface"
                    and (choice.backend == "gguf" or choice.key == "other")
                )
                if not should_check_files:
                    begin(())
                    return

                def inspect_worker() -> None:
                    try:
                        options = list_huggingface_gguf_options(
                            repo_id,
                            runner,
                        )

                        def handle_options() -> None:
                            if not installer.winfo_exists():
                                runner.cancel()
                                set_download_finished(runner)
                                return
                            if not options:
                                if choice.backend == "gguf":
                                    fail_download(
                                        RuntimeError(
                                            "仓库中没有找到 GGUF 模型文件"
                                        ),
                                        runner,
                                    )
                                else:
                                    begin(())
                                return
                            if len(options) == 1:
                                begin(options[0].files)
                                return

                            def cancel_selection() -> None:
                                set_download_finished(runner)
                                restore_buttons()

                            show_file_selection(
                                options,
                                begin,
                                cancel_selection,
                            )

                        self.after(0, handle_options)
                    except Exception as exc:
                        self.after(
                            0,
                            lambda error=exc: fail_download(error, runner),
                        )

                threading.Thread(
                    target=inspect_worker,
                    daemon=True,
                ).start()

            def confirm_close_installer() -> None:
                if download_state.get("busy"):
                    if not messagebox.askyesno(
                        "确认关闭",
                        "模型正在下载，确定停止下载并关闭窗口吗？",
                        parent=installer,
                    ):
                        return
                    runner = download_state.get("runner")
                    if isinstance(runner, ProcessRunner):
                        runner.cancel()
                        set_download_finished(runner)
                installer.destroy()

            def uninstall_selected() -> None:
                installed = selected_table_model()
                if not installed:
                    return
                if not messagebox.askyesno(
                    "确认卸载",
                    f"删除模型 {installed.repo_id}？",
                    parent=installer,
                ):
                    return
                download_button.configure(state="disabled")
                uninstall_button.configure(state="disabled")

                def worker() -> None:
                    try:
                        uninstall_model(
                            installed,
                            ProcessRunner(model_log),
                        )

                        def finish() -> None:
                            clear_uninstalled(installed)
                            reload_installed()
                            refresh_installed_table()
                            if installer.winfo_exists():
                                installer.destroy()

                        self.after(0, finish)
                    except Exception as exc:
                        model_log(f"卸载失败：{exc}")
                        self.after(
                            0,
                            lambda: download_button.configure(state="normal")
                            if installer.winfo_exists()
                            else None,
                        )
                        self.after(
                            0,
                            lambda: uninstall_button.configure(state="normal")
                            if installer.winfo_exists()
                            else None,
                        )

                threading.Thread(target=worker, daemon=True).start()

            install_combo.bind("<<ComboboxSelected>>", update_custom)
            installed_tree.bind(
                "<<TreeviewSelect>>",
                lambda _event: uninstall_button.configure(
                    state="normal" if selected_table_model() else "disabled"
                ),
            )
            download_button.configure(command=download_selected)
            uninstall_button.configure(
                command=uninstall_selected,
                state="disabled",
            )
            installer.protocol(
                "WM_DELETE_WINDOW",
                confirm_close_installer,
            )
            update_custom()
            refresh_installed_table()
            installer.update_idletasks()
            self._center_dialog(
                installer,
                max(560, installer.winfo_reqwidth()),
                installer.winfo_reqheight(),
            )

        def close_dialog() -> None:
            if download_state.get("busy"):
                if not messagebox.askyesno(
                    "确认关闭",
                    "模型正在下载，确定停止下载并关闭窗口吗？",
                    parent=dialog,
                ):
                    return
                runner = download_state.get("runner")
                if isinstance(runner, ProcessRunner):
                    runner.cancel()
                    if self.model_download_runner is runner:
                        self.model_download_runner = None
                installer = download_state.get("installer")
                if isinstance(installer, tk.Toplevel) and installer.winfo_exists():
                    installer.destroy()
            save_reference()
            dialog.destroy()

        reference_browse.configure(command=browse_reference)
        reference_text_browse.configure(command=browse_reference_text)
        custom_voice_check.configure(command=refresh_custom_voice_state)
        voice_preset_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: save_reference(),
        )
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
        lock_button.configure(command=toggle_lock)
        install_open_button.configure(command=show_install_dialog)
        reload_installed(keep_locked=self.model_lock_state[kind])
        self._center_dialog(dialog, 780, 510)

    def _set_cache_directory(self) -> None:
        selected = filedialog.askdirectory(
            parent=self,
            title="选择缓存目录",
            initialdir=self.config_data.cache_dir,
        )
        if not selected:
            return
        target = Path(selected).expanduser().resolve()
        source = Path(self.config_data.cache_dir).expanduser().resolve()
        if target == source:
            return
        try:
            migrate_cache_directory(source, target)
            self.config_data.cache_dir = str(target)
            configure_cache_directory(target)
            self._persist_config()
        except OSError as exc:
            messagebox.showerror("迁移失败", f"无法迁移缓存目录：{exc}", parent=self)
            return
        messagebox.showinfo("缓存目录", f"缓存已迁移到：\n{target}", parent=self)

    def _open_cache_directory(self) -> None:
        target = Path(self.config_data.cache_dir)
        target.mkdir(parents=True, exist_ok=True)
        try:
            open_in_file_manager(target)
        except OSError as exc:
            messagebox.showerror("打开失败", str(exc), parent=self)

    def _show_version(self) -> None:
        messagebox.showinfo(
            "版本",
            f"scip - YouTube 视频中文化工具\n版本 {__version__}\n\nGitHub：{GITHUB_REPOSITORY}",
            parent=self,
        )

    def _check_update(self) -> None:
        self._separator("检查更新")

        def worker() -> None:
            try:
                headers = {"User-Agent": f"scip/{__version__}"}
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
        if self.model_download_runner is not None:
            if not messagebox.askyesno(
                "退出",
                "模型正在下载，确定停止下载并退出吗？",
                parent=self,
            ):
                return
            self.model_download_runner.cancel()
            self.model_download_runner = None
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("退出", "任务仍在运行，停止并退出？", parent=self):
                return
            self.runner.cancel()
            self._cancel_active_runners()
        try:
            self._persist_config()
        finally:
            self.destroy()


def main() -> None:
    VideoDubApp().mainloop()


if __name__ == "__main__":
    main()
