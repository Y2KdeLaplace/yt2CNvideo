from __future__ import annotations

import json
import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
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
    load_language_model_info,
    load_config,
    migrate_cache_directory,
    migrate_work_directory,
    reset_temporary_directory,
    save_config,
    save_language_model_info,
)
from videodub.dependencies import (
    inspect_dependency_versions,
    update_dependencies,
)
from videodub.video_download import (
    cleanup_new_download_directories,
    download,
    snapshot_download_directories,
)
from videodub.video_download.ui import build_download_section
from videodub.media import VideoJob, discover_video_jobs
from videodub.model_management import read_installed_model
from videodub.model_management.dialogs import (
    show_language_model_dialog,
    show_model_manager_dialog,
    show_speech_model_dialog,
)
from videodub.model_runtime import append_runtime_diagnostic
from videodub.speech_service_manager import ManagedSpeechService
from videodub.platform_utils import open_in_file_manager
from videodub.processing import (
    run_dubbing_stage,
    run_extract_stage,
    run_repair_stage,
    run_translate_stage,
)
from videodub.processing.ui import build_processing_section, job_status_values
from videodub.speech_settings import resolve_tts_reference
from videodub.speech.constants import SPEECH_WORKER_BASE_PORT
from videodub.runner import CancelledError, ProcessRunner
from videodub.subtitle_workflow import SubtitleRepairWorkflow
from videodub.subtitles import find_source_subtitle


GITHUB_REPOSITORY = "Y2KdeLaplace/yt2CNvideo"
GUI_HEARTBEAT_INTERVAL_MS = 500
GUI_WATCHDOG_INTERVAL_SECONDS = 1.0
GUI_STALL_THRESHOLD_SECONDS = 3.0
FOCUSABLE_WIDGET_CLASSES = {
    "Entry",
    "Listbox",
    "Menu",
    "Scale",
    "Scrollbar",
    "TButton",
    "TCheckbutton",
    "TCombobox",
    "TEntry",
    "TMenubutton",
    "TRadiobutton",
    "TScale",
    "TScrollbar",
    "Treeview",
    "Text",
}


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
        self.title("YouTube 视频中文化工具")
        self.geometry("1080x738")
        self.minsize(900, 630)
        self.config_data = load_config()
        load_language_model_info(self.config_data)
        configure_cache_directory(self.config_data.cache_dir)
        self.config_data.ensure_directories()
        reset_temporary_directory(self.config_data)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.mlx_inference_lock = threading.Lock()
        self._last_gui_heartbeat = time.monotonic()
        self._gui_stall_started_at: float | None = None
        self._gui_stall_detected = False
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = threading.Thread(
            target=self._watch_gui_heartbeat,
            name="tk-heartbeat-watchdog",
            daemon=True,
        )
        self.download_runner = ProcessRunner(
            lambda line: self.events.put(("log", line))
        )
        self.active_runners: list[ProcessRunner] = []
        self.model_download_runner: ProcessRunner | None = None
        self.runners_lock = threading.Lock()
        self.download_worker: threading.Thread | None = None
        self.process_worker: threading.Thread | None = None
        self.download_running = False
        self.process_running = False
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
        self.after_idle(self.focus_set)
        self.after(100, self._drain_events)
        self.after(180, self._check_tools)
        self.after(GUI_HEARTBEAT_INTERVAL_MS, self._gui_heartbeat)
        self._watchdog_thread.start()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _gui_heartbeat(self) -> None:
        if self._watchdog_stop.is_set():
            return
        self._last_gui_heartbeat = time.monotonic()
        self.after(GUI_HEARTBEAT_INTERVAL_MS, self._gui_heartbeat)

    def _note_gui_activity(self, _event: tk.Event | None = None) -> None:
        # Native live-resize loops can pause ``after`` callbacks while Tk is
        # still processing Configure events. Count those events as GUI activity.
        self._last_gui_heartbeat = time.monotonic()

    def _emit_watchdog_event(self, message: str) -> None:
        self.events.put(("log", message))
        append_runtime_diagnostic(self.config_data.cache_dir, message)

    def _check_gui_heartbeat(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        delay = now - self._last_gui_heartbeat
        if delay > GUI_STALL_THRESHOLD_SECONDS:
            if self._gui_stall_started_at is None:
                self._gui_stall_started_at = self._last_gui_heartbeat
                self._gui_stall_detected = True
                self._emit_watchdog_event(
                    f"GUI watchdog: Tk event loop stalled for {delay:.1f} s"
                )
            return
        if self._gui_stall_started_at is not None:
            stalled_for = now - self._gui_stall_started_at
            self._gui_stall_started_at = None
            self._emit_watchdog_event(
                f"GUI watchdog: Tk event loop recovered after {stalled_for:.1f} s"
            )

    def _watch_gui_heartbeat(self) -> None:
        try:
            while not self._watchdog_stop.wait(GUI_WATCHDOG_INTERVAL_SECONDS):
                self._check_gui_heartbeat()
        except Exception as exc:
            append_runtime_diagnostic(
                self.config_data.cache_dir,
                f"GUI watchdog stopped after internal error: {exc}",
            )

    def _stop_gui_watchdog(self) -> None:
        self._watchdog_stop.set()
        thread = self._watchdog_thread
        if thread is not threading.current_thread():
            thread.join(timeout=2)
        if not self._gui_stall_detected:
            append_runtime_diagnostic(
                self.config_data.cache_dir,
                "GUI watchdog: heartbeat normal; no stalls detected",
            )

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        self.option_add("*Font", (self.ui_font, 10))
        style.configure("TLabelframe.Label", font=(self.ui_font, 11, "bold"))
        style.configure("TButton", font=(self.ui_font, 11), padding=(6, 2))
        style.configure("Toolbutton.TButton", font=(self.ui_font, 11), padding=(5, 2))
        style.configure("Main.TLabel", font=(self.ui_font, 11))
        style.configure("Main.TEntry", font=(self.ui_font, 11))
        style.configure("Main.TButton", font=(self.ui_font, 11), padding=(5, 1))
        style.configure("Main.TMenubutton", font=(self.ui_font, 11), padding=(5, 1))
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
        dialog.deiconify()
        dialog.lift()
        dialog.bind("<Button-1>", self._clear_focus_on_blank_click, add="+")
        dialog.after_idle(dialog.focus_set)

    def _clear_focus_on_blank_click(self, event: tk.Event) -> None:
        if event.widget.winfo_class() in FOCUSABLE_WIDGET_CLASSES:
            return
        event.widget.winfo_toplevel().focus_set()

    def _release_header_button_focus(self, _event: tk.Event) -> None:
        self.after_idle(self.focus_set)

    def _keep_cascade_open(self, menu: tk.Menu, event: tk.Event) -> str | None:
        try:
            index = menu.index(f"@{event.y}")
        except tk.TclError:
            return None
        if index is None or menu.type(index) != "cascade":
            return None
        menu.activate(index)
        menu.tk.call(menu._w, "postcascade", index)
        return "break"

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
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=2, uniform="main-resizable-content")
        outer.rowconfigure(3, weight=1, uniform="main-resizable-content")

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 9))
        ttk.Label(header, text="工作路径", style="Main.TLabel").pack(side="left")
        self.work_path_entry = ttk.Entry(
            header,
            textvariable=self.work_dir,
            state="readonly",
            style="Main.TEntry",
        )
        self.work_path_entry.pack(
            side="left", fill="x", expand=True, padx=(9, 7)
        )
        header_actions = ttk.Frame(header)
        header_actions.pack(side="left")
        self.select_work_button = ttk.Button(
            header_actions,
            text="选择",
            command=self._browse_work_folder,
            style="Main.TButton",
            takefocus=False,
        )
        self.select_work_button.grid(row=0, column=0, sticky="ew", padx=3)
        self.open_work_button = ttk.Button(
            header_actions,
            text="打开",
            command=self._open_work_folder,
            style="Main.TButton",
            takefocus=False,
        )
        self.open_work_button.grid(row=0, column=1, sticky="ew", padx=3)
        for button in (self.select_work_button, self.open_work_button):
            button.bind(
                "<ButtonRelease-1>",
                self._release_header_button_focus,
                add="+",
            )
        model_menu = tk.Menu(self, tearoff=False)
        model_menu.add_command(label="语言模型", command=self._show_language_model)
        model_menu.add_command(label="语音模型", command=self._show_speech_models)
        model_menu.add_command(label="语音模型管理", command=self._show_model_manager)
        self.model_menu_button = ttk.Menubutton(
            header_actions,
            text="模型",
            menu=model_menu,
            style="Main.TMenubutton",
            takefocus=False,
        )
        self.model_menu_button.grid(row=0, column=2, sticky="ew", padx=3)
        about_menu = tk.Menu(self, tearoff=False)
        about_menu.add_command(label="更新", command=self._check_update)
        about_menu.add_command(label="版本", command=self._show_version)
        cache_menu = tk.Menu(about_menu, tearoff=False)
        cache_menu.add_command(label="设置缓存目录", command=self._set_cache_directory)
        cache_menu.add_command(label="打开缓存目录", command=self._open_cache_directory)
        about_menu.add_cascade(label="缓存目录", menu=cache_menu)
        about_menu.bind(
            "<ButtonRelease-1>",
            lambda event: self._keep_cascade_open(about_menu, event),
            add="+",
        )
        self.about_menu_button = ttk.Menubutton(
            header_actions,
            text="关于",
            menu=about_menu,
            style="Main.TMenubutton",
            takefocus=False,
        )
        self.about_menu_button.grid(row=0, column=3, sticky="ew", padx=3)
        for column in range(4):
            header_actions.columnconfigure(column, weight=1, uniform="header-action")

        download_section = ttk.Frame(outer)
        download_section.grid(row=1, column=0, sticky="ew")
        process_section = ttk.Frame(outer)
        process_section.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        self._build_download_section(download_section)
        self._build_processing_section(process_section)

        log_box = ttk.LabelFrame(outer, text="运行日志", padding=7)
        log_box.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)
        self.log = tk.Text(
            log_box,
            height=9,
            wrap="word",
            state="disabled",
            font=(self.mono_font, 9),
            background="#111827",
            foreground="#e5e7eb",
            insertbackground="#e5e7eb",
        )
        scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.bind("<Button-1>", self._clear_focus_on_blank_click, add="+")
        self.bind_all("<Configure>", self._note_gui_activity, add="+")

    def _build_download_section(self, parent: ttk.Frame) -> None:
        build_download_section(self, parent)

    def _build_processing_section(self, parent: ttk.Frame) -> None:
        build_processing_section(self, parent)

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _separator(self, title: str) -> None:
        self._append_log(f"== {title} " + "=" * max(4, 50 - len(title)))

    def _check_tools(self) -> None:
        def worker() -> None:
            runner = ProcessRunner(lambda line: self.events.put(("log", line)))
            try:
                summary = update_dependencies(runner)
            except (OSError, RuntimeError) as exc:
                summary = f"依赖自动更新失败：{exc}"
            self.events.put(("log", summary))
            try:
                versions = inspect_dependency_versions(self.config_data, runner)
                self.events.put(("log", versions.log_line()))
            except (OSError, RuntimeError) as exc:
                self.events.put(("log", f"依赖检查失败：{exc}"))

        threading.Thread(target=worker, daemon=True).start()

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
        if getattr(self, "process_running", False):
            return "break"
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
            source = job.source_subtitle_path or find_source_subtitle(job.video_path)
            values = job_status_values(
                job,
                source,
                self.config_data.translation_language,
                self.config_data.tts_language,
                self.config_data.output_dir,
            )
            item = self.job_tree.insert("", "end", iid=str(index), values=values)
            if str(job.video_path) in selected_paths:
                self.job_tree.selection_add(item)

    def _selected_jobs(self) -> list[VideoJob]:
        selected = set(self.job_tree.selection())
        return [
            self.jobs[int(item)]
            for item in self.job_tree.get_children()
            if item in selected and item.isdigit() and int(item) < len(self.jobs)
        ]

    def _clear_job_selection_on_blank(self, event: tk.Event) -> str | None:
        if getattr(self, "process_running", False):
            return "break"
        if self.job_tree.identify_row(event.y):
            return None
        self.job_tree.selection_remove(*self.job_tree.selection())
        self.focus_set()
        return "break"

    def _toggle_job_selection(self, event: tk.Event) -> str:
        if getattr(self, "process_running", False):
            return "break"
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
        if getattr(self, "process_running", False):
            return "break"
        if self.job_tree.selection():
            self.job_tree.selection_set(self.job_tree.get_children())
        return "break"

    def _set_running(self, running: bool, task: str) -> None:
        if task == "download":
            self.download_running = running
        elif task == "process":
            self.process_running = running
        else:
            raise ValueError("未知任务状态")
        self.url_text.configure(
            state="disabled" if self.download_running else "normal"
        )
        self.refresh_jobs_button.configure(
            state="disabled" if self.process_running else "normal"
        )
        self.job_tree.state(
            ("disabled",) if self.process_running else ("!disabled",)
        )
        if self.download_running:
            self.download_button.configure(
                text="停止",
                command=self._stop_download,
                state="normal",
            )
        else:
            self.download_button.configure(
                text="下载",
                command=self._start_download,
                state="normal",
            )
        if self.process_running:
            self.process_button.configure(
                text="停止",
                command=self._stop_processing,
                state="normal",
            )
        else:
            self.process_button.configure(
                text="运行",
                command=self._start_processing,
                state="normal",
            )

    def _start_download(self) -> None:
        if self.download_worker and self.download_worker.is_alive():
            return
        urls = [line.strip() for line in self.url_text.get("1.0", "end").splitlines() if line.strip()]
        if not urls:
            messagebox.showwarning("缺少链接", "请先输入 YouTube 链接。", parent=self)
            return
        try:
            config = replace(self._sync_basic_config())
        except Exception as exc:
            messagebox.showerror("设置错误", str(exc), parent=self)
            return
        self._separator("YouTube 视频下载")
        self.download_runner.reset()
        self._set_running(True, "download")
        self.download_worker = threading.Thread(
            target=self._download_worker,
            args=(config, urls),
            daemon=True,
        )
        self.download_worker.start()

    def _download_worker(self, config: AppConfig, urls: list[str]) -> None:
        try:
            for url in urls:
                self.download_runner.check_cancelled()
                before = snapshot_download_directories(config)
                try:
                    download(config, self.download_runner, url)
                except Exception:
                    cleanup_new_download_directories(
                        config,
                        before,
                        self.download_runner,
                    )
                    raise
            self.events.put(("task_done", ("download", "下载完成")))
        except CancelledError:
            self.events.put(("task_done", ("download", "下载已停止")))
        except Exception as exc:
            self.events.put(("task_error", ("download", exc)))

    def _start_processing(self) -> None:
        if self.process_worker and self.process_worker.is_alive():
            return
        jobs = self._selected_jobs()
        stages = (
            self.stage_asr.get(),
            self.stage_repair.get(),
            self.stage_translate.get(),
            self.stage_dub.get(),
        )
        if not jobs:
            messagebox.showwarning("没有选择任务", "请至少选择一个视频或字幕。", parent=self)
            return
        if not any(stages):
            messagebox.showwarning("没有任务", "请至少选择一个处理步骤。", parent=self)
            return
        try:
            config = replace(self._sync_basic_config())
            if (stages[1] or stages[2] or stages[3]) and (
                not config.subtitle_api_base_url or not config.subtitle_model
            ):
                raise ValueError("请先在“模型 → 语言模型”中完成配置。")
            if stages[0] and any(job.has_video for job in jobs) and not config.asr_model_path:
                raise ValueError("请先在“模型 → 语音模型”中选择语音识别。")
            if stages[3] and not config.tts_model_path:
                raise ValueError("请先在“模型 → 语音模型”中选择语音生成。")
            if stages[3]:
                tts_model = read_installed_model(config.tts_model_path, config)
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
        self._set_running(True, "process")
        self.process_worker = threading.Thread(
            target=self._processing_worker,
            args=(config, jobs, stages, parallel),
            daemon=True,
        )
        self.process_worker.start()

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
                self.events.put(("log", f"并行处理：{worker_count} 个任务"))
                failed_count = 0
                with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="videodub",
                ) as executor:
                    futures = {
                        executor.submit(
                            self._process_one_job,
                            config,
                            job,
                            stages,
                            index,
                        ): job
                        for index, job in enumerate(jobs)
                    }
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except CancelledError:
                            self._cancel_active_runners()
                            for pending in futures:
                                pending.cancel()
                            raise
                        except Exception as exc:
                            failed_count += 1
                            job = futures[future]
                            self.events.put(
                                (
                                    "log",
                                    f"[{job.title}] 处理失败，已释放资源并跳过：{exc}",
                                )
                            )
                message = (
                    f"处理完成，{failed_count} 个任务失败"
                    if failed_count
                    else "处理完成"
                )
                self.events.put(("task_done", ("process", message)))
                return
            self.events.put(("task_done", ("process", "处理完成")))
        except CancelledError:
            self._cancel_active_runners()
            self.events.put(("task_done", ("process", "处理已停止")))
        except Exception as exc:
            self._cancel_active_runners()
            self.events.put(("task_error", ("process", exc)))

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
            asr_url = ""
            if extract and job.has_video:
                with self._mlx_inference_slot(
                    runner,
                    enabled=config.asr_backend == "mlx",
                ):
                    with ManagedSpeechService(
                        config,
                        runner,
                        "asr",
                        port=SPEECH_WORKER_BASE_PORT + slot,
                    ) as asr_service:
                        asr_url = asr_service.base_url
                        run_extract_stage(
                            config,
                            runner,
                            job,
                            base_url=asr_url,
                        )
            elif extract:
                run_extract_stage(config, runner, job, base_url=asr_url)
            if repair or translate:
                workflow = SubtitleRepairWorkflow(
                    config,
                    runner,
                    api_key=self.session_api_key
                    or api_key_from_runtime(config=config),
                )
                if translate:
                    run_translate_stage(workflow, job, repair_first=repair)
                else:
                    run_repair_stage(workflow, job)
            if dubbing:
                with self._mlx_inference_slot(
                    runner,
                    enabled=config.tts_backend == "mlx",
                ):
                    with ManagedSpeechService(
                        config,
                        runner,
                        "tts",
                        port=SPEECH_WORKER_BASE_PORT + slot,
                    ) as tts_service:
                        tts_url = tts_service.base_url
                        output = run_dubbing_stage(
                            config,
                            runner,
                            job,
                            base_url=tts_url,
                        )
                        runner.logger(f"配音输出：{output}")
        finally:
            with self.runners_lock:
                if runner in self.active_runners:
                    self.active_runners.remove(runner)

    def _acquire_mlx_inference(self, runner: ProcessRunner) -> None:
        runner.check_cancelled()
        if self.mlx_inference_lock.acquire(blocking=False):
            try:
                runner.check_cancelled()
            except Exception:
                self.mlx_inference_lock.release()
                raise
            runner.logger("已获得 MLX 推理资源")
            return
        runner.logger("等待本机 MLX 推理资源…")
        while True:
            runner.check_cancelled()
            if not self.mlx_inference_lock.acquire(timeout=0.2):
                continue
            try:
                runner.check_cancelled()
            except Exception:
                self.mlx_inference_lock.release()
                raise
            runner.logger("已获得 MLX 推理资源")
            return

    @contextmanager
    def _mlx_inference_slot(self, runner: ProcessRunner, *, enabled: bool):
        if not enabled:
            yield
            return
        self._acquire_mlx_inference(runner)
        try:
            yield
        finally:
            self.mlx_inference_lock.release()

    def _stop_download(self) -> None:
        self.download_runner.cancel()
        self._append_log("正在停止下载任务…")

    def _stop_processing(self) -> None:
        self._cancel_active_runners()
        self._append_log("正在停止处理任务…")

    def _cancel_active_runners(self) -> None:
        with self.runners_lock:
            runners = list(self.active_runners)
        for runner in runners:
            runner.cancel()

    def _defer_model_dialog(
        self,
        opener: Callable[[VideoDubApp], None],
    ) -> None:
        def open_after_menu_closes() -> None:
            self.model_menu_button.state(("!pressed", "!active"))
            opener(self)

        self.after_idle(open_after_menu_closes)

    def _show_language_model(self) -> None:
        self._defer_model_dialog(show_language_model_dialog)

    def _show_speech_models(self) -> None:
        self._defer_model_dialog(show_speech_model_dialog)

    def _show_model_manager(self) -> None:
        self._defer_model_dialog(show_model_manager_dialog)

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
            f"YouTube 视频中文化工具\n版本 {__version__}\n\nGitHub：{GITHUB_REPOSITORY}",
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
            except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
                self.events.put(("error", RuntimeError(f"检查更新失败：{exc}")))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "task_done":
                    task, message = payload
                    self._append_log(str(message))
                    self._set_running(False, str(task))
                    self._refresh_jobs()
                elif kind == "task_error":
                    task, error = payload
                    self._append_log(f"错误：{error}")
                    self._set_running(False, str(task))
                    self._refresh_jobs()
                    messagebox.showerror("任务失败", str(error), parent=self)
                elif kind == "error":
                    self._append_log(f"错误：{payload}")
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
                        messagebox.showinfo(
                            "更新",
                            f"当前已是最新版本 {__version__}。",
                            parent=self,
                        )
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
        download_alive = self.download_worker and self.download_worker.is_alive()
        process_alive = self.process_worker and self.process_worker.is_alive()
        if download_alive or process_alive:
            if not messagebox.askyesno("退出", "任务仍在运行，停止并退出？", parent=self):
                return
            self.download_runner.cancel()
            self._cancel_active_runners()
        try:
            self._persist_config()
        finally:
            self._stop_gui_watchdog()
            self.destroy()


def main() -> None:
    VideoDubApp().mainloop()


if __name__ == "__main__":
    main()
