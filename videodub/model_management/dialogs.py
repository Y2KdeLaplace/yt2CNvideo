from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Protocol

from ..config import encrypt_api_key
from ..runner import ProcessRunner
from .backend import (
    InstalledModel,
    ModelFileOption,
    download_model,
    list_huggingface_gguf_options,
    list_installed_models,
    normalize_repository_id,
    repair_model_dependencies,
    uninstall_model,
)
from ..speech.registry import find_model_spec
from .voices import import_voice_sample, list_voice_samples


PLATFORM_LABELS = {
    "Hugging Face": "huggingface",
    "ModelScope": "modelscope",
}


class ModelDialogHost(Protocol):
    config_data: object
    session_api_key: str
    mono_font: str
    model_download_runner: ProcessRunner | None

    def after(self, delay_ms: int, callback): ...

    def _center_dialog(self, dialog: tk.Toplevel, width: int, height: int) -> None: ...

    def _persist_config(self) -> None: ...

    def _append_log(self, message: str) -> None: ...


def _model_label(model: InstalledModel) -> str:
    source = "ModelScope" if model.source == "modelscope" else "Hugging Face"
    suffix = f" / {Path(model.path).name}" if Path(model.path).is_file() else ""
    return f"{model.repo_id}（{source}）{suffix}"


class _ModelPicker(ttk.Frame):
    def __init__(self, master: tk.Misc, on_select) -> None:
        super().__init__(master)
        self._on_select = on_select
        self._models: dict[str, InstalledModel] = {}
        self._selected_path = ""
        self.label = tk.StringVar(value="未选择")
        self.button = ttk.Menubutton(self, textvariable=self.label)
        self.button.pack(fill="x", expand=True)
        self.menu = tk.Menu(self.button, tearoff=False)
        self.button.configure(menu=self.menu)

    @property
    def selected_path(self) -> str:
        return self._selected_path

    def selected_model(self) -> InstalledModel | None:
        return self._models.get(self._selected_path)

    def set_models(
        self,
        models: list[InstalledModel],
        selected_path: str,
        excluded_path: str,
    ) -> None:
        self._models = {model.path: model for model in models}
        self._selected_path = selected_path if selected_path in self._models else ""
        self.label.set(
            _model_label(self._models[self._selected_path])
            if self._selected_path
            else "未选择"
        )
        self.menu.delete(0, "end")
        self.menu.add_command(label="未选择", command=lambda: self._choose(""))
        self.menu.add_separator()
        for model in models:
            self.menu.add_command(
                label=_model_label(model),
                state="disabled" if model.path == excluded_path else "normal",
                command=lambda path=model.path: self._choose(path),
            )

    def _choose(self, path: str) -> None:
        self._selected_path = path
        self.label.set(_model_label(self._models[path]) if path else "未选择")
        self._on_select()


def show_language_model_dialog(app: ModelDialogHost) -> None:
    dialog = tk.Toplevel(app)
    dialog.withdraw()
    dialog.title("语言模型")
    dialog.transient(app)
    dialog.grab_set()
    dialog.resizable(False, False)
    frame = ttk.Frame(dialog, padding=16)
    frame.pack(fill="both", expand=True)
    base = tk.StringVar(value=app.config_data.subtitle_api_base_url)
    key = tk.StringVar(value=app.session_api_key)
    model = tk.StringVar(value=app.config_data.subtitle_model)
    save = tk.BooleanVar(value=app.config_data.save_model_info)
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
            messagebox.showwarning(
                "信息不完整", "请填写 API 地址和模型。", parent=dialog
            )
            return
        app.config_data.subtitle_api_base_url = base.get().strip()
        app.config_data.subtitle_model = model.get().strip()
        app.config_data.save_model_info = save.get()
        app.session_api_key = key.get().strip()
        app.config_data.subtitle_api_key_encrypted = (
            encrypt_api_key(app.session_api_key) if save.get() else ""
        )
        app._persist_config()
        dialog.destroy()

    buttons = ttk.Frame(frame)
    buttons.grid(row=4, column=0, columnspan=2, sticky="e")
    ttk.Button(buttons, text="保存", command=commit).pack(side="left")
    dialog.update_idletasks()
    app._center_dialog(dialog, dialog.winfo_reqwidth(), dialog.winfo_reqheight())


def show_speech_model_dialog(app: ModelDialogHost) -> None:
    dialog = tk.Toplevel(app)
    dialog.withdraw()
    dialog.title("语音模型")
    dialog.transient(app)
    dialog.grab_set()
    dialog.resizable(False, False)
    frame = ttk.Frame(dialog, padding=12)
    frame.pack(fill="both", expand=True)
    asr_models = list_installed_models("asr", config=app.config_data)
    tts_models = list_installed_models("tts", config=app.config_data)
    model_paths = {model.path for model in (*asr_models, *tts_models)}
    asr_path = (
        app.config_data.asr_model_path
        if app.config_data.asr_model_path in model_paths
        else ""
    )
    tts_path = (
        app.config_data.tts_model_path
        if app.config_data.tts_model_path in model_paths
        else ""
    )
    if asr_path == tts_path:
        tts_path = ""

    ttk.Label(frame, text="语音识别").grid(row=0, column=0, sticky="w", pady=4)
    ttk.Label(frame, text="语音生成").grid(row=1, column=0, sticky="w", pady=4)

    refreshing = False

    def refresh_pickers() -> None:
        nonlocal refreshing
        if refreshing:
            return
        refreshing = True
        current_asr = asr_picker.selected_path
        current_tts = tts_picker.selected_path
        asr_picker.set_models(asr_models, current_asr, current_tts)
        tts_picker.set_models(tts_models, current_tts, current_asr)
        refreshing = False

    asr_picker = _ModelPicker(frame, refresh_pickers)
    asr_picker.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=4)
    tts_picker = _ModelPicker(frame, refresh_pickers)
    tts_picker.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=4)
    asr_picker.set_models(asr_models, asr_path, tts_path)
    tts_picker.set_models(tts_models, tts_path, asr_path)

    voices = list_voice_samples()
    voice_names = [sample.name for sample in voices]
    selected_voice = tk.StringVar(
        value=(
            app.config_data.tts_voice_preset
            if app.config_data.tts_voice_preset in voice_names
            else ""
        )
    )
    ttk.Label(frame, text="声音").grid(row=2, column=0, sticky="w", pady=4)
    voice_row = ttk.Frame(frame)
    voice_row.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=4)
    voice_combo = ttk.Combobox(
        voice_row,
        textvariable=selected_voice,
        values=voice_names,
        state="readonly",
        width=12,
    )
    voice_combo.pack(side="left")

    def show_import_dialog() -> None:
        importer = tk.Toplevel(dialog)
        importer.withdraw()
        importer.title("导入声音")
        importer.transient(dialog)
        importer.grab_set()
        importer.resizable(False, False)
        content = ttk.Frame(importer, padding=16)
        content.pack(fill="both", expand=True)
        media = tk.StringVar()
        text = tk.StringVar()
        name = tk.StringVar()
        ttk.Label(content, text="音频或视频文件").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(content, textvariable=media, width=55).grid(
            row=0, column=1, sticky="ew", padx=(10, 8), pady=6
        )
        ttk.Label(content, text="文本或字幕文件").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(content, textvariable=text, width=55).grid(
            row=1, column=1, sticky="ew", padx=(10, 8), pady=6
        )
        ttk.Label(content, text="声音名称").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(content, textvariable=name, width=55).grid(
            row=2, column=1, sticky="ew", padx=(10, 8), pady=6
        )

        def browse_media() -> None:
            selected = filedialog.askopenfilename(
                parent=importer,
                title="选择音频或视频文件",
                filetypes=(
                    ("媒体文件", "*.wav *.mp3 *.m4a *.flac *.aac *.ogg *.opus *.mp4 *.mkv *.mov *.webm *.avi"),
                    ("所有文件", "*.*"),
                ),
            )
            if selected:
                media.set(selected)
                name.set(Path(selected).stem)

        def browse_text() -> None:
            selected = filedialog.askopenfilename(
                parent=importer,
                title="选择对应文本文件",
                filetypes=(
                    ("文本文件", "*.txt *.text *.md *.srt *.vtt"),
                    ("所有文件", "*.*"),
                ),
            )
            if selected:
                text.set(selected)

        ttk.Button(content, text="选择", command=browse_media).grid(row=0, column=2, pady=6)
        ttk.Button(content, text="选择", command=browse_text).grid(row=1, column=2, pady=6)
        actions = ttk.Frame(content)
        actions.grid(row=3, column=0, columnspan=3, sticky="e", pady=(10, 0))
        import_button = ttk.Button(actions, text="导入")
        import_button.pack(side="left")

        def finish(sample_name: str) -> None:
            if not importer.winfo_exists():
                return
            current = [sample.name for sample in list_voice_samples()]
            voice_combo.configure(values=current)
            selected_voice.set(sample_name)
            importer.destroy()

        def fail(error: Exception) -> None:
            if importer.winfo_exists():
                import_button.configure(state="normal")
                messagebox.showerror("导入失败", str(error), parent=importer)

        def start_import() -> None:
            media_path = media.get().strip()
            text_path = text.get().strip()
            sample_name = name.get().strip()
            if not media_path or not text_path or not sample_name:
                messagebox.showwarning(
                    "信息不完整",
                    "请选择媒体文件、文本或字幕文件，并填写声音名称。",
                    parent=importer,
                )
                return
            import_button.configure(state="disabled")

            def worker() -> None:
                try:
                    sample = import_voice_sample(
                        media_path,
                        text_path,
                        app.config_data.ffmpeg_path,
                        ProcessRunner(
                            lambda line: app.after(
                                0,
                                lambda message=line: app._append_log(message),
                            )
                        ),
                        name=sample_name,
                    )
                    app.after(0, lambda: finish(sample.name))
                except Exception as exc:
                    app.after(0, lambda error=exc: fail(error))

            threading.Thread(target=worker, daemon=True).start()

        import_button.configure(command=start_import)
        importer.update_idletasks()
        app._center_dialog(importer, max(640, importer.winfo_reqwidth()), importer.winfo_reqheight())

    ttk.Button(voice_row, text="导入声音", command=show_import_dialog).pack(
        side="left",
        padx=(6, 0),
    )

    if not asr_models and not tts_models:
        ttk.Label(
            frame,
            text="当前清单中没有已下载模型，请先打开“语音模型管理”。",
            foreground="#9a3412",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def apply_model(kind: str, model: InstalledModel | None) -> None:
        prefix = "asr" if kind == "asr" else "tts"
        setattr(app.config_data, f"{prefix}_backend", model.backend if model else "")
        setattr(app.config_data, f"{prefix}_model_id", model.repo_id if model else "")
        setattr(app.config_data, f"{prefix}_model_path", model.path if model else "")
        if kind == "tts":
            app.config_data.tts_codec_path = model.codec_path if model else ""

    def save() -> None:
        apply_model("asr", asr_picker.selected_model())
        apply_model("tts", tts_picker.selected_model())
        app.config_data.tts_voice_preset = selected_voice.get().strip()
        app.config_data.tts_use_custom_voice = False
        app.config_data.tts_reference_audio = ""
        app.config_data.tts_reference_text = ""
        app.config_data.tts_reference_text_file = ""
        app._persist_config()
        dialog.destroy()

    actions = ttk.Frame(frame)
    actions.grid(row=4, column=0, columnspan=2, sticky="e", pady=(10, 0))
    ttk.Button(actions, text="保存", command=save).pack(side="left")
    frame.columnconfigure(1, weight=1)
    dialog.update_idletasks()
    app._center_dialog(dialog, 700, dialog.winfo_reqheight())


def show_model_manager_dialog(app: ModelDialogHost) -> None:
    dialog = tk.Toplevel(app)
    dialog.withdraw()
    dialog.title("语音模型管理")
    dialog.transient(app)
    dialog.grab_set()
    dialog.minsize(760, 480)
    frame = ttk.Frame(dialog, padding=14)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(1, weight=3, uniform="model-manager-content")
    frame.rowconfigure(2, weight=2, uniform="model-manager-content")
    style = ttk.Style(dialog)
    style.map(
        "ModelManager.Treeview",
        background=[("selected", "#005a9e")],
        foreground=[("selected", "#ffffff")],
    )

    download_frame = ttk.LabelFrame(frame, text="下载管理", padding=12)
    download_frame.grid(row=0, column=0, sticky="ew")
    platform_name = tk.StringVar(value="Hugging Face")
    repo_id = tk.StringVar()
    ttk.Label(download_frame, text="模型平台").grid(row=0, column=0, sticky="w", pady=4)
    ttk.Combobox(
        download_frame,
        textvariable=platform_name,
        values=tuple(PLATFORM_LABELS),
        state="readonly",
        width=20,
    ).grid(row=0, column=1, sticky="w", padx=(10, 0), pady=4)
    ttk.Label(download_frame, text="模型名称").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Entry(download_frame, textvariable=repo_id).grid(
        row=1, column=1, sticky="ew", padx=(10, 8), pady=4
    )
    download_button = ttk.Button(download_frame, text="下载")
    download_button.grid(row=1, column=2, sticky="e", pady=4)
    download_frame.columnconfigure(1, weight=1)

    model_frame = ttk.LabelFrame(frame, text="已下载模型", padding=8)
    model_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
    model_frame.columnconfigure(0, weight=1)
    model_frame.rowconfigure(0, weight=1)
    tree = ttk.Treeview(
        model_frame,
        columns=("platform", "model", "runtime", "support", "status", "path"),
        show="headings",
        height=6,
        style="ModelManager.Treeview",
    )
    tree.heading("platform", text="平台")
    tree.heading("model", text="模型")
    tree.heading("runtime", text="Runtime")
    tree.heading("support", text="SCIP support")
    tree.heading("status", text="状态")
    tree.heading("path", text="位置")
    tree.column("platform", width=105, stretch=False)
    tree.column("model", width=240, stretch=False)
    tree.column("runtime", width=95, stretch=False)
    tree.column("support", width=95, stretch=False)
    tree.column("status", width=105, stretch=False)
    tree.column("path", width=260, stretch=True)
    tree.grid(row=0, column=0, sticky="nsew")
    tree_scroll = ttk.Scrollbar(model_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=tree_scroll.set)
    tree_scroll.grid(row=0, column=1, sticky="ns")
    detail = tk.StringVar(value="选择模型可查看 capabilities 与 companion dependencies。")
    ttk.Label(model_frame, textvariable=detail, wraplength=820, justify="left").grid(
        row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0)
    )
    actions = ttk.Frame(model_frame)
    actions.grid(row=2, column=0, columnspan=2, sticky="e", pady=(7, 0))
    repair_button = ttk.Button(actions, text="修复依赖", state="disabled")
    repair_button.pack(side="left")
    uninstall_button = ttk.Button(actions, text="卸载", state="disabled")
    uninstall_button.pack(side="left", padx=(6, 0))

    log_frame = ttk.LabelFrame(frame, text="下载命令输出", padding=8)
    log_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(0, weight=1)
    log = tk.Text(
        log_frame,
        height=6,
        state="disabled",
        wrap="word",
        font=(app.mono_font, 9),
        background="#111827",
        foreground="#e5e7eb",
    )
    log.grid(row=0, column=0, sticky="nsew")
    log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log.yview)
    log.configure(yscrollcommand=log_scroll.set)
    log_scroll.grid(row=0, column=1, sticky="ns")
    rows: dict[str, InstalledModel] = {}
    state: dict[str, object] = {"busy": False, "runner": None}
    progress_active = False

    def write_log(text: str, *, progress: bool = False) -> None:
        def write() -> None:
            nonlocal progress_active
            if not dialog.winfo_exists():
                return
            message = text.rstrip()
            is_progress = progress or message.startswith("下载进度（")
            log.configure(state="normal")
            if is_progress and progress_active:
                log.delete("progress_start", "end-1c")
            elif is_progress:
                log.mark_set("progress_start", "end-1c")
                log.mark_gravity("progress_start", "left")
            elif progress_active:
                log.mark_unset("progress_start")
            log.insert("end", message + "\n")
            log.see("end")
            log.configure(state="disabled")
            progress_active = is_progress

        app.after(0, write)

    def refresh() -> None:
        rows.clear()
        tree.delete(*tree.get_children())
        installed_models = sorted(
            list_installed_models(config=app.config_data),
            key=lambda item: (find_model_spec(item.repo_id) is None, item.repo_id.casefold()),
        )
        for index, model in enumerate(installed_models):
            spec = find_model_spec(model.repo_id)
            missing = [] if spec is None else [
                dependency.display_name
                for dependency in spec.dependencies
                if not getattr(model, dependency.manifest_field, "")
                or not Path(getattr(model, dependency.manifest_field, "")).exists()
            ]
            row_id = str(index)
            rows[row_id] = model
            tree.insert(
                "",
                "end",
                iid=row_id,
                values=(
                    "ModelScope" if model.source == "modelscope" else "Hugging Face",
                    model.repo_id,
                    spec.engine.upper() if spec else "Unknown",
                    "Supported" if spec else "Unsupported",
                    "依赖缺失" if missing else "已下载",
                    model.path,
                ),
            )
        uninstall_button.configure(state="disabled")
        repair_button.configure(state="disabled")
        detail.set("选择模型可查看 capabilities 与 companion dependencies。")

    def selected_model() -> InstalledModel | None:
        selected = tree.selection()
        return rows.get(selected[0]) if selected else None

    def clear_selection_on_blank(event: tk.Event) -> str | None:
        if tree.identify_row(event.y):
            return None
        tree.selection_remove(*tree.selection())
        uninstall_button.configure(state="disabled")
        dialog.focus_set()
        return "break"

    def set_busy(busy: bool, runner: ProcessRunner | None = None) -> None:
        state["busy"] = busy
        state["runner"] = runner
        app.model_download_runner = runner if busy else None
        if dialog.winfo_exists():
            download_button.configure(state="disabled" if busy else "normal")
            uninstall_button.configure(
                state="disabled" if busy or selected_model() is None else "normal"
            )
            model = selected_model()
            spec = find_model_spec(model.repo_id) if model else None
            missing = bool(spec and any(
                not getattr(model, dependency.manifest_field, "")
                or not Path(getattr(model, dependency.manifest_field, "")).exists()
                for dependency in spec.dependencies
            ))
            repair_button.configure(state="normal" if not busy and missing else "disabled")

    def finish_download(installed: InstalledModel, runner: ProcessRunner) -> None:
        if state.get("runner") is not runner:
            return
        set_busy(False)
        if dialog.winfo_exists():
            refresh()
            write_log(f"已加入本地模型清单：{installed.repo_id}")

    def fail_download(error: Exception, runner: ProcessRunner) -> None:
        if state.get("runner") is runner:
            set_busy(False)
        if dialog.winfo_exists():
            write_log(f"下载失败：{error}")

    def run_download(
        source: str,
        repository: str,
        files: tuple[str, ...],
        runner: ProcessRunner,
    ) -> None:
        def worker() -> None:
            try:
                installed = download_model(
                    app.config_data,
                    source,
                    repository,
                    runner,
                    files,
                )
                app.after(0, lambda: finish_download(installed, runner))
            except Exception as exc:
                app.after(0, lambda error=exc: fail_download(error, runner))

        threading.Thread(target=worker, daemon=True).start()

    def choose_gguf(
        options: tuple[ModelFileOption, ...],
        on_selected,
        on_cancel,
    ) -> None:
        chooser = tk.Toplevel(dialog)
        chooser.withdraw()
        chooser.title("选择模型版本")
        chooser.transient(dialog)
        chooser.grab_set()
        content = ttk.Frame(chooser, padding=14)
        content.pack(fill="both", expand=True)
        option_map = {option.label: option for option in options}
        selection = tk.StringVar(value=options[0].label)
        ttk.Label(content, text="检测到多个 GGUF 版本，请选择要下载的模型：").pack(anchor="w")
        ttk.Combobox(
            content,
            textvariable=selection,
            values=list(option_map),
            state="readonly",
            width=68,
        ).pack(fill="x", pady=(10, 12))
        actions = ttk.Frame(content)
        actions.pack(anchor="e")

        def close() -> None:
            chooser.grab_release()
            chooser.destroy()
            if dialog.winfo_exists():
                dialog.grab_set()

        def accept() -> None:
            files = option_map[selection.get()].files
            close()
            on_selected(files)

        def cancel() -> None:
            close()
            on_cancel()

        ttk.Button(actions, text="下载", command=accept).pack(side="left")
        ttk.Button(actions, text="取消", command=cancel).pack(side="left", padx=(6, 0))
        chooser.protocol("WM_DELETE_WINDOW", cancel)
        chooser.update_idletasks()
        app._center_dialog(chooser, max(620, chooser.winfo_reqwidth()), chooser.winfo_reqheight())

    def start_download() -> None:
        try:
            repository = normalize_repository_id(repo_id.get())
        except ValueError as exc:
            messagebox.showerror("模型名称无效", str(exc), parent=dialog)
            return
        source = PLATFORM_LABELS[platform_name.get()]
        runner = ProcessRunner(write_log, progress_logger=lambda text: write_log(text, progress=True))
        set_busy(True, runner)

        def begin(files: tuple[str, ...] = ()) -> None:
            run_download(source, repository, files, runner)

        if source != "huggingface" or "gguf" not in repository.casefold():
            begin()
            return

        def inspect() -> None:
            try:
                options = list_huggingface_gguf_options(repository, runner)

                def handle() -> None:
                    if not options:
                        fail_download(RuntimeError("仓库中没有找到 GGUF 模型文件"), runner)
                    elif len(options) == 1:
                        begin(options[0].files)
                    else:
                        choose_gguf(options, begin, lambda: set_busy(False))

                app.after(0, handle)
            except Exception as exc:
                app.after(0, lambda error=exc: fail_download(error, runner))

        threading.Thread(target=inspect, daemon=True).start()

    def clear_selection(model: InstalledModel) -> None:
        if Path(app.config_data.asr_model_path) == Path(model.path):
            app.config_data.asr_backend = ""
            app.config_data.asr_model_id = ""
            app.config_data.asr_model_path = ""
        if Path(app.config_data.tts_model_path) == Path(model.path):
            app.config_data.tts_backend = ""
            app.config_data.tts_model_id = ""
            app.config_data.tts_model_path = ""
            app.config_data.tts_codec_path = ""
        app._persist_config()

    def start_uninstall() -> None:
        model = selected_model()
        if model is None:
            return
        if not messagebox.askyesno(
            "确认卸载", f"删除模型 {model.repo_id}？", parent=dialog
        ):
            return
        runner = ProcessRunner(write_log)
        set_busy(True, runner)

        def worker() -> None:
            try:
                uninstall_model(app.config_data, model, runner)

                def finish() -> None:
                    clear_selection(model)
                    if state.get("runner") is runner:
                        set_busy(False)
                    if dialog.winfo_exists():
                        refresh()

                app.after(0, finish)
            except Exception as exc:
                app.after(0, lambda error=exc: fail_download(error, runner))

        threading.Thread(target=worker, daemon=True).start()

    def start_repair() -> None:
        model = selected_model()
        if model is None:
            return
        runner = ProcessRunner(write_log, progress_logger=lambda text: write_log(text, progress=True))
        set_busy(True, runner)

        def worker() -> None:
            try:
                repair_model_dependencies(app.config_data, model, runner)
                app.after(0, lambda: (set_busy(False), refresh()))
            except Exception as exc:
                app.after(0, lambda error=exc: fail_download(error, runner))

        threading.Thread(target=worker, daemon=True).start()

    def update_selection() -> None:
        model = selected_model()
        if model is None:
            uninstall_button.configure(state="disabled")
            repair_button.configure(state="disabled")
            return
        spec = find_model_spec(model.repo_id)
        uninstall_button.configure(state="disabled" if state.get("busy") else "normal")
        if spec is None:
            detail.set("Runtime: Unknown · SCIP support: Unsupported · 该仓库只作为已下载缓存记录。")
            repair_button.configure(state="disabled")
            return
        dependency_lines = []
        missing = False
        for dependency in spec.dependencies:
            ready = bool(
                getattr(model, dependency.manifest_field, "")
                and Path(getattr(model, dependency.manifest_field, "")).exists()
            )
            missing = missing or not ready
            dependency_lines.append(f"{'✓' if ready else '✗'} {dependency.display_name}")
        unavailable = {
            capability
            for dependency in spec.dependencies
            if not getattr(model, dependency.manifest_field, "")
            or not Path(getattr(model, dependency.manifest_field, "")).exists()
            for capability in dependency.capabilities
        }
        capabilities = "  ".join(
            f"{'✗' if capability in unavailable else '✓'} {capability}"
            for capability in spec.capabilities
        )
        dependencies = "  ".join(dependency_lines) or "无"
        detail.set(f"Capabilities: {capabilities}\nDependencies: {dependencies}")
        repair_button.configure(state="normal" if missing and not state.get("busy") else "disabled")

    def close() -> None:
        if state.get("busy"):
            if not messagebox.askyesno(
                "确认关闭", "模型任务仍在运行，确定停止并关闭窗口吗？", parent=dialog
            ):
                return
            runner = state.get("runner")
            if isinstance(runner, ProcessRunner):
                runner.cancel()
            app.model_download_runner = None
            state["busy"] = False
            state["runner"] = None
        dialog.destroy()

    tree.bind("<<TreeviewSelect>>", lambda _event: update_selection())
    tree.bind("<Button-1>", clear_selection_on_blank, add="+")
    download_button.configure(command=start_download)
    uninstall_button.configure(command=start_uninstall)
    repair_button.configure(command=start_repair)
    dialog.protocol("WM_DELETE_WINDOW", close)
    refresh()
    app._center_dialog(dialog, 900, 610)
