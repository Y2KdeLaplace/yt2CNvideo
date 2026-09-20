from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any


def build_download_tab(app: Any) -> None:
    """Build the download feature UI against the host application."""
    link_box = ttk.LabelFrame(app.download_tab, text="链接", padding=10)
    link_box.pack(fill="x")
    app.url_text = tk.Text(link_box, height=8, wrap="word")
    app.url_text.pack(fill="x")
    app.url_placeholder = ttk.Label(
        app.url_text,
        text="粘贴一个或多个 YouTube 视频或播放列表链接，每行一个",
        foreground="#808080",
    )
    app.url_placeholder.place(x=7, y=6)
    app.url_placeholder.bind("<Button-1>", lambda _e: app.url_text.focus_set())
    app.url_text.bind("<KeyRelease>", app._update_url_placeholder)
    app.url_text.bind("<FocusIn>", app._update_url_placeholder)
    app.url_text.bind("<Button-3>", app._show_url_menu)
    app.url_text.bind("<Button-2>", app._show_url_menu)

    settings = ttk.LabelFrame(app.download_tab, text="下载设置", padding=10)
    settings.pack(fill="x", pady=(10, 0))
    ttk.Radiobutton(
        settings,
        text="单个视频",
        variable=app.link_type,
        value="single",
    ).grid(row=0, column=0, sticky="w")
    ttk.Radiobutton(
        settings,
        text="播放列表",
        variable=app.link_type,
        value="playlist",
    ).grid(row=0, column=1, sticky="w", padx=(18, 0))
    ttk.Label(settings, text="字幕语言").grid(
        row=1,
        column=0,
        sticky="w",
        pady=(9, 0),
    )
    ttk.Entry(settings, textvariable=app.subtitle_languages).grid(
        row=1,
        column=1,
        columnspan=3,
        sticky="ew",
        padx=(10, 0),
        pady=(9, 0),
    )
    settings.columnconfigure(3, weight=1)
    controls = ttk.Frame(app.download_tab)
    controls.pack(fill="x", pady=(10, 0))
    app.download_button = ttk.Button(
        controls,
        text="下载",
        command=app._start_download,
        style="Main.TButton",
    )
    app.download_button.pack(side="right")
