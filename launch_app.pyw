from __future__ import annotations

"""Windowless Windows launcher with a visible startup error dialog."""

import ctypes
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ERROR_LOG = ROOT / "startup-error.log"


try:
    from videodub.ui import main

    main()
except BaseException:
    ERROR_LOG.write_text(traceback.format_exc(), encoding="utf-8")
    ctypes.windll.user32.MessageBoxW(
        0,
        f"Application startup failed.\n\nDetails were saved to:\n{ERROR_LOG}",
        "YouTube Video Localizer",
        0x10,
    )
