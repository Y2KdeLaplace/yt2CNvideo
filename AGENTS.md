# AGENTS.md

## Product invariants

- The active flow is: yt-dlp download → YouTube English subtitles → one OpenAI-compatible model for repair/translation → Chinese SRT → Chinese TTS → ffmpeg mux.
- Do not add Whisper or another ASR fallback. Missing YouTube subtitles should produce a clear error.
- Subtitle AI uses one model field. `subtitle_use_vision` only controls whether suspect screenshots are attached to that same model.
- Keep the default UI simple. Batch size, context radius, confidence threshold, and screenshot count are internal defaults, not normal user controls.
- The application must remain usable on Windows, macOS, and Linux. Do not introduce an OS-only dependency without a portable fallback.
- TTS defaults to Edge TTS. Piper is the offline fallback. Avoid large model stacks such as CosyVoice.
- Never commit downloaded videos, generated subtitles, API keys, runtime settings, TTS models, or output videos.

## Read only what the task needs

- UI or orchestration: `videodub/ui.py`, then the one module used by the affected tab.
- Download behavior: `videodub/downloader.py`, `videodub/media.py`, `videodub/config.py`.
- Subtitle repair/translation: `videodub/subtitle_workflow.py`, `videodub/openai_compatible.py`, `videodub/subtitles.py`.
- Dubbing: `videodub/tts.py`, `videodub/media.py`, `videodub/subtitles.py`.
- Cross-platform launch/tool discovery: `videodub/platform_utils.py`, root launch scripts.
- Do not scan `yt-video/`, `bl-video/`, caches, or generated reports unless the task explicitly concerns a specific artifact.

## Commands

```bash
python -m unittest discover -s tests -v
python -m compileall -q videodub tests launch_app.pyw
python -m videodub
```

Run the tests after behavior changes. GUI work also needs a hidden-window smoke test when the environment supports Tk.

## Code conventions

- Python 3.10+, standard library first; add dependencies only when they materially simplify the user workflow.
- Use `pathlib.Path`, argument-list subprocess calls, UTF-8, and platform-neutral paths.
- API keys stay in memory or environment variables and must never be written to `settings.json`.
- External commands go through `ProcessRunner` so cancellation and logging keep working.
- Preserve subtitle IDs and timestamps. Model output must be validated before writing files.
