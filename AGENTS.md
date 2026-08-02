# AGENTS.md

## Product invariants

- The active flow is: yt-dlp download → YouTube English subtitles + Qwen3-ASR/Forced Aligner → one OpenAI-compatible model for repair/translation → translated SRT → Qwen3-TTS → ffmpeg mux.
- Do not add Whisper or another ASR fallback. Missing YouTube subtitles should produce a clear error.
- Subtitle AI uses one model field. `subtitle_use_vision` only controls whether suspect screenshots are attached to that same model.
- Keep the default UI simple. Batch size, context radius, confidence threshold, and screenshot count are internal defaults, not normal user controls.
- The application must remain usable on Windows, macOS, and Linux. Do not introduce an OS-only dependency without a portable fallback.
- TTS uses Qwen3-TTS Base or CustomVoice. Avoid unrelated model stacks such as Edge TTS, Piper, or CosyVoice.
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
- Model output must be validated before writing files.

## Subtitle timeline invariants

- Acoustic word alignment is the highest-priority timing evidence. Never move, extend, or proportionally invent ASR timestamps merely to force a duration target.
- `1.75–4.8 s` is a soft readability range for splitting long sentences at real punctuation or pauses, never a hard minimum or maximum.
- Corrected subtitles use the ASR timeline when ASR exists. Text repair may merge adjacent ASR cues, but it must absorb their complete time span: the merged cue starts at the first cue start, ends at the last cue end, concatenates the corrected text, and receives a new consecutive ID.
- Never leave an empty cue, silently discard a cue time span, or keep non-consecutive IDs after merging. Saved and in-memory cue IDs must describe the same sequence.
- Full-document translation operates on timestamp-free sentence groups with stable `group_id` values. If the model omits groups, retry only the missing groups with full-document context.
- Do not request one unbounded model response containing every timed subtitle cue. After sentence-group translation, generate timed target-language cues in bounded batches of complete groups; never split a sentence group across batches.
- Do not ask the segmentation model to reproduce global cue IDs or group IDs. Pass ordered `source_parts` and require an equally sized positional `parts` array; map array positions to cue timestamps locally.
- A one-cue sentence receives its translated group text directly. For a multi-cue sentence, the model may return empty positions only when their complete time spans are absorbed into adjacent non-empty target cues. The result must have consecutive IDs, no empty cues, full time-span coverage within each group, and concatenated text exactly matching the translated group.
- Segmentation-model parse or schema failures must not abort a completed translation. After bounded retries, fall back deterministically to one target cue per complete sentence group, spanning that group's first start through last end. Once a batch exhausts its retries, use that fallback for all remaining batches in the same translation instead of repeating known-invalid model calls.
- Video download and subtitle/dubbing processing are independent foreground tasks. Keep separate workers, cancellation runners, button states, and completion events; both may append to the shared run log concurrently.
- MLX Qwen3-TTS should use small bounded batches with one shared voice reference. Keep a sequential fallback for GGUF and model versions without batch support; batch size remains an internal default.
