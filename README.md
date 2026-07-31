# YouTube Video Localizer

一个基于 Tk 的跨平台 YouTube 视频中文化工具：

1. 用 yt-dlp 下载视频和指定语言的 YouTube 字幕。
2. 用 Qwen3-ASR 从视频音轨提取第二份带时间轴的字幕。
3. 用一个 OpenAI 兼容文本模型比对两份字幕，修复原语言字幕并翻译为简体中文。
4. 把中文字幕改写成适合朗读的临时字幕，再生成中文语音、替换原音轨并嵌入中文字幕。

程序不会把 Qwen3-ASR 当作 YouTube 字幕缺失时的自动后备：字幕修复要求 YouTube 字幕和 ASR 字幕都存在。下载字幕失败或触发限流时，视频仍会继续下载。

> 只下载和处理你有权使用的内容，并遵守 YouTube 条款和所在地法律。

## 目录布局

默认工作路径是项目根目录下的 `work`：

```text
work/
├── 视频与 YouTube 字幕
├── 视频.asr.srt
└── output/
    ├── 视频.corrected.srt
    ├── 视频.zh-CN.srt
    ├── 视频.speech.zh-CN.srt
    ├── 视频.subtitle-report.json
    └── 视频.中文配音.mp4
```

播放列表的子目录结构会在 `output` 中保持一致。更换工作路径时，程序会把当前工作目录中的内容合并迁移到新路径，文件冲突按新任务的默认规则覆盖。

## 使用 uv 安装

需要：

- [uv](https://docs.astral.sh/uv/)
- Python 3.10 或更高版本，并带 Tk
- yt-dlp
- ffmpeg 和 ffprobe

安装主程序：

```bash
uv sync
uv run python -m videodub
```

启动方式：

- Windows：双击 `启动程序.bat`
- macOS：首次执行 `chmod +x 启动程序.command`，之后双击
- Linux：执行 `bash start.sh`
- 通用：`uv run python -m videodub`

常用系统依赖：

```bash
# macOS
brew install python-tk yt-dlp ffmpeg

# Ubuntu / Debian
sudo apt install python3-tk ffmpeg
uv tool install "yt-dlp[default]"
```

## Qwen3-ASR 与 Qwen3-TTS

应用通过两个本机 HTTP 服务使用 Qwen：

- ASR：`http://127.0.0.1:9956`
- TTS：`http://127.0.0.1:9955`

打开“字幕提取”或“中文配音”页时会自动检测服务，并在“模型路径”中显示服务报告的模型名称。

### 下载模型前设置 Hugging Face 镜像

在中国大陆网络环境中，建议在**首次启动 Qwen 服务前**设置：

macOS / Linux：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Windows PowerShell：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

这个设置只影响当前终端。确认镜像可信，并遵守模型许可证。

### macOS Apple Silicon

macOS 使用 [royisme/qwen-speech-mlx](https://github.com/royisme/qwen-speech-mlx)，它要求 Apple Silicon、Python 3.13 和 uv：

```bash
git clone https://github.com/royisme/qwen-speech-mlx.git
cd qwen-speech-mlx
uv sync
```

分别打开两个终端：

```bash
# 终端 1
export HF_ENDPOINT=https://hf-mirror.com
uv run qwen-asr

# 终端 2
export HF_ENDPOINT=https://hf-mirror.com
# 让服务返回绝对音频路径，便于主程序读取
export QWEN_OUTPUT_DIR="$PWD/output"
uv run qwen-tts
```

参考项目当前使用：

- `mlx-community/Qwen3-ASR-0.6B-8bit`
- `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit`

### Windows / Linux

Windows 和 Linux 使用本项目附带的兼容服务，以及 Qwen 官方 Transformers 包。建议 Python 3.12；NVIDIA GPU 明显更快，CPU 也可启动但速度可能很慢。

当前官方 `qwen-asr` 和 `qwen-tts` 固定了不同的 Transformers 小版本，因此不能装进同一个 Python 环境。下面使用 uv 的隔离环境分别启动，主程序环境不会被大模型依赖污染。

分别打开两个终端：

```bash
# 终端 1
uv run --isolated --python 3.12 --extra qwen-asr-service videodub-qwen-asr

# 终端 2
uv run --isolated --python 3.12 --extra qwen-tts-service videodub-qwen-tts
```

默认模型：

- ASR：`Qwen/Qwen3-ASR-0.6B`
- 时间对齐：`Qwen/Qwen3-ForcedAligner-0.6B`
- TTS：`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`

时间对齐模型用于生成可靠的 SRT 时间轴。可以用环境变量 `QWEN_ASR_MODEL`、`QWEN_ALIGNER_MODEL` 和 `QWEN_TTS_MODEL` 改成本地模型目录或其他兼容模型。

## 视频如何交给 ASR

Qwen3-ASR 接收的是音频，不直接处理视频画面。程序不会先生成 MP3：它让 ffmpeg 从视频音轨直接解码成临时的 16 kHz 单声道无损 WAV，识别完成后立即删除。这省掉了 MP3 有损编码，也不会在工作目录留下中间音频。

## 四个页面

### YouTube 视频下载

- 每行粘贴一个链接。
- 选择单视频或播放列表。
- “字幕语言”非空时始终请求字幕；默认是 `en-orig,en`。
- 字幕失败或限流时跳过字幕并继续下载视频。
- 所有下载默认覆盖同名文件。

### 字幕提取

- 选择已下载的视频。
- 确认 Qwen3-ASR 服务可用并启用功能。
- ASR 字幕写在视频旁边，文件名以 `.asr.srt` 结尾。

### 字幕修复与翻译

- 列表只显示同时拥有 YouTube 字幕和 `.asr.srt` 的视频。
- 文本模型先判断主题并建立术语表，再逐条比对两份字幕。
- 时间轴和字幕 ID 以 YouTube 字幕为准；模型输出会在写盘前验证。
- 修复字幕和中文字幕写入 `output`。
- “保存信息”默认开启，只保存 API 地址和模型名称；API Key 始终只保留在内存或 `OPENAI_API_KEY` 环境变量中。

### 中文配音

- 文本模型先把中文字幕改写成适合朗读的临时字幕，例如把公式改成自然的中文运算表达。
- 启用 Qwen3-TTS 时使用本机 Qwen 服务；未启用时保留轻量的 Edge TTS 默认方案。
- 每条语音按原字幕时间段对齐；过长语音会加速，较短语音会补静音。
- 最终视频替换原音轨，并嵌入原始中文字幕。

## OpenAI 兼容接口

程序调用 `POST /v1/chat/completions`。API 地址可以是公网、localhost 或局域网服务，例如：

```text
http://127.0.0.1:8000/v1
```

同一个模型负责主题分析、双字幕修复、中文翻译和配音文本改写。整个流程只发送文字，不上传视频画面。

## 开发与验证

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q videodub tests launch_app.pyw
```

项目约定见 `AGENTS.md`。下载视频、字幕、运行设置、API Key、模型和输出视频均不应提交到 Git。
