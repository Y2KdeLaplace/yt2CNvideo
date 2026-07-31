# YouTube Video Localizer 2.1

一个基于 Tk 的跨平台 YouTube 视频中文化工具：

1. 用 yt-dlp 下载视频，优先取视频作者提供的字幕，没有时再取 YouTube 自动字幕。
2. 用 Qwen3-ASR 从音轨生成一份独立识别字幕。
3. 用一个 OpenAI 兼容语言模型比对两份字幕、修复原文并翻译为简体中文。
4. 把中文字幕改写成适合朗读的文本，生成中文语音，再由 ffmpeg 对齐时间、替换音轨并嵌入字幕。

ASR 不会被当作 YouTube 字幕缺失时的自动后备。修复步骤同时需要下载字幕与识别字幕；缺少下载字幕会明确报错。

> 只下载和处理你有权使用的内容，并遵守 YouTube 条款、模型许可证和所在地法律。

## 两阶段安装

### 1. 基础安装

基础安装只包含界面、下载、轻量配置和 Edge TTS，可直接打开查看程序，不会安装大模型环境。

需要：

- [uv](https://docs.astral.sh/uv/)
- Python 3.10 或更高版本，并带 Tk
- yt-dlp
- ffmpeg 与 ffprobe

```bash
uv sync
uv run python -m videodub
```

快捷启动：

- Windows：双击 `启动程序.bat`
- macOS：首次执行 `chmod +x 启动程序.command`，之后双击
- Linux：`bash start.sh`

### 2. 模型安装

打开顶部“模型”菜单：

- “语言模型”：配置 OpenAI 兼容 API 地址、API Key 和模型名称。开启“保存信息”时全部保存；API Key 会以与当前设备绑定的加密形式写入 `external/settings.json`。
- “语音识别模型”：选择、安装或卸载 ASR 模型。
- “语音生成模型”：选择、安装或卸载 TTS 模型。

模型、各自独立的 Python 环境和 GGUF 运行时都在 `external` 下。主程序基础环境不会被 PyTorch、MLX 或 GGUF 依赖污染。将鼠标放在已安装模型选项上可查看实际路径。配音默认仍可直接使用轻量的 Edge TTS；选择已安装的 Qwen3-TTS 后自动切换到本地模型，Piper 保留为离线后备。

在中国大陆网络环境中，下载 Hugging Face 模型前可先设置镜像：

macOS / Linux：

```bash
export HF_ENDPOINT=https://hf-mirror.com
uv run python -m videodub
```

Windows PowerShell：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
uv run python -m videodub
```

请自行确认镜像可信。该变量只影响从当前终端启动的程序。

## 平台模型

### macOS Apple Silicon

程序直接使用 MLX 模型及 `mlx-audio` 的调用方式，不依赖第三方 qwen-speech-mlx 包：

- ASR：`mlx-community/Qwen3-ASR-0.6B-8bit`
- TTS：`mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit`

安装按钮会用 uv 建立独立 Python 3.13 环境。实现参考了
[royisme/qwen-speech-mlx](https://github.com/royisme/qwen-speech-mlx)
的加载与推理流程，并配合两个模型仓库的接口。

### Windows / Linux

每种模型都可选择官方版本或 GGUF 版本：

- 官方 ASR：`Qwen/Qwen3-ASR-0.6B`，同时安装 `Qwen/Qwen3-ForcedAligner-0.6B` 生成时间轴。
- GGUF ASR：`handy-computer/Qwen3-ASR-0.6B-gguf`。
- 官方 TTS：`Qwen/Qwen3-TTS-12Hz-0.6B-Base`。
- GGUF TTS：`cstr/qwen3-tts-0.6b-base-GGUF`，同时安装配套 tokenizer。

官方模型分别安装在隔离的 uv 环境中。GGUF 使用按当前系统下载的 CrispASR 预编译运行时。选择“其他模型”后可输入 `owner/model`，例如 `seanghay/Qwen3-ASR-0.6B-Khmer`。

模型若以本机服务运行，会在任务开始前启动、就绪后执行，并在任务完成、失败或取消后终止；无需用户手动管理服务。

## 工作目录

默认工作路径为项目根目录下的 `work`：

```text
work/
├── 视频.mp4
├── 视频.en.srt
├── 视频.asr.srt
└── output/
    ├── 视频.corrected.srt
    ├── 视频.zh-CN.srt
    ├── 视频.speech.zh-CN.srt
    ├── 视频.subtitle-report.json
    └── 视频.中文配音.mp4
```

更换工作路径时，原工作目录的内容会合并迁移并覆盖同名结果，但项目内的 `work` 文件夹始终保留。程序会记住迁移后的路径；如果该路径以后不存在，自动回退到默认 `work`。

下载视频与下载字幕放在视频旁边；除下载产物和 ASR 配对字幕外，修复、翻译、报告及成品都写到工作路径下的 `output`。

## 界面与流程

“视频下载”页支持单视频与播放列表，链接框支持右键粘贴。字幕语言非空时必须尝试下载字幕；作者字幕优先，自动字幕其次。字幕失败或限流时跳过字幕并继续任务。下载按钮在任务开始后会变成停止按钮；停止或失败时会删除本次任务新建的未完成下载目录，不会删除任务开始前已经存在的目录。

“处理”页用表格显示：

- 视频名
- 下载字幕
- 识别字幕
- 修复字幕
- 中文字幕

按行选择一个或多个视频，再组合“提取、修复、翻译、配音”四个步骤。四项默认开启；表格右键菜单可以全选。开启“并行处理”后，可填写一个大于 1 的正整数，同时处理指定数量的视频。运行按钮在任务开始后变为停止按钮。所有标签页共用窗口底部的运行日志，切换标签页或开始新任务都不会清空，任务之间以 `==` 分隔。

ASR 不需要先生成 MP3。程序让 ffmpeg 直接把视频音轨解码为临时的 16 kHz 单声道无损 WAV，识别结束即删除。

字幕修复与翻译只通过纯文本和语言模型交互；模型会判断领域、统一专业名词，并允许在修复字幕中保留 LaTeX 公式。配音前会生成临时朗读字幕，把公式改写为自然口语。语音会压缩或补静音以匹配原字幕时间段。

## 更新与版本

顶部“关于 → 版本”显示当前版本；“关于 → 更新”读取本项目 GitHub 最新 Release 并比较版本号。

## 开发验证

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q videodub tests launch_app.pyw
```

下载视频、字幕、运行设置、API Key、模型、模型环境和输出文件均不得提交到 Git。
