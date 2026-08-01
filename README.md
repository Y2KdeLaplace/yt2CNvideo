# scip 2.1.3

scip 是一个基于 Tk 的跨平台 YouTube 视频中文化工具：

1. 用 yt-dlp 下载视频，优先取视频作者提供的字幕，没有时再取 YouTube 自动字幕。
2. 用 Qwen3-ASR 从音轨生成一份独立识别字幕。
3. 用一个 OpenAI 兼容语言模型比对两份字幕、修复原文并翻译为简体中文。
4. 把中文字幕改写成适合朗读的文本，生成中文语音，再由 ffmpeg 对齐时间、替换音轨并嵌入字幕。

ASR 不会被当作 YouTube 字幕缺失时的自动后备。修复步骤同时需要下载字幕与识别字幕；缺少下载字幕会明确报错。

> 只下载和处理你有权使用的内容，并遵守 YouTube 条款、模型许可证和所在地法律。

## 两阶段安装

### 1. 基础安装

基础安装只包含界面、下载和轻量配置，可直接打开查看程序，不会安装大模型环境。

需要：

- [uv](https://docs.astral.sh/uv/)
- Python 3.10 或更高版本，并带 Tk
- yt-dlp
- ffmpeg 与 ffprobe

```bash
uv sync
uv run scip
```

快捷启动：

- Windows：双击 `启动程序.bat`
- macOS：首次执行 `chmod +x 启动程序.command`，之后双击
- Linux：`bash start.sh`

### 2. 模型下载与选择

打开顶部“模型”菜单：

- “语言模型”：配置 OpenAI 兼容 API 地址、API Key 和模型名称。开启“保存信息”时全部保存；API Key 会以与当前设备绑定的加密形式写入系统用户配置目录中的 `settings.json`。
- “语音识别模型”：选择并锁定 ASR 模型；点击“管理”打开模型管理窗口。
- “语音生成模型”：选择并锁定 TTS 模型；点击“管理”打开模型管理窗口。

模型管理窗口会列出当前已安装的模型；选中表格中的模型后可直接卸载。模型锁定状态会在本次程序运行期间保留，关闭模型窗口后再次打开仍然有效；退出并重新启动程序后会回到待锁定状态。处理页只会调用本次会话中已锁定的模型，未锁定表示模型尚未选择完成。锁定时将鼠标放在模型选项上可查看实际路径。

Qwen 官方模型从 ModelScope 下载且不需要指定单个权重文件；MLX、GGUF 和“其他模型”优先使用 Hugging Face 官方 `hf download`，失败后依次尝试镜像与 hfd。下载 GGUF 或其他 Hugging Face 模型时，程序先检查仓库文件：单一 GGUF 自动选择，多个量化版本则由用户选择具体版本；分片 GGUF 会作为一个版本成组下载。下载过程会在模型窗口的运行日志中持续更新缓存写入进度；下载失败或停止时会清理本次产生的不完整模型目录，避免它被误认为可用模型。下载尚未结束时关闭窗口会先询问是否停止。

在中国大陆网络环境中，下载非 Qwen 官方的 Hugging Face 模型前可设置镜像：

macOS / Linux：

```bash
export HF_ENDPOINT=https://hf-mirror.com
uv run scip
```

Windows PowerShell：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
uv run scip
```

请自行确认镜像可信。该变量只影响从当前终端启动的程序。

## 平台模型

### macOS Apple Silicon

程序直接使用 MLX 模型及 `mlx-audio` 的调用方式：

- ASR：`mlx-community/Qwen3-ASR-0.6B-8bit`
- ASR 对齐：`mlx-community/Qwen3-ForcedAligner-0.6B-8bit`
- TTS Base：`mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit`
- TTS CustomVoice：`mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`

uv 会按需准备 Python 3.13 隔离运行环境。实现参考了
[royisme/qwen-speech-mlx](https://github.com/royisme/qwen-speech-mlx)
的加载与推理流程，并配合上述模型仓库的接口。
下载 Mac ASR 模型时会同时准备 MLX Forced Aligner。识别文本先取得逐词
时间戳，再按句末标点、真实停顿和字幕可读长度生成自然分段，不使用固定
秒数作为最终字幕边界。

### Windows / Linux

每种模型都可选择官方版本或 GGUF 版本：

- 官方 ASR：ModelScope 的 `Qwen/Qwen3-ASR-0.6B`，同时安装 `Qwen/Qwen3-ForcedAligner-0.6B` 取得逐词时间戳，并保留原始空格与标点生成自然字幕段。
- GGUF ASR：`cstr/qwen3-asr-0.6b-GGUF`（CrispASR 兼容转换），安装时同时准备 Silero VAD 和 `cstr/qwen3-forced-aligner-0.6b-GGUF`。识别后通过逐词对齐、停顿与句末标点生成字幕时间轴，运行时无需再下载辅助模型。
- 官方 TTS Base：ModelScope 的 `Qwen/Qwen3-TTS-12Hz-0.6B-Base`。
- 官方 TTS CustomVoice：ModelScope 的 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`。
- GGUF TTS：Base 与 CustomVoice 均可选择，并自动安装配套 tokenizer。

GGUF 使用按当前系统下载的 CrispASR 预编译运行时。选择“其他模型”后可输入 Hugging Face 的 `owner/model`，例如 `seanghay/Qwen3-ASR-0.6B-Khmer`。

模型若以本机服务运行，会在任务开始前启动、就绪后执行，并在任务完成、失败或取消后终止；无需用户手动管理服务。

Qwen3-TTS CustomVoice 使用内置中文音色 Vivian。Base 模型用于声音克隆，锁定 Base 后必须在语音生成模型窗口选择参考 WAV，并填写与音频内容一致的文本；该配置跟随锁定的模型保存。

## 应用数据位置

运行设置写入操作系统的用户配置目录：

- Windows：`%APPDATA%\YouTube Video Localizer\settings.json`
- macOS：`~/Library/Application Support/YouTube Video Localizer/settings.json`
- Linux：`${XDG_CONFIG_HOME:-~/.config}/youtube-video-localizer/settings.json`

为保留升级前的设置，scip 继续使用上述原配置目录名称，不会在启动时执行一次性迁移。

“关于 → 缓存目录”只管理本应用的运行组件与可变应用数据，例如 CrispASR 运行组件。可选择“设置缓存目录”指定位置，程序会将旧的应用缓存完整迁移到新位置；也可选择“打开缓存目录”直接查看。Hugging Face 与 ModelScope 下载的模型保持各自原有的标准缓存位置，便于与其他工具共用，不会被本应用移动或重定向。

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

顶部“关于”菜单提供缓存目录、版本与更新功能；“更新”读取本项目 GitHub 最新 Release 并比较版本号。

## 开发验证

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q videodub tests launch_app.pyw
```

下载视频、字幕、运行设置、API Key、模型、模型环境和输出文件均不得提交到 Git。
