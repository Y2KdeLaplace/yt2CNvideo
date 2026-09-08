# scip 2.1.3

scip 是一个基于 Tk 的跨平台 YouTube 视频中文化工具：

1. 用 yt-dlp 下载视频，优先取视频作者提供的字幕，没有时再取 YouTube 自动字幕。
2. 用 Qwen3-ASR 从音轨生成一份独立识别字幕。
3. 用一个 OpenAI 兼容语言模型比对两份字幕、修复原文并翻译为简体中文。
4. 从可靠声学时间轴构建 SentenceUnit，句级校正和翻译保持同一身份；每句独立配音，适配时长后合成音轨。

处理链路：ASR / 下载字幕 → 构建可靠自然句 → 句级校正 → 句级翻译 → 独立显示字幕层（可选拆分）→ 句级 TTS → 全局 + 局部时长适配 → 时间轴合成 → mux video。

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

“关于 → 缓存目录”只管理本应用的运行组件与可变应用数据，例如 CrispASR 运行组件和配音中间音频。可选择“设置缓存目录”指定位置，程序会将旧的应用缓存完整迁移到新位置；也可选择“打开缓存目录”直接查看。缓存目录下的 `tmp` 在每次打开程序时清空并重建。Hugging Face 与 ModelScope 下载的模型保持各自原有的标准缓存位置，便于与其他工具共用，不会被本应用移动或重定向。

## 工作目录

默认工作路径为项目根目录下的 `work`：

```text
work/
├── 视频.mp4
├── 视频.en.srt
├── 视频.asr.srt
├── 视频.zh-CN.txt
├── 视频.subtitle-report.json
└── output/
    ├── 视频.corrected.sentences.json
    ├── 视频.corrected.srt
    ├── 视频.zh-CN.sentences.json
    ├── 视频.zh-CN.srt
    └── 视频.中文配音.mp4
```

更换工作路径时，原工作目录的内容会合并迁移并覆盖同名结果，但项目内的 `work` 文件夹始终保留。程序会记住迁移后的路径；如果该路径以后不存在，自动回退到默认 `work`。

下载视频、下载字幕、ASR 配对字幕、完整翻译 TXT 和处理报告放在视频旁边；`output` 保存校正/翻译后的句级 JSON、显示 SRT 和配音成品；不再生成旧的 segmentation 调试草稿。

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

字幕修复和翻译共享 `SentenceUnit`：包含稳定 `group_id`、来源 cue 范围、声学 start/end、文本和 spoken/sound 类型。模型始终看到全文上下文，但每批最多输出 24 句；只允许按身份替换文本，禁止丢失、重复、新增或合并身份。遗漏时只补齐缺失句。中文对白以自然口语、人物关系、动作、幽默和节奏为重点，不做摘要。

源时间轴先做严格 QC：结束不晚于开始、时间倒退、超过 100 ms 的重叠、正常语言 cue 不超过 100 ms、纯标点、句内大间隔、超过 12 秒的普通句窗都会被检查。纯标点不独立形成句子，声效单独标记；超过 1 秒的 cue 间隔会终止合句。过长组只能在真实停顿或标点处分开。无法确认的极短语言片段直接报错并显示 cue、文本、start/end 和异常类型，不把非法时间改成 1 ms，也不拉长窗口或用视频总长伪造时间戳。请检查/重新生成 ASR 对齐后再处理。

显示字幕与 TTS 分句相互独立：目前显示层每个自然句输出一条 SRT；以后可在显示层拆分，但配音只读取相邻的 `.sentences.json`，不会从显示 cue 重建句子。只有旧 SRT、没有句级数据的任务，需要重新执行校正/翻译。修改显示 SRT 不会修改配音文本，要重新翻译或修改并验证句级数据。

TTS 不再使用 Forced Aligner，不生成完整长音频后重新找句界。Forced Aligner 若存在，仅用于 ASR 声学时间戳。TTS 输入统一做 Unicode NFKC、空白清洗并移除 `[声效标签]`，空文本和纯标点不调用模型，纯声效只保留显示字幕。每句生成 `sentence-00000.raw.wav` 等独立文件；超长文本只在同一句内部受控拆块再合并。

全局 duration ratio 为目标句窗总长 / 原始音频总长，统计值限制在 0.80–1.20；实际适配优先压缩长句，不用大于 1 的因子拉长音频。已经短于时间窗的句子保持原速，剩余时间留静音；长句局部 ratio 限制在 0.90–1.00。使用 FFmpeg `atempo` 保持音高，句间最多 15 ms 交叠，生成与视频等长的单声道音轨。超过受控压缩能力的句子仍可能产生串行延迟，日志会报告；仅最后一句允许在视频结尾截断。

MLX 空 iterable 转为明确错误，不再穿透 `StopIteration`。生成最多两次尝试，由客户端统一重试一次，失败会显示句序号、总数、来源 cue、清洗后的完整文本、模型和后端；不以静音代替对白。批次中成功的音频保留，只重试失败句。

句级缓存位于应用缓存的 `tts-sentences-v2`，独立于每次清理的 `tmp`。缓存键包括清洗后文本、模型/后端/语言、speaker、实际参考声音内容和参考文本、模型文件版本信息及生成配置版本。只有完整且非空的 WAV 才原子写入缓存。第 51 句失败后再次运行可以复用前 50 句；模型、声音或文本改变时重新生成。旧版本无有效键的临时音频不作为可信缓存。

## 更新与版本

顶部“关于”菜单提供缓存目录、版本与更新功能；“更新”读取本项目 GitHub 最新 Release 并比较版本号。

## 开发验证

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q videodub tests launch_app.pyw
```

下载视频、字幕、运行设置、API Key、模型、模型环境和输出文件均不得提交到 Git。
