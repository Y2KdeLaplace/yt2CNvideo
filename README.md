# YouTube Video Localizer

一个小型、跨平台的 YouTube 视频中文化工具：

1. 使用 yt-dlp 下载视频和 YouTube 英文字幕。
2. 使用一个 OpenAI 兼容模型校正英文字幕并翻译为简体中文。
3. 根据中文字幕生成中文语音，用 ffmpeg 替换或混合原音轨。

支持 Windows、macOS 和 Linux。不包含 Whisper，不会重复进行语音识别。

> 只下载和处理你有权使用的内容，并遵守 YouTube 条款和所在地法律。

## 快速安装

需要 Python 3.10+、Tk、yt-dlp、ffmpeg 和 ffprobe。

```bash
python -m pip install -e .
```

Windows 当前会自动发现 `D:\software` 中的 yt-dlp 和 ffmpeg。其他系统会从 PATH、
Homebrew、`~/.local/bin` 等常见位置查找。

启动方式：

- Windows：双击 `启动程序.bat`
- macOS：首次执行 `chmod +x 启动程序.command`，之后双击运行
- Linux：执行 `bash start.sh`
- 通用方式：`python -m videodub`

macOS 可以使用：

```bash
brew install python-tk yt-dlp ffmpeg
python3 -m pip install -e .
```

Ubuntu/Debian 可以使用：

```bash
sudo apt install python3 python3-tk ffmpeg
python3 -m pip install -U "yt-dlp[default]"
python3 -m pip install -e .
```

## 使用

### 1. 下载

在“1. 下载任务”中：

- 选择下载文件夹；
- 每行粘贴一个视频链接，或粘贴标准 YouTube 播放列表链接；
- 选择链接类型；
- 点击“开始下载”。

默认只请求 `en-orig,en`，减少字幕 HTTP 429。字幕失败不会中断视频下载。

### 2. 修复并翻译字幕

在“2. 字幕修复与翻译”中只需要填写：

- OpenAI 兼容 API 地址；
- API Key，本地无鉴权服务可留空；
- 一个模型名称；
- 是否“图像使用”；
- 是否覆盖已有结果。

模型默认只接收文字。“图像使用”开启后，程序只对高可能错误的字幕时间段截取少量
画面，并把截图发送给同一个多模态模型。批量、上下文数量、疑点阈值和截图数量都是
内部默认值，不再放在普通 UI 中。

输出位于原视频旁：

- `.corrected.srt`：校正后的英文字幕；
- `.zh-CN.srt`：简体中文字幕；
- `.subtitle-report.json`：领域、术语、疑点、修改和 token 统计；
- `.subtitle-evidence/`：启用图像时生成的疑点截图。

### 3. 中文配音

在“3. 中文配音”中选择已有 `.zh-CN.srt` 的视频，然后：

- 选择 Edge TTS 或 Piper；
- 选择中文声音；
- 选择替换原音轨，或保留少量原声混合；
- 点击“生成中文配音视频”。

程序逐条生成中文语音，按字幕时间自动对齐；过长语音会自动加速，较短语音会补静音。
输出视频位于 `bl-video`。

## 轻量 TTS 选择

| 方案 | 状态 | 优点 | 局限 |
| --- | --- | --- | --- |
| Edge TTS | 已接入、默认 | 声音自然、客户端很小、免费、无需 API Key、跨平台 | 必须联网；使用非正式 Edge 在线服务，服务变化时可能需要升级包 |
| Piper | 已接入、离线备用 | 完全离线、跨平台、中文中等模型约几十 MB、启动后资源占用小 | 自然度通常低于 Edge；首次需要安装运行时和下载声音模型 |
| sherpa-onnx | 候选，暂未接入 | 完全离线、跨平台、中文模型丰富 | 配置文件更多，常用中文模型约 100–160 MB，接入复杂度高于 Piper |

Edge TTS 项目提供 Python 包和在线 TTS，无需 Edge 浏览器或 API Key：
[rany2/edge-tts](https://github.com/rany2/edge-tts)。

离线 Piper 安装：

```bash
python -m pip install -e ".[offline-tts]"
python -m piper.download_voices zh_CN-huayan-x_low
```

然后在 UI 中选择下载得到的 `.onnx`。每个 Piper 声音有自己的模型许可，分发前应检查
对应的 `MODEL_CARD`。Piper 文档：
[OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)。

sherpa-onnx 的中文模型列表：
[sherpa-onnx TTS models](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html)。

本项目不会引入 CosyVoice 一类体积较大、环境复杂的 TTS 栈。

## API 地址

程序使用 OpenAI 风格的 `POST /v1/chat/completions`：

- 公网 API：平台提供的兼容地址；
- 当前电脑：`http://127.0.0.1:8000/v1/`；
- 局域网模型：例如 `http://192.168.1.20:8000/v1/`。

API Key 只保存在当前窗口内，不写入 `settings.json`。

## 开发

```bash
python -m unittest discover -s tests -v
python -m compileall -q videodub tests launch_app.pyw
```

项目约定和最小文件读取路线见 `AGENTS.md`。

主要模块：

```text
videodub/ui.py                 Tk 桌面界面和三个阶段的编排
videodub/downloader.py         yt-dlp 下载
videodub/subtitle_workflow.py  领域分析、疑点定位、截图、修复与翻译
videodub/openai_compatible.py  OpenAI 兼容客户端
videodub/tts.py                Edge/Piper 配音、时间对齐和音轨合成
videodub/platform_utils.py     跨平台工具发现和文件管理器
```

## 上传 GitHub

仓库已经初始化。创建一个空的 GitHub 仓库后执行：

```bash
git add .
git commit -m "Initial release"
git branch -M main
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main
```

`settings.json`、API Key、视频、字幕、TTS 模型和输出视频均已加入 `.gitignore`。
