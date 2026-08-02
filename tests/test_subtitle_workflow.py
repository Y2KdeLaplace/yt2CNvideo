import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from videodub.config import AppConfig
from videodub.media import VideoJob
from videodub.openai_compatible import (
    OpenAICompatibleClient,
    chat_completions_endpoint,
    openai_base_url,
)
from videodub.runner import ProcessRunner
from videodub.subtitle_workflow import (
    SUBTITLE_SEGMENTATION_SYSTEM,
    TRANSCRIPT_REPAIR_SYSTEM,
    TRANSLATION_SYSTEM,
    SpeechSubtitleWorkflow,
    SubtitleRepairWorkflow,
)
from videodub.subtitles import Cue, find_source_subtitle, read_srt


class _MockChatServer:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.authorization_headers: list[str] = []

        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                owner.calls.append(payload)
                owner.authorization_headers.append(
                    self.headers.get("Authorization", "")
                )
                if (
                    payload["model"] == "legacy-model"
                    and "max_completion_tokens" in payload
                ):
                    body = json.dumps(
                        {"error": {"message": "unknown max_completion_tokens"}}
                    ).encode("utf-8")
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if (
                    payload["model"] == "no-thinking-control"
                    and "thinking" in payload
                ):
                    body = json.dumps(
                        {"error": {"message": "unknown parameter: thinking"}}
                    ).encode("utf-8")
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                system = payload["messages"][0]["content"]
                if "专业术语分析专家" in system:
                    content = json.dumps(
                        {
                            "domain": "computational neuroscience",
                            "subdomains": ["spiking neurons"],
                            "summary": "A short lecture.",
                            "glossary": [
                                {
                                    "term": "spike train",
                                    "preferred_zh": "脉冲序列",
                                    "notes": "",
                                }
                            ],
                        }
                    )
                elif "视频文稿校对专家" in system:
                    content = json.dumps(
                        {
                            "corrected_transcript": (
                                "A spike train. This is the next sentence."
                            )
                        }
                    )
                elif "专业的视频文稿译者" in system:
                    requested = json.loads(payload["messages"][1]["content"])
                    groups = requested["source_sentence_groups"]
                    if payload["model"] == "split-sentence-model":
                        translations = ["如果我们现在出发，我们就能赶上飞机。"]
                    elif payload["model"] == "large-batch-model":
                        translations = [
                            f"译{group['group_id']}甲。译{group['group_id']}乙。"
                            for group in groups
                        ]
                    elif payload["model"] == "sentence-boundary-model":
                        translations = [
                            "你知道的？而且只要电源被中断，它就会重启，顺便把历史记录全抹干净了。",
                            "说实话，宝贝。",
                            "是色情内容吧？",
                            "色情内容。",
                            "什么？",
                        ]
                    elif payload["model"] in {
                        "omitting-segmentation-model",
                        "invalid-timing-model",
                    }:
                        translations = ["一个脉冲序列。"]
                    else:
                        translations = []
                        for group in groups:
                            source_text = group["source_text"]
                            if requested["target_language"] == "German":
                                translated = (
                                    "Eine Impulsfolge. Das ist der nächste Satz."
                                    if "next sentence" in source_text
                                    and "spike train" in source_text
                                    else "Eine Impulsfolge."
                                    if "spike train" in source_text
                                    else "Das ist der nächste Satz."
                                )
                            else:
                                translated = (
                                    "一个脉冲序列。这是下一句话。"
                                    if "next sentence" in source_text
                                    and "spike train" in source_text
                                    else "一个脉冲序列。"
                                    if "spike train" in source_text
                                    else "这是下一句话。"
                                )
                            translations.append(translated)
                    translated_rows = [
                            {
                                "group_id": group["group_id"],
                                "translated_text": translated,
                            }
                            for group, translated in zip(
                                groups,
                                translations,
                                strict=True,
                            )
                        ]
                    if payload["model"] == "omitting-group-model":
                        translation_calls = [
                            call
                            for call in owner.calls
                            if "专业的视频文稿译者"
                            in call["messages"][0]["content"]
                        ]
                        if len(translation_calls) == 1:
                            translated_rows.pop()
                    content = json.dumps(translated_rows)
                elif "字幕语义分段专家" in system:
                    requested = json.loads(payload["messages"][1]["content"])
                    source = requested["source_subtitles"]
                    translated = "".join(
                        group["translated_text"]
                        for group in requested["translated_sentence_groups"]
                    )
                    if payload["model"] == "split-sentence-model":
                        parts = ["如果我们现在出发，", "我们就能", "赶上飞机。"]
                    elif payload["model"] == "large-batch-model":
                        parts = [
                            part
                            for group in requested["translated_sentence_groups"]
                            for part in (
                                f"译{group['group_id']}甲。",
                                f"译{group['group_id']}乙。",
                            )
                        ]
                    elif payload["model"] == "sentence-boundary-model":
                        parts = [
                            "你知道的？而且只要电源被中断，",
                            "它就会重启，顺便把",
                            "历史记录全抹干净了。",
                        ]
                    elif payload["model"] in {
                        "omitting-segmentation-model",
                        "invalid-timing-model",
                    }:
                        parts = ["一个", "脉冲序列。"]
                    elif len(source) == 1:
                        parts = [translated]
                    elif requested["target_language"] == "German":
                        parts = ["Eine Impulsfolge.", "Das ist der nächste Satz."]
                    else:
                        parts = ["一个脉冲序列。", "这是下一句话。"]
                    if payload["model"] == "formatting-model":
                        parts = [part.replace("。", "") for part in parts]
                    segmented_rows = [
                        {
                            "id": cue["id"],
                            "sentence_group": cue["sentence_group"],
                            "translated_text": part,
                        }
                        for cue, part in zip(source, parts, strict=True)
                    ]
                    if payload["model"] == "omitting-segmentation-model":
                        segmentation_calls = [
                            call
                            for call in owner.calls
                            if "字幕语义分段专家"
                            in call["messages"][0]["content"]
                        ]
                        if len(segmentation_calls) == 1:
                            segmented_rows.pop(0)
                    if payload["model"] == "invalid-timing-model":
                        segmented_rows[-1]["id"] = "invalid"
                    content = json.dumps(segmented_rows)
                elif "讲解稿改写专家" in system:
                    content = json.dumps(
                        [
                            {"id": 1, "speech_text": "x i 从 i 为 1 到 10 求和。"},
                        ]
                    )
                else:
                    self.send_error(400)
                    return
                body = json.dumps(
                    {
                        "choices": [{"message": {"content": content}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1/"

    def __enter__(self) -> "_MockChatServer":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class SubtitleWorkflowTests(unittest.TestCase):
    def test_openai_compatible_endpoint_variants(self) -> None:
        self.assertEqual(
            chat_completions_endpoint("http://localhost:8000/v1/"),
            "http://localhost:8000/v1/chat/completions",
        )
        self.assertEqual(
            chat_completions_endpoint("http://mac-mini.local:8000"),
            "http://mac-mini.local:8000/chat/completions",
        )
        self.assertEqual(
            openai_base_url("https://api.deepseek.com/chat/completions"),
            "https://api.deepseek.com",
        )
        self.assertEqual(
            chat_completions_endpoint("https://api.moonshot.cn/v1"),
            "https://api.moonshot.cn/v1/chat/completions",
        )

    def test_deepseek_uses_max_tokens_and_disables_thinking(self) -> None:
        client = OpenAICompatibleClient(
            "https://api.deepseek.com",
            "deepseek-v4-pro",
            "secret",
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=None,
        )
        client._client.chat.completions.create = Mock(return_value=response)

        client.chat("system", "user", max_tokens=123)

        request = client._client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["max_tokens"], 123)
        self.assertNotIn("max_completion_tokens", request)
        self.assertEqual(
            request["extra_body"]["thinking"],
            {"type": "disabled"},
        )

    def test_old_local_server_falls_back_to_legacy_token_field(self) -> None:
        with _MockChatServer() as mock:
            client = OpenAICompatibleClient(mock.base_url, "legacy-model")
            client.chat("你是专业术语分析专家。", "{}", max_tokens=123)
            self.assertIn("max_completion_tokens", mock.calls[0])
            self.assertEqual(mock.calls[1]["max_tokens"], 123)
            self.assertNotIn("max_completion_tokens", mock.calls[1])
            self.assertEqual(
                mock.calls[1]["thinking"],
                {"type": "disabled"},
            )

    def test_unknown_thinking_parameter_is_removed_and_cached(self) -> None:
        with _MockChatServer() as mock:
            client = OpenAICompatibleClient(mock.base_url, "no-thinking-control")
            client.chat("你是专业术语分析专家。", "{}")
            client.chat("你是专业术语分析专家。", "{}")

            self.assertIn("thinking", mock.calls[0])
            self.assertNotIn("thinking", mock.calls[1])
            self.assertNotIn("thinking", mock.calls[2])

    def test_forced_kimi_thinking_model_is_rejected(self) -> None:
        with _MockChatServer() as mock:
            client = OpenAICompatibleClient(mock.base_url, "kimi-k2-thinking")
            with self.assertRaisesRegex(RuntimeError, "强制使用思考模式"):
                client.chat("system", "user")

    def test_translation_uses_full_document_then_builds_timed_subtitles(self) -> None:
        with _MockChatServer() as mock:
            workflow = SubtitleRepairWorkflow(
                AppConfig(
                    subtitle_api_base_url=mock.base_url,
                    subtitle_model="mock-text-model",
                ),
                ProcessRunner(),
            )
            cues = [
                Cue(1, 1000, 3000, "A spike train."),
                Cue(2, 3000, 5000, "This is the next sentence."),
            ]

            translated = workflow.translate(cues, {})

            self.assertEqual(
                translated,
                [
                    Cue(1, 1000, 3000, "一个脉冲序列。"),
                    Cue(2, 3000, 5000, "这是下一句话。"),
                ],
            )
            document_call = next(
                call
                for call in mock.calls
                if "专业的视频文稿译者" in call["messages"][0]["content"]
            )
            document_prompt = json.loads(document_call["messages"][1]["content"])
            self.assertEqual(
                document_prompt["corrected_transcript"],
                "A spike train. This is the next sentence.",
            )
            self.assertNotIn("source_subtitles", document_prompt)
            self.assertEqual(
                document_prompt["source_sentence_groups"],
                [
                    {
                        "group_id": 1,
                        "ids": [1],
                        "source_text": "A spike train.",
                    },
                    {
                        "group_id": 2,
                        "ids": [2],
                        "source_text": "This is the next sentence.",
                    },
                ],
            )
            segmentation_calls = [
                call
                for call in mock.calls
                if "字幕语义分段专家" in call["messages"][0]["content"]
            ]
            self.assertEqual(segmentation_calls, [])
            self.assertEqual(
                [(cue.start_ms, cue.end_ms) for cue in translated],
                [(cue.start_ms, cue.end_ms) for cue in cues],
            )

    def test_translation_uses_target_language_specific_guidance(self) -> None:
        with _MockChatServer() as mock:
            workflow = SubtitleRepairWorkflow(
                AppConfig(
                    subtitle_api_base_url=mock.base_url,
                    subtitle_model="mock-text-model",
                    translation_language="German",
                ),
                ProcessRunner(),
            )
            cues = [Cue(1, 0, 2000, "A spike train. This is the next sentence.")]

            translated = workflow.translate(cues, {})

            self.assertEqual(
                " ".join(cue.text for cue in translated),
                "Eine Impulsfolge. Das ist der nächste Satz.",
            )
            relevant_systems = [
                call["messages"][0]["content"]
                for call in mock.calls
                if "文稿译者" in call["messages"][0]["content"]
                or "语义分段" in call["messages"][0]["content"]
            ]
            self.assertTrue(all("German" in system for system in relevant_systems))

    def test_translation_retries_incomplete_segmentation_ids(self) -> None:
        with _MockChatServer() as mock:
            logs: list[str] = []
            workflow = SubtitleRepairWorkflow(
                AppConfig(
                    subtitle_api_base_url=mock.base_url,
                    subtitle_model="omitting-segmentation-model",
                ),
                ProcessRunner(logs.append),
            )
            cues = [
                Cue(1, 1000, 3000, "A spike"),
                Cue(2, 3000, 5000, "train."),
            ]

            translated = workflow.translate(cues, {})

            self.assertEqual([cue.index for cue in translated], [1, 2])
            segmentation_calls = [
                call
                for call in mock.calls
                if "字幕语义分段专家" in call["messages"][0]["content"]
            ]
            self.assertEqual(len(segmentation_calls), 2)
            retry_prompt = json.loads(
                segmentation_calls[1]["messages"][1]["content"]
            )
            self.assertEqual(retry_prompt["expected_ids"], [1, 2])
            self.assertIn("严格包含且只包含", retry_prompt["retry_instruction"])
            self.assertTrue(any("批次 1 校验失败" in line for line in logs))

    def test_translation_only_retries_missing_sentence_groups(self) -> None:
        with _MockChatServer() as mock:
            workflow = SubtitleRepairWorkflow(
                AppConfig(
                    subtitle_api_base_url=mock.base_url,
                    subtitle_model="omitting-group-model",
                ),
                ProcessRunner(),
            )
            cues = [
                Cue(1, 0, 1000, "A spike train."),
                Cue(2, 1000, 2000, "This is the next sentence."),
            ]

            translated = workflow.translate(cues, {})

            self.assertEqual(len(translated), 2)
            translation_calls = [
                call
                for call in mock.calls
                if "专业的视频文稿译者" in call["messages"][0]["content"]
            ]
            self.assertEqual(len(translation_calls), 2)
            retry = json.loads(translation_calls[1]["messages"][1]["content"])
            self.assertEqual(
                [group["group_id"] for group in retry["source_sentence_groups"]],
                [2],
            )

    def test_large_timeline_is_segmented_in_bounded_batches(self) -> None:
        with _MockChatServer() as mock:
            workflow = SubtitleRepairWorkflow(
                AppConfig(
                    subtitle_api_base_url=mock.base_url,
                    subtitle_model="large-batch-model",
                    subtitle_translation_batch_size=30,
                ),
                ProcessRunner(),
            )
            cues = [
                Cue(
                    index,
                    index * 1000,
                    (index + 1) * 1000,
                    f"part {index}" + ("." if index % 2 == 0 else ""),
                )
                for index in range(1, 87)
            ]

            translated = workflow.translate(cues, {})

            self.assertEqual(len(translated), 86)
            self.assertEqual(
                [(cue.start_ms, cue.end_ms) for cue in translated],
                [(cue.start_ms, cue.end_ms) for cue in cues],
            )
            segmentation_calls = [
                call
                for call in mock.calls
                if "字幕语义分段专家" in call["messages"][0]["content"]
            ]
            self.assertEqual(len(segmentation_calls), 3)
            batch_sizes = [
                len(json.loads(call["messages"][1]["content"])["expected_ids"])
                for call in segmentation_calls
            ]
            self.assertEqual(batch_sizes, [30, 30, 26])

    def test_single_cue_sentence_does_not_need_a_segmentation_call(self) -> None:
        with _MockChatServer() as mock:
            logs: list[str] = []
            workflow = SubtitleRepairWorkflow(
                AppConfig(
                    subtitle_api_base_url=mock.base_url,
                    subtitle_model="formatting-model",
                ),
                ProcessRunner(logs.append),
            )

            translated = workflow.translate(
                [Cue(1, 0, 2000, "A spike train. This is the next sentence.")],
                {},
            )

            self.assertEqual(
                "".join(cue.text for cue in translated),
                "一个脉冲序列。这是下一句话。",
            )
            self.assertFalse(
                any(
                    "字幕语义分段专家" in call["messages"][0]["content"]
                    for call in mock.calls
                )
            )

    def test_invalid_segmentation_saves_debug_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp, _MockChatServer() as mock:
            base = Path(temp) / "output" / "Lecture"
            base.parent.mkdir()
            workflow = SubtitleRepairWorkflow(
                AppConfig(
                    subtitle_api_base_url=mock.base_url,
                    subtitle_model="invalid-timing-model",
                ),
                ProcessRunner(),
            )

            with self.assertRaisesRegex(RuntimeError, "translation-debug.txt"):
                workflow.translate(
                    [
                        Cue(1, 0, 1000, "A spike"),
                        Cue(2, 1000, 2000, "train."),
                    ],
                    {},
                    artifact_base=base,
                )

            self.assertTrue(
                (base.parent / "Lecture.translation-debug.txt").is_file()
            )
            self.assertTrue(
                (base.parent / "Lecture.segmentation-debug.json").is_file()
            )

    def test_split_source_sentence_uses_the_same_source_cue_boundaries(self) -> None:
        with _MockChatServer() as mock:
            workflow = SubtitleRepairWorkflow(
                AppConfig(
                    subtitle_api_base_url=mock.base_url,
                    subtitle_model="split-sentence-model",
                ),
                ProcessRunner(),
            )
            source = [
                Cue(1, 1000, 2200, "If we leave now,"),
                Cue(2, 2200, 3100, "we can"),
                Cue(3, 3100, 4600, "catch the plane."),
            ]

            translated = workflow.translate(source, {})

            self.assertEqual(
                translated,
                [
                    Cue(1, 1000, 2200, "如果我们现在出发，"),
                    Cue(2, 2200, 3100, "我们就能"),
                    Cue(3, 3100, 4600, "赶上飞机。"),
                ],
            )

    def test_sentence_ending_cannot_move_out_of_its_source_cue_group(self) -> None:
        with _MockChatServer() as mock:
            workflow = SubtitleRepairWorkflow(
                AppConfig(
                    subtitle_api_base_url=mock.base_url,
                    subtitle_model="sentence-boundary-model",
                ),
                ProcessRunner(),
            )
            source = [
                Cue(
                    26,
                    59_520,
                    62_240,
                    "you know? And it's just one of those things whenever it gets interrupted",
                ),
                Cue(
                    27,
                    62_240,
                    65_760,
                    "from the power source, it has to reboot and it just totally wipes out the",
                ),
                Cue(28, 65_760, 66_320, "history."),
                Cue(29, 69_120, 69_840, "Be honest, babe."),
                Cue(30, 70_320, 71_200, "It's porn, right?"),
                Cue(31, 73_360, 73_680, "Porn."),
                Cue(32, 74_480, 74_720, "What's?"),
            ]

            translated = workflow.translate(source, {})

            self.assertEqual(translated[2].text, "历史记录全抹干净了。")
            self.assertEqual(translated[3].text, "说实话，宝贝。")
            self.assertEqual(
                [(cue.start_ms, cue.end_ms) for cue in translated],
                [(cue.start_ms, cue.end_ms) for cue in source],
            )
            segmentation_call = next(
                call
                for call in mock.calls
                if "字幕语义分段专家" in call["messages"][0]["content"]
            )
            segmentation_prompt = json.loads(
                segmentation_call["messages"][1]["content"]
            )
            self.assertEqual(
                segmentation_prompt["translated_sentence_groups"][0]["ids"],
                [26, 27, 28],
            )

    def test_subtitle_lookup_treats_video_id_brackets_literally(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "Lecture [abc123].mp4"
            video.touch()
            subtitle = root / "Lecture [abc123].en-orig.srt"
            subtitle.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
                encoding="utf-8",
            )
            self.assertEqual(find_source_subtitle(video), subtitle)

    def test_dual_subtitle_repair_and_translation_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp, _MockChatServer() as mock:
            root = Path(temp)
            video = root / "Lecture [abc123].mp4"
            video.touch()
            source = root / "Lecture [abc123].en-orig.srt"
            source.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nA SPEAK TRAIN.\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\nTHIS IS THE NEXT SENTENCE.\n",
                encoding="utf-8",
            )
            asr = root / "Lecture [abc123].asr.srt"
            asr.write_text(
                "1\n00:00:00,000 --> 00:00:02,000\n"
                "A spike train. This is the next sentence.\n",
                encoding="utf-8",
            )
            config = AppConfig(
                subtitle_api_base_url=mock.base_url,
                subtitle_model="mock-text-model",
                subtitle_detection_batch_size=10,
                subtitle_translation_batch_size=10,
            )
            workflow = SubtitleRepairWorkflow(config, ProcessRunner(), api_key="secret")
            job = VideoJob(video, title="Lecture", generated_dir=root / "output")
            report = workflow.process_job(job)

            corrected = read_srt(job.corrected_subtitle_path)
            chinese = read_srt(job.chinese_subtitle_path)
            self.assertEqual(
                corrected,
                [
                    Cue(
                        1,
                        0,
                        2000,
                        "A spike train. This is the next sentence.",
                    )
                ],
            )
            self.assertEqual(
                chinese,
                [Cue(1, 0, 2000, "一个脉冲序列。这是下一句话。")],
            )
            self.assertEqual(report["domain"]["domain"], "computational neuroscience")
            repair_calls = [
                call
                for call in mock.calls
                if "视频文稿校对专家" in call["messages"][0]["content"]
            ]
            self.assertEqual(len(repair_calls), 1)
            self.assertTrue(
                all(call["model"] == "mock-text-model" for call in mock.calls)
            )
            repair_content = repair_calls[0]["messages"][1]["content"]
            self.assertIsInstance(repair_content, str)
            self.assertEqual(repair_content.count("A spike train."), 1)
            self.assertIn("max_completion_tokens", repair_calls[0])
            self.assertNotIn("max_tokens", repair_calls[0])
            self.assertEqual(
                repair_calls[0]["thinking"],
                {"type": "disabled"},
            )
            self.assertFalse(repair_calls[0]["stream"])
            self.assertTrue(
                all(value == "Bearer secret" for value in mock.authorization_headers)
            )
            self.assertEqual(report["asr_subtitle"], str(asr))

    def test_prompts_keep_asr_format_and_translate_for_both_content_types(self) -> None:
        self.assertIn("时间分段由系统统一沿用 Qwen3-ASR 文稿", TRANSCRIPT_REPAIR_SYSTEM)
        self.assertIn("文本排版与书写风格也以 Qwen3-ASR 文稿为准", TRANSCRIPT_REPAIR_SYSTEM)
        self.assertIn("不要继承这些格式", TRANSCRIPT_REPAIR_SYSTEM)
        self.assertIn("1 - exp(-r(t_k)*delta)", TRANSLATION_SYSTEM)
        self.assertIn("不要使用 LaTeX", TRANSLATION_SYSTEM)
        self.assertIn("科普、课程或技术内容", TRANSLATION_SYSTEM)
        self.assertIn("电影、剧集或生活对白", TRANSLATION_SYSTEM)
        self.assertIn("视频画面不可用", SUBTITLE_SEGMENTATION_SYSTEM)
        self.assertIn("不要把中文规则机械套用到其他语言", SUBTITLE_SEGMENTATION_SYSTEM)
        self.assertIn("每个输入 id 必须且只能返回一次", SUBTITLE_SEGMENTATION_SYSTEM)
        self.assertIn("最后一个 id 只有一个单词", SUBTITLE_SEGMENTATION_SYSTEM)

    def test_subtitle_only_job_repairs_without_asr(self) -> None:
        with tempfile.TemporaryDirectory() as temp, _MockChatServer() as mock:
            root = Path(temp)
            source = root / "Lecture.en-orig.srt"
            source.write_text(
                "1\n00:00:00,000 --> 00:00:02,000\n"
                "A speak train. This is the next sentence.\n",
                encoding="utf-8",
            )
            (root / "Lecture.asr.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nIgnore stale ASR.\n",
                encoding="utf-8",
            )
            output = root / "output"
            job = VideoJob(
                root / "Lecture.mp4",
                title="Lecture",
                generated_dir=output,
                source_subtitle_path=source,
            )
            workflow = SubtitleRepairWorkflow(
                AppConfig(
                    subtitle_api_base_url=mock.base_url,
                    subtitle_model="mock-text-model",
                ),
                ProcessRunner(),
            )

            report = workflow.process_job(job, repair=True, translate=False)

            self.assertFalse(job.has_video)
            self.assertEqual(
                read_srt(job.corrected_subtitle_path),
                [
                    Cue(
                        1,
                        0,
                        2000,
                        "A spike train. This is the next sentence.",
                    )
                ],
            )
            self.assertEqual(report["asr_subtitle"], "")
            repair_call = next(
                call
                for call in mock.calls
                if "视频文稿校对专家" in call["messages"][0]["content"]
            )
            repair_prompt = json.loads(repair_call["messages"][1]["content"])
            self.assertNotIn("qwen_asr_transcript", repair_prompt)

    def test_speech_subtitle_rewrites_formula_for_tts(self) -> None:
        with tempfile.TemporaryDirectory() as temp, _MockChatServer() as mock:
            root = Path(temp)
            video = root / "Lecture.mp4"
            video.touch()
            output = root / "output"
            output.mkdir()
            job = VideoJob(video, generated_dir=output)
            job.chinese_subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:02,000\n"
                "$\\sum_{i=1}^{10}x_i$\n",
                encoding="utf-8",
            )
            workflow = SpeechSubtitleWorkflow(
                AppConfig(
                    subtitle_api_base_url=mock.base_url,
                    subtitle_model="mock-text-model",
                ),
                ProcessRunner(),
            )

            path = workflow.process_job(job)

            self.assertEqual(
                read_srt(path)[0].text,
                "x i 从 i 为 1 到 10 求和。",
            )


if __name__ == "__main__":
    unittest.main()
