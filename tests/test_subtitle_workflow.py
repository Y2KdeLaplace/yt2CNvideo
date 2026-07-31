import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from videodub.config import AppConfig
from videodub.media import VideoJob
from videodub.openai_compatible import chat_completions_endpoint
from videodub.runner import ProcessRunner
from videodub.subtitle_workflow import SpeechSubtitleWorkflow, SubtitleRepairWorkflow
from videodub.subtitles import find_source_subtitle, read_srt


class _MockChatServer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                owner.calls.append(payload)
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
                elif "字幕校对专家" in system:
                    content = json.dumps(
                        [
                            {
                                "id": 1,
                                "corrected_text": "A spike train.",
                            },
                            {
                                "id": 2,
                                "corrected_text": "This is the next sentence.",
                            },
                        ]
                    )
                elif "中文字幕译者" in system:
                    content = json.dumps(
                        [
                            {"id": 1, "zh": "一个脉冲序列。"},
                            {"id": 2, "zh": "这是下一句话。"},
                        ]
                    )
                elif "中文讲解稿改写专家" in system:
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
            "http://mac-mini.local:8000/v1/chat/completions",
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
                "1\n00:00:00,000 --> 00:00:01,000\nA speak train.\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\nThis is the next sentence.\n",
                encoding="utf-8",
            )
            asr = root / "Lecture [abc123].asr.srt"
            asr.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nA spike train.\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\nThis is the next sentence.\n",
                encoding="utf-8",
            )
            config = AppConfig(
                subtitle_api_base_url=mock.base_url,
                subtitle_model="mock-text-model",
                subtitle_detection_batch_size=10,
                subtitle_translation_batch_size=10,
            )
            workflow = SubtitleRepairWorkflow(config, ProcessRunner())
            job = VideoJob(video, title="Lecture", generated_dir=root / "output")
            report = workflow.process_job(job)

            corrected = read_srt(job.corrected_subtitle_path)
            chinese = read_srt(job.chinese_subtitle_path)
            self.assertEqual(corrected[0].text, "A spike train.")
            self.assertEqual(chinese[0].text, "一个脉冲序列。")
            self.assertEqual(report["domain"]["domain"], "computational neuroscience")
            repair_calls = [
                call
                for call in mock.calls
                if "字幕校对专家" in call["messages"][0]["content"]
            ]
            self.assertEqual(len(repair_calls), 1)
            self.assertTrue(
                all(call["model"] == "mock-text-model" for call in mock.calls)
            )
            repair_content = repair_calls[0]["messages"][1]["content"]
            self.assertIsInstance(repair_content, str)
            self.assertEqual(report["asr_subtitle"], str(asr))

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
