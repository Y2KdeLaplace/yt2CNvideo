import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

from videodub.openai_compatible import (
    OpenAICompatibleClient,
    chat_completions_endpoint,
    openai_base_url,
)


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


if __name__ == "__main__":
    unittest.main()
