from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def chat_completions_endpoint(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("API Base URL 不能为空")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAICompatibleClient:
    """Small dependency-free client for OpenAI-compatible Chat Completions APIs."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        timeout: int = 300,
    ):
        if not model.strip():
            raise ValueError("模型名称不能为空")
        self.endpoint = chat_completions_endpoint(base_url)
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.timeout = timeout

    @staticmethod
    def _image_part(path: Path) -> dict[str, Any]:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{encoded}",
                "detail": "low",
            },
        }

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        image_paths: list[Path] | None = None,
        max_tokens: int = 4096,
    ) -> ChatResult:
        if image_paths:
            content: str | list[dict[str, Any]] = [
                {"type": "text", "text": user_prompt},
                *(self._image_part(path) for path in image_paths),
            ]
        else:
            content = user_prompt
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"模型 API 返回 HTTP {exc.code}：{detail[:1500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接模型 API：{exc.reason}") from exc
        try:
            raw_content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"模型 API 返回格式异常：{result}") from exc
        if isinstance(raw_content, str):
            text = raw_content
        elif isinstance(raw_content, list):
            text = "\n".join(
                str(item.get("text", ""))
                for item in raw_content
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
            )
        else:
            text = str(raw_content)
        usage = result.get("usage") or {}
        return ChatResult(
            text=text.strip(),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )
