from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


def openai_base_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("API Base URL 不能为空")
    if base.endswith("/chat/completions"):
        return base.removesuffix("/chat/completions")
    return base


def chat_completions_endpoint(base_url: str) -> str:
    return openai_base_url(base_url) + "/chat/completions"


def _preferred_token_limit_field(base_url: str) -> str:
    if "api.deepseek.com" in base_url.casefold():
        return "max_tokens"
    return "max_completion_tokens"


def _is_forced_thinking_model(model: str) -> bool:
    normalized = model.strip().casefold()
    return normalized.startswith("kimi-k2-thinking")


def _unsupported_parameter(exc: APIStatusError, parameter: str) -> bool:
    if exc.status_code not in {400, 422}:
        return False
    detail = exc.response.text.casefold()
    return parameter.casefold() in detail and any(
        marker in detail
        for marker in (
            "unknown",
            "unsupported",
            "unrecognized",
            "not permitted",
            "extra",
            "invalid",
        )
    )


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAICompatibleClient:
    """OpenAI SDK wrapper for compatible Chat Completions APIs."""

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
        self.base_url = openai_base_url(base_url)
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.timeout = timeout
        self._token_limit_field = _preferred_token_limit_field(self.base_url)
        self._thinking_control_supported = True
        self._client = OpenAI(
            api_key=self.api_key or "local-no-key-required",
            base_url=self.base_url,
            timeout=self.timeout,
        )

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
        if _is_forced_thinking_model(self.model):
            raise RuntimeError(
                f"模型 {self.model} 强制使用思考模式，请换用可关闭思考的 Kimi 模型，"
                "例如 kimi-k2.6 或 kimi-k2.5。"
            )

        request_args: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "stream": False,
        }

        for _attempt in range(3):
            current_args = dict(request_args)
            current_args[self._token_limit_field] = max_tokens
            if self._thinking_control_supported:
                current_args["extra_body"] = {
                    "thinking": {"type": "disabled"},
                }
            try:
                result = self._client.chat.completions.create(**current_args)
                break
            except APIStatusError as exc:
                if self._thinking_control_supported and _unsupported_parameter(
                    exc, "thinking"
                ):
                    self._thinking_control_supported = False
                    continue
                if self._token_limit_field == "max_completion_tokens" and (
                    _unsupported_parameter(exc, "max_completion_tokens")
                ):
                    self._token_limit_field = "max_tokens"
                    continue
                raise RuntimeError(
                    f"模型 API 返回 HTTP {exc.status_code}："
                    f"{exc.response.text[:1500]}"
                ) from exc
            except APITimeoutError as exc:
                raise RuntimeError("连接模型 API 超时") from exc
            except APIConnectionError as exc:
                raise RuntimeError(f"无法连接模型 API：{exc}") from exc
        else:
            raise RuntimeError("模型 API 不支持所需的兼容参数")

        try:
            raw_content = result.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(f"模型 API 返回格式异常：{result}") from exc
        if isinstance(raw_content, str):
            text = raw_content
        elif isinstance(raw_content, list):
            text = "\n".join(
                str(
                    item.get("text", "")
                    if isinstance(item, dict)
                    else getattr(item, "text", "")
                )
                for item in raw_content
                if (
                    isinstance(item, dict)
                    and item.get("type") in {"text", "output_text"}
                )
                or getattr(item, "type", "") in {"text", "output_text"}
            )
        else:
            text = str(raw_content)
        usage = result.usage
        return ChatResult(
            text=text.strip(),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
