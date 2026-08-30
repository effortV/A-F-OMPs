from __future__ import annotations

import json
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings, get_settings

JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


class LLMNotConfigured(RuntimeError):
    pass


class DeepSeekClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_name: str | None = None,
        enable_thinking: bool | None = None,
    ):
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.llm_model
        self.enable_thinking = enable_thinking
        self._client: AsyncOpenAI | None = None
        if self.settings.siliconflow_api_key:
            self._client = AsyncOpenAI(
                api_key=self.settings.siliconflow_api_key.get_secret_value(),
                base_url=self.settings.siliconflow_base_url,
                timeout=self.settings.llm_timeout_seconds,
            )

    @property
    def configured(self) -> bool:
        return self._client is not None

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int = 4096,
        enable_thinking: bool | None = None,
    ) -> str:
        if not self._client:
            raise LLMNotConfigured("未配置 SILICONFLOW_API_KEY")
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        if thinking is not None and "DeepSeek-V4" in self.model_name:
            request["extra_body"] = {"enable_thinking": thinking}
        response = await self._client.chat.completions.create(**request)  # type: ignore[arg-type]
        return response.choices[0].message.content or ""

    async def json_chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any] | list[Any]:
        text = await self.chat(
            [
                {"role": "system", "content": system + "\n只输出合法 JSON，不要使用 Markdown 代码块。"},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        cleaned = JSON_FENCE.sub("", text.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start_candidates = [pos for pos in (cleaned.find("{"), cleaned.find("[")) if pos >= 0]
            if not start_candidates:
                raise
            start = min(start_candidates)
            end = max(cleaned.rfind("}"), cleaned.rfind("]"))
            return json.loads(cleaned[start : end + 1])

