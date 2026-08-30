"""SiliconFlow LLM 客户端：对话 / 工具调用 / 向量化，带重试、降级与用量记账。"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Iterable

import numpy as np
from openai import OpenAI

from . import db
from .config import CFG


class LLMError(RuntimeError):
    pass


def approx_tokens(text: str) -> int:
    """粗略 token 估计：中文约 1 字 1 token，英文约 4 字符 1 token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return int(cjk + (len(text) - cjk) / 3.6)


def json_from_text(text: str) -> Any:
    """从模型输出里稳健地抠出 JSON（容忍 ```json 围栏、前后废话）。"""
    if text is None:
        raise LLMError("空输出")
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            frag = t[i:j + 1]
            try:
                return json.loads(frag)
            except Exception:
                try:
                    return json.loads(re.sub(r",\s*([}\]])", r"\1", frag))
                except Exception:
                    continue
    raise LLMError(f"无法解析为 JSON: {text[:400]}")


class LLM:
    _client: OpenAI | None = None

    def __init__(self, agent: str = "system", model: str | None = None,
                 fallbacks: Iterable[str] | None = None,
                 usage_kind: str = "chat"):
        self.agent = agent
        self.model = model or CFG.llm_model
        self.embed_model = CFG.embed_model
        self.fallbacks: list[str] = (list(fallbacks) if fallbacks is not None
                                    else list(CFG.fallback_models))
        self.usage_kind = usage_kind

    # ---- 底层 client -------------------------------------------------
    @classmethod
    def client(cls) -> OpenAI:
        if cls._client is None:
            if not CFG.api_key:
                raise LLMError("SILICONFLOW_API_KEY 未设置（检查 .env）")
            cls._client = OpenAI(api_key=CFG.api_key, base_url=CFG.base_url,
                                 timeout=float(CFG.get("llm.timeout", 180)),
                                 max_retries=0)
        return cls._client

    # ---- 对话 ---------------------------------------------------------
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             force_json: bool = False, model: str | None = None,
             thinking: bool | None = None,
             request_timeout: float | None = None,
             attempts: int | None = None) -> Any:
        """返回 openai ChatCompletionMessage。失败自动重试并降级到备用模型。

        thinking=False 会关闭思维链（SiliconFlow 的 enable_thinking）。
        DeepSeek-V4-Pro 是推理模型，一句话问题也会烧近千个思考 token；
        机械任务（结构化抽取、相关性打分、检索式扩展、历史压缩）关掉可快 3-4 倍。
        """
        if thinking is None:
            thinking = bool(CFG.get("llm.thinking_default", True))
        kwargs: dict[str, Any] = {
            "temperature": CFG.get("llm.temperature", 0.3) if temperature is None else temperature,
        }
        # Scientific JSON/documents must be allowed to finish naturally.  No
        # output ceiling is added unless an external caller explicitly requests one.
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        extra: dict[str, Any] = {}
        if not thinking:
            extra["enable_thinking"] = False
        elif CFG.get("llm.thinking_budget"):
            extra["thinking_budget"] = int(CFG.get("llm.thinking_budget"))
        if extra:
            kwargs["extra_body"] = extra
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if force_json and not tools:
            kwargs["response_format"] = {"type": "json_object"}
        if request_timeout is not None:
            # The SDK accepts a per-request timeout.  Long scientific manuals
            # keep the global 600 s budget, while bounded discovery helpers can
            # opt into a much smaller ceiling without changing L2/L3 behavior.
            kwargs["timeout"] = max(5.0, float(request_timeout))

        chain = [model or self.model] + [m for m in self.fallbacks if m != (model or self.model)]
        max_retries = max(1, int(attempts if attempts is not None
                                 else CFG.get("llm.max_retries", 4)))
        last_err: Exception | None = None

        for mdl in chain:
            for attempt in range(max_retries):
                try:
                    resp = self.client().chat.completions.create(
                        model=mdl, messages=messages, **kwargs)
                    u = getattr(resp, "usage", None)
                    db.log_usage(self.agent, mdl, self.usage_kind,
                                 getattr(u, "prompt_tokens", 0) or 0,
                                 getattr(u, "completion_tokens", 0) or 0)
                    return resp.choices[0].message
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    msg = str(e)
                    # 明确的模型不可用 -> 直接降级，不再重试
                    if any(s in msg for s in ("model_not_found", "Model does not exist",
                                              "does not exist", "20012", "404")):
                        break
                    time.sleep(min(2 ** attempt, 20))
        raise LLMError(f"LLM 调用失败（已试模型 {chain}）: {last_err}")

    def chat_stream(self, messages: list[dict], tools: list[dict] | None = None,
                    temperature: float | None = None, max_tokens: int | None = None,
                    thinking: bool | None = None):
        """流式对话。逐块 yield ('delta', 文本) / ('reasoning', 思考文本)，
        最后 yield ('done', 组装好的 message-like 对象)。

        首字延迟从整段生成时间（30-60s）降到 2-3s，是界面体感的关键。
        工具调用增量按 index 累积，与非流式返回结构保持一致。
        """
        if thinking is None:
            thinking = bool(CFG.get("llm.thinking_default", True))
        kwargs: dict[str, Any] = {
            "temperature": CFG.get("llm.temperature", 0.3) if temperature is None else temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        extra: dict[str, Any] = {}
        if not thinking:
            extra["enable_thinking"] = False
        elif CFG.get("llm.thinking_budget"):
            extra["thinking_budget"] = int(CFG.get("llm.thinking_budget"))
        if extra:
            kwargs["extra_body"] = extra
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        content_parts: list[str] = []
        tool_acc: dict[int, dict] = {}
        pt = ct = 0
        stream = self.client().chat.completions.create(
            model=self.model, messages=messages, **kwargs)
        for chunk in stream:
            u = getattr(chunk, "usage", None)
            if u:
                pt = getattr(u, "prompt_tokens", 0) or pt
                ct = getattr(u, "completion_tokens", 0) or ct
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                yield ("reasoning", rc)
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                yield ("delta", delta.content)
            for tc in (getattr(delta, "tool_calls", None) or []):
                slot = tool_acc.setdefault(
                    tc.index, {"id": "", "type": "function",
                               "function": {"name": "", "arguments": ""}})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["function"]["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["function"]["arguments"] += tc.function.arguments

        db.log_usage(self.agent, self.model, self.usage_kind, pt, ct)
        yield ("done", {"content": "".join(content_parts),
                        "tool_calls": [tool_acc[i] for i in sorted(tool_acc)]})

    def ask(self, system: str, user: str, force_json: bool = False,
            temperature: float | None = None, max_tokens: int | None = None,
            thinking: bool | None = None,
            request_timeout: float | None = None,
            attempts: int | None = None) -> str:
        m = self.chat([{"role": "system", "content": system},
                       {"role": "user", "content": user}],
                      force_json=force_json, temperature=temperature,
                      max_tokens=max_tokens, thinking=thinking,
                      request_timeout=request_timeout, attempts=attempts)
        return m.content or ""

    def ask_json(self, system: str, user: str, temperature: float = 0.2,
                 max_tokens: int | None = None, thinking: bool | None = None,
                 request_timeout: float | None = None,
                 attempts: int | None = None) -> Any:
        sys_j = system + "\n\n只输出合法 JSON，不要任何解释文字、不要 markdown 围栏。"
        txt = self.ask(sys_j, user, force_json=True, temperature=temperature,
                       max_tokens=max_tokens, thinking=thinking,
                       request_timeout=request_timeout, attempts=attempts)
        return json_from_text(txt)

    # ---- 向量 ---------------------------------------------------------
    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts = [t if t and t.strip() else " " for t in texts]
        if not texts:
            return np.zeros((0, int(CFG.get("llm.embed_dim", 1024))), dtype=np.float32)
        bs = int(CFG.get("literature.embed_batch", 16))
        out: list[list[float]] = []
        for i in range(0, len(texts), bs):
            batch = [t[:8000] for t in texts[i:i + bs]]
            for attempt in range(4):
                try:
                    r = self.client().embeddings.create(model=self.embed_model, input=batch)
                    out.extend([d.embedding for d in r.data])
                    u = getattr(r, "usage", None)
                    db.log_usage(self.agent, self.embed_model, "embed",
                                 getattr(u, "prompt_tokens", 0) or 0, 0)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        raise LLMError(f"embedding 失败: {e}") from e
                    time.sleep(min(2 ** attempt, 15))
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def health_check() -> dict:
    """连通性自检：返回可用模型、embedding 维度。"""
    llm = LLM("doctor")
    res: dict[str, Any] = {"base_url": CFG.base_url, "key_set": bool(CFG.api_key)}
    try:
        msg = llm.chat([{"role": "user", "content": "回复两个字：正常"}])
        res["chat_model"] = llm.model
        res["chat_ok"] = True
        res["chat_reply"] = (msg.content or "").strip()[:50]
    except Exception as e:  # noqa: BLE001
        res["chat_ok"] = False
        res["chat_error"] = str(e)[:400]
    try:
        v = llm.embed(["测试 test"])
        res["embed_ok"] = True
        res["embed_dim"] = int(v.shape[1])
    except Exception as e:  # noqa: BLE001
        res["embed_ok"] = False
        res["embed_error"] = str(e)[:400]
    return res
