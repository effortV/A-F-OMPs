"""换模型体检：把一个候选模型跑一遍系统真正用到的所有调用形态。

系统对模型有 6 项硬要求，缺任何一项都会有功能悄悄失效：
  1. 基础对话        —— 所有地方
  2. 工具调用        —— Agent 循环的命根子，不支持则四个智能体全废
  3. 流式输出        —— 界面首字延迟
  4. enable_thinking —— 机械任务提速 3-4 倍（不支持不致命，但会慢很多）
  5. JSON 模式       —— 结构化抽取、矛盾研判、跨界扫描
  6. 长文生成        —— L2/L3 方案（12000 max_tokens）
"""
from __future__ import annotations

import time
from typing import Any

from .config import CFG
from .llm import LLM

TOOLS = [{"type": "function", "function": {
    "name": "get_r2", "description": "返回模型的 R2 指标",
    "parameters": {"type": "object",
                   "properties": {"split": {"type": "string"}},
                   "required": ["split"]}}}]

Q = [{"role": "user", "content": "列出3种纳滤膜型号及其MWCO，用JSON输出。"}]


def _try(fn) -> dict:
    t0 = time.time()
    try:
        info = fn()
        info["ok"] = True
        info["seconds"] = round(time.time() - t0, 1)
        return info
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "seconds": round(time.time() - t0, 1),
                "error": str(e)[:220]}


def check_model(model: str, timeout: int = 90) -> dict:
    """对单个模型跑全套兼容性检查。不改任何配置，纯只读探测。"""
    from openai import OpenAI
    c = OpenAI(api_key=CFG.api_key, base_url=CFG.base_url,
               timeout=timeout, max_retries=0)

    def basic():
        r = c.chat.completions.create(model=model, messages=Q)
        msg = r.choices[0].message
        rc = getattr(msg, "reasoning_content", None) or ""
        return {"out_tokens": r.usage.completion_tokens,
                "reasoning_chars": len(rc),
                "is_reasoning_model": len(rc) > 0}

    def no_think():
        r = c.chat.completions.create(model=model, messages=Q,
                                      extra_body={"enable_thinking": False})
        rc = getattr(r.choices[0].message, "reasoning_content", None) or ""
        return {"out_tokens": r.usage.completion_tokens, "reasoning_chars": len(rc),
                "actually_disabled": len(rc) == 0}

    def json_mode():
        r = c.chat.completions.create(model=model, messages=Q,
                                      response_format={"type": "json_object"})
        from .llm import json_from_text
        json_from_text(r.choices[0].message.content or "")
        return {"parsed": True}

    def tool_call():
        r = c.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "用工具查一下测试集R2"}],
            tools=TOOLS, tool_choice="auto")
        tc = r.choices[0].message.tool_calls
        return {"n_tool_calls": len(tc) if tc else 0,
                "called": bool(tc), "name": tc[0].function.name if tc else None}

    def streaming():
        t0, first, n = time.time(), None, 0
        s = c.chat.completions.create(model=model, messages=Q,
                                      stream=True,
                                      stream_options={"include_usage": True})
        for ch in s:
            if ch.choices and ch.choices[0].delta.content:
                n += 1
                if first is None:
                    first = round(time.time() - t0, 2)
        return {"first_token_s": first, "n_chunks": n}

    def long_form():
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content":
                       "写一份纳滤膜实验的标准操作流程，分 6 节，尽量详细。"}],
            extra_body={"enable_thinking": False})
        txt = r.choices[0].message.content or ""
        return {"out_tokens": r.usage.completion_tokens, "chars": len(txt),
                "truncated": r.choices[0].finish_reason == "length"}

    res: dict[str, Any] = {"model": model, "base_url": CFG.base_url}
    res["checks"] = {
        "1_基础对话": _try(basic),
        "2_工具调用": _try(tool_call),
        "3_流式输出": _try(streaming),
        "4_关思维链": _try(no_think),
        "5_JSON模式": _try(json_mode),
        "6_长文生成": _try(long_form),
    }
    ch = res["checks"]
    blocking = [k for k in ("1_基础对话", "2_工具调用") if not ch[k]["ok"]]
    degraded = [k for k in ("3_流式输出", "4_关思维链", "5_JSON模式", "6_长文生成")
                if not ch[k]["ok"]]
    res["verdict"] = ("不可用" if blocking else "可用但有降级" if degraded else "完全可用")
    res["blocking_failures"] = blocking
    res["degraded"] = degraded

    advice = []
    if blocking:
        advice.append(f"致命项失败 {blocking} —— 这个模型不能用作主模型。"
                      "429/超时可能只是限流，隔几分钟重试一次再下结论。")
    if "4_关思维链" in degraded:
        advice.append("不支持 enable_thinking：把 config.yaml 的 llm.thinking_default "
                      "改成 false，否则机械任务会慢 3-4 倍。")
    if "2_工具调用" in ch and ch["2_工具调用"].get("ok") and not ch["2_工具调用"].get("called"):
        advice.append("接受了 tools 参数但没有真的调用工具，Agent 可能会退化成纯聊天。"
                      "换个提问再试，若仍不调用则不适合做主模型。")
    if ch["1_基础对话"].get("is_reasoning_model") is False:
        advice.append("非推理模型：thinking_default 建议设 false，thinking_budget 可删。")
    if ch["6_长文生成"].get("truncated"):
        advice.append("长文被 max_tokens 截断，L2/L3 方案生成可能不完整，考虑调大。")
    res["advice"] = advice or ["无需额外调整，直接改 .env 的 LLM_MODEL 即可。"]
    return res


def check_current() -> dict:
    return check_model(LLM().model)
