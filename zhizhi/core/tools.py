"""工具注册表：把普通 Python 函数暴露成 LLM 可调用的 function tool。"""
from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from . import db

# 单次工具返回给模型的最大字符数（超出部分落盘，返回摘要 + 文件指针）
MAX_TOOL_CHARS = 14000


# Long-running tools execute synchronously inside an Agent worker thread.  These
# context variables let a tool publish stage progress and observe the task's
# cooperative stop flag without coupling scientific tools to the jobs module.
_TOOL_PROGRESS: ContextVar[Callable[[str, float | None], None] | None] = \
    ContextVar("zhizhi_tool_progress", default=None)
_TOOL_CANCEL: ContextVar[Callable[[], bool] | None] = \
    ContextVar("zhizhi_tool_cancel", default=None)


@contextmanager
def tool_runtime(progress: Callable[[str, float | None], None] | None = None,
                 should_cancel: Callable[[], bool] | None = None) -> Iterator[None]:
    """Expose the current Agent job's progress/cancel hooks to nested tools."""
    p_token = _TOOL_PROGRESS.set(progress)
    c_token = _TOOL_CANCEL.set(should_cancel)
    try:
        yield
    finally:
        _TOOL_CANCEL.reset(c_token)
        _TOOL_PROGRESS.reset(p_token)


def report_tool_progress(message: str, fraction: float | None = None) -> None:
    """Publish a durable stage update for the currently executing tool."""
    callback = _TOOL_PROGRESS.get()
    if callback:
        value = None if fraction is None else max(0.0, min(1.0, float(fraction)))
        callback(str(message), value)


def tool_cancel_requested() -> bool:
    """Return the owning Agent job's cooperative cancellation state."""
    callback = _TOOL_CANCEL.get()
    return bool(callback and callback())


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    category: str = "general"
    long_running: bool = False

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self.tools)

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        sel = names or self.names()
        return [self.tools[n].schema() for n in sel if n in self.tools]

    def describe(self, names: list[str] | None = None) -> str:
        sel = names or self.names()
        return "\n".join(f"- {n}: {self.tools[n].description.splitlines()[0]}"
                         for n in sel if n in self.tools)

    def call(self, name: str, args: dict, agent: str = "?") -> Any:
        if name not in self.tools:
            return {"error": f"未知工具 {name}；可用：{', '.join(self.names())}"}
        t0 = time.time()
        try:
            out = self.tools[name].fn(**(args or {}))
            db.audit(agent, name, args, True, (time.time() - t0) * 1000)
            return out
        except TypeError as e:
            db.audit(agent, name, args, False, (time.time() - t0) * 1000)
            return {"error": f"参数错误: {e}", "expected": self.tools[name].parameters}
        except Exception as e:  # noqa: BLE001
            db.audit(agent, name, args, False, (time.time() - t0) * 1000)
            return {"error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=4)[-1500:]}


REGISTRY = Registry()


def tool(name: str, description: str, parameters: dict | None = None,
         category: str = "general", long_running: bool = False):
    """装饰器：注册一个工具。parameters 用 JSON Schema 描述。"""
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY.add(Tool(name=name, description=description,
                          parameters=parameters or {"type": "object", "properties": {}},
                          fn=fn, category=category, long_running=long_running))
        return fn
    return deco


def obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


def P(typ: str, desc: str, **kw) -> dict:
    d = {"type": typ, "description": desc}
    d.update(kw)
    return d


def to_model_text(result: Any, max_chars: int | None = None) -> str:
    """把工具返回值压成模型能吃的文本；过长则落盘并返回指针。"""
    if isinstance(result, str):
        text = result
    else:
        text = json.dumps(result, ensure_ascii=False, default=str, indent=1)
    limit = max(500, int(max_chars or MAX_TOOL_CHARS))
    if len(text) <= limit:
        return text
    from .config import CFG
    path = CFG.logs_dir / f"tool_out_{int(time.time()*1000)}.json"
    path.write_text(text, encoding="utf-8")
    return (text[:limit]
            + f"\n\n…[输出被截断，共 {len(text)} 字符。完整结果已存至 {path}]")
