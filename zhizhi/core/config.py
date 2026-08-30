"""全局配置：从 config.yaml + .env 装载，暴露单例 CFG。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]          # 项目根 = .../Claude
PKG = Path(__file__).resolve().parents[1]           # .../Claude/zhizhi


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析（不引 python-dotenv）。已存在的环境变量优先。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.split(" #")[0].strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv(ROOT / ".env")

# Windows 11 已移除 wmic，joblib/loky 探测物理核心数时会打出一大段无害但刺眼的
# traceback。预先把它的缓存填上，probe 就不会执行。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
try:  # pragma: no cover - 纯环境噪音抑制
    from joblib.externals.loky.backend import context as _loky_ctx
    if getattr(_loky_ctx, "physical_cores_cache", None) is None:
        _loky_ctx.physical_cores_cache = max(1, (os.cpu_count() or 4) // 2)
except Exception:  # noqa: BLE001
    pass


class Config:
    def __init__(self, yaml_path: Path | None = None):
        self.path = yaml_path or (PKG / "config.yaml")
        self._d: dict[str, Any] = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.root = ROOT
        for key in ("store", "figures", "cards", "logs", "cache", "new_pdf_dir"):
            self.abs_path(self._d["paths"][key]).mkdir(parents=True, exist_ok=True)

    # ---- 字典式访问 -------------------------------------------------
    def __getitem__(self, k: str) -> Any:
        return self._d[k]

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self._d
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def abs_path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    # ---- 常用快捷方式 -----------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.abs_path(self._d["paths"]["db"])

    @property
    def dataset_path(self) -> Path:
        return self.abs_path(self._d["paths"]["dataset"])

    @property
    def reference_dir(self) -> Path:
        return self.abs_path(self._d["paths"]["reference_dir"])

    @property
    def new_pdf_dir(self) -> Path:
        return self.abs_path(self._d["paths"]["new_pdf_dir"])

    @property
    def cache_dir(self) -> Path:
        return self.abs_path(self._d["paths"]["cache"])

    @property
    def figures_dir(self) -> Path:
        return self.abs_path(self._d["paths"]["figures"])

    @property
    def cards_dir(self) -> Path:
        return self.abs_path(self._d["paths"]["cards"])

    @property
    def logs_dir(self) -> Path:
        return self.abs_path(self._d["paths"]["logs"])

    # ---- 密钥（只从环境变量取，绝不写进 yaml）------------------------
    @staticmethod
    def env(key: str, default: str = "") -> str:
        return os.environ.get(key, default) or default

    @property
    def api_key(self) -> str:
        return self.env("SILICONFLOW_API_KEY")

    @property
    def base_url(self) -> str:
        return self.env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

    @property
    def llm_model(self) -> str:
        return self.env("LLM_MODEL", self.get("llm.model"))

    @property
    def literature_preprocess_model(self) -> str:
        """文献低风险预处理专用模型；不影响关键语义判断与聊天主模型。"""
        return self.env(
            "LIT_PREPROCESS_MODEL",
            self.get("llm.literature_preprocess_model", "Pro/deepseek-ai/DeepSeek-V3.2"),
        )

    @property
    def embed_model(self) -> str:
        return self.env("EMBED_MODEL", self.get("llm.embed_model"))

    @property
    def fallback_models(self) -> list[str]:
        """备选/降级模型链。

        默认 = 只有主模型本身，即「只重试，绝不静默降级到别的模型」。
        想启用降级就在 .env 里写 LLM_FALLBACK_MODELS=模型A,模型B（逗号分隔）。
        这样换模型真的只改 .env 一处，不会漏掉 yaml 里写死的旧模型。
        """
        raw = self.env("LLM_FALLBACK_MODELS", "").strip()
        if raw:
            return [m.strip() for m in raw.split(",") if m.strip()]
        return [self.llm_model]


CFG = Config()
