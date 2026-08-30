"""描述符仓库：{描述符名 -> {SMILES: 值}} 的持久化与注册。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..core import db
from ..core.config import CFG

DIR = CFG.abs_path("store/descriptors")
DIR.mkdir(parents=True, exist_ok=True)


def save_values(name: str, values: dict[str, float]) -> Path:
    p = DIR / f"{name}.json"
    p.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
    return p


def load_values(name: str) -> dict[str, float]:
    p = DIR / f"{name}.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {k: (float(v) if v is not None else float("nan")) for k, v in raw.items()}


def register(name: str, hypothesis: str, code: str, spec: dict,
             card_id: str = "", status: str = "computed",
             metrics: dict | None = None) -> None:
    db.ex("INSERT INTO descriptors(name,card_id,hypothesis,code,spec,status,values_path,"
          "metrics,created_at) VALUES(?,?,?,?,?,?,?,?,?) "
          "ON CONFLICT(name) DO UPDATE SET hypothesis=excluded.hypothesis,"
          "code=excluded.code,spec=excluded.spec,status=excluded.status,"
          "metrics=excluded.metrics",
          (name, card_id, hypothesis, code,
           json.dumps(spec, ensure_ascii=False), status, str(DIR / f"{name}.json"),
           json.dumps(metrics or {}, ensure_ascii=False), time.time()))


def set_status(name: str, status: str, metrics: dict | None = None) -> None:
    if metrics is None:
        db.ex("UPDATE descriptors SET status=? WHERE name=?", (status, name))
    else:
        db.ex("UPDATE descriptors SET status=?, metrics=? WHERE name=?",
              (status, json.dumps(metrics, ensure_ascii=False), name))


def listing(status: str | None = None) -> list[dict]:
    if status:
        rows = db.q("SELECT * FROM descriptors WHERE status=? ORDER BY created_at DESC",
                    (status,))
    else:
        rows = db.q("SELECT * FROM descriptors ORDER BY created_at DESC")
    return db.rows_to_dicts(rows)


def active_extra(names: list[str] | None = None) -> dict[str, dict[str, float]]:
    """取出可用于建模的描述符值映射。names=None 表示所有已通过检验的。"""
    if names is None:
        names = [r["name"] for r in listing() if r["status"] in ("passed",)]
    return {n: load_values(n) for n in names if load_values(n)}


def n_tested() -> int:
    """已检验过的描述符总数 —— FDR 多重比较校正的分母。"""
    r = db.q1("SELECT COUNT(*) c FROM descriptors WHERE status IN "
              "('tested','passed','failed')")
    return int(r["c"]) if r else 0
