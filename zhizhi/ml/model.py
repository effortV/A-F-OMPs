"""模型内核：XGBoost 训练、分组交叉验证、OOF 残差、不确定度、适用域。

关键设计：发现层只吃 **分组 OOF 残差**。
随机切分会把同一 (化合物,膜) 的重复记录同时放进 train/test（本数据集单对最多 30 条重复），
测试指标被严重高估，残差里剩下的是"记忆残留"而不是"机理缺失"。
legacy 模式仅用于与既有工作对齐，不参与任何发现。
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split

from ..core import db
from ..core.config import CFG
from ..dataio import loader

PARAMS = dict(CFG.get("model.params"))


def make_model(**override) -> xgb.XGBRegressor:
    p = dict(PARAMS)
    p.update(override)
    return xgb.XGBRegressor(**p, n_jobs=-1, tree_method="hist")


def _metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {"r2": round(float(r2_score(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 3),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
            "n": int(len(y_true))}


@dataclass
class Bundle:
    """一次建模的完整产物。"""
    key: str
    X: pd.DataFrame
    y: pd.Series
    groups: dict
    meta: pd.DataFrame
    model: Any
    oof: np.ndarray
    oof_group: str
    metrics: dict

    @property
    def residual(self) -> np.ndarray:
        return self.y.to_numpy() - self.oof

    def as_frame(self) -> pd.DataFrame:
        f = self.meta.copy()
        f["y_true"] = self.y.to_numpy()
        f["y_oof"] = self.oof
        f["residual"] = self.residual
        f["abs_residual"] = np.abs(f["residual"])
        return f


_CACHE: dict[str, Bundle] = {}


def _key(extra_names: list[str], drop: list[str] | None, group: str,
         substructure: bool) -> str:
    raw = json.dumps([sorted(extra_names), sorted(drop or []), group, substructure])
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def get_bundle(extra: dict[str, dict[str, float]] | None = None,
               drop: list[str] | None = None,
               group: str | None = None,
               with_substructure: bool = True,
               refit: bool = False,
               cache: bool = True) -> Bundle:
    """构造 + 训练 + 分组 OOF，带内存与磁盘缓存。

    cache=False 用于 y-scrambling 等一次性重训：同一个 key 会被反复覆盖，
    落盘毫无意义还会产生几十 MB 的写入churn。
    """
    group = group or CFG.get("model.cv.default_group", "compound")
    k = _key(sorted((extra or {}).keys()), drop, group, with_substructure)
    if not refit and cache and k in _CACHE:
        return _CACHE[k]

    disk = CFG.cache_dir / f"bundle_{k}.pkl"
    if not refit and cache and disk.exists():
        try:
            b = pickle.loads(disk.read_bytes())
            _CACHE[k] = b
            return b
        except Exception:  # noqa: BLE001
            pass

    built = loader.build_matrix(extra=extra, drop=drop, with_substructure=with_substructure)
    X, y = built["X"], built["y"]
    g = built["groups"][group] if group in built["groups"] else np.arange(len(y))

    n_splits = int(CFG.get("model.cv.n_splits", 5))
    n_groups = len(np.unique(g))
    splitter = (GroupKFold(n_splits=min(n_splits, n_groups))
                if n_groups >= 2 else KFold(n_splits=n_splits, shuffle=True, random_state=108))

    oof = np.full(len(y), np.nan)
    for tr, te in splitter.split(X, y, groups=g):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])

    full = make_model()
    full.fit(X, y)
    in_sample = full.predict(X)

    metrics = {
        "group_cv": {"group_by": group, "n_groups": int(n_groups), **_metrics(y, oof)},
        "in_sample": _metrics(y, in_sample),
        "n_features": int(X.shape[1]),
        "features": list(X.columns),
    }
    b = Bundle(key=k, X=X, y=y, groups=built["groups"], meta=built["meta"],
               model=full, oof=oof, oof_group=group, metrics=metrics)
    if not cache:
        return b
    _CACHE[k] = b
    try:
        disk.write_bytes(pickle.dumps(b))
    except Exception:  # noqa: BLE001
        pass
    db.ex("INSERT INTO model_runs(name,params,metrics,note,created_at) VALUES(?,?,?,?,?)",
          (f"bundle:{k}", json.dumps(PARAMS),
           json.dumps(metrics, ensure_ascii=False),
           f"extra={sorted((extra or {}).keys())} drop={drop} group={group}", time.time()))
    return b


def legacy_report() -> dict:
    """复现既有工作的随机切分口径，并与分组 CV 对照，量化泄漏幅度。"""
    built = loader.build_matrix()
    X, y = built["X"], built["y"]
    cfg = CFG.get("model.legacy_split")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"])
    m = make_model()
    m.fit(Xtr, ytr)
    rnd = {"train": _metrics(ytr, m.predict(Xtr)), "test": _metrics(yte, m.predict(Xte))}

    out: dict[str, Any] = {"random_split_legacy": rnd, "grouped": {}}
    for gname in ("compound", "membrane", "reference"):
        b = get_bundle(group=gname)
        out["grouped"][gname] = b.metrics["group_cv"]
    out["leakage_gap_r2"] = round(
        rnd["test"]["r2"] - out["grouped"]["compound"]["r2"], 4)
    out["verdict"] = (
        "随机切分测试 R²={:.4f}，按化合物分组 CV R²={:.4f}，差值 {:.4f} 即泄漏幅度。"
        "该差值本身就是第一条发现：现有特征体系对**没见过的新分子**的外推能力，"
        "远低于随机切分所显示的水平。".format(
            rnd["test"]["r2"], out["grouped"]["compound"]["r2"], out["leakage_gap_r2"]))
    return out


# ---- 不确定度与适用域 --------------------------------------------------
def ensemble_std(X_query: pd.DataFrame, b: Bundle, n_models: int = 5) -> np.ndarray:
    """多种子集成方差作为预测不确定度。"""
    ck = CFG.cache_dir / f"ens_{b.key}_{n_models}.pkl"
    models = None
    if ck.exists():
        try:
            models = pickle.loads(ck.read_bytes())
        except Exception:  # noqa: BLE001
            models = None
    if models is None:
        models = []
        for s in range(n_models):
            m = make_model(random_state=100 + s, subsample=0.8, colsample_bytree=0.6)
            m.fit(b.X, b.y)
            models.append(m)
        try:
            ck.write_bytes(pickle.dumps(models))
        except Exception:  # noqa: BLE001
            pass
    preds = np.vstack([m.predict(X_query) for m in models])
    return preds.std(axis=0)


def _standardize(b: Bundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = b.X.to_numpy(float)
    med = np.nanmedian(A, axis=0)
    A = np.where(np.isnan(A), med, A)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    return (A - mu) / sd, mu, sd


def applicability(X_query: pd.DataFrame, b: Bundle, k: int = 5) -> np.ndarray:
    """到训练流形的 kNN 平均距离（标准化空间），越大越外推。"""
    Z, mu, sd = _standardize(b)
    Q = X_query.to_numpy(float)
    med = np.nanmedian(b.X.to_numpy(float), axis=0)
    Q = np.where(np.isnan(Q), med, Q)
    Q = (Q - mu) / sd
    out = np.zeros(len(Q))
    for i, row in enumerate(Q):
        d = np.sqrt(((Z - row) ** 2).sum(axis=1))
        out[i] = np.sort(d)[:k].mean()
    return out


def domain_threshold(b: Bundle, k: int = 5, pct: float = 95.0) -> float:
    return float(np.percentile(applicability(b.X, b, k=k), pct))
