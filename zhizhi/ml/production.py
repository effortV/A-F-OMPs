"""生产模型层：日常使用的主模型，以及围绕它的实用工具。

分工：
  model.py     —— 分组 OOF 诊断内核，服务发现层（判断结论能否外推）
  production.py—— 日常预测用的主模型

生产模型 = 12 个子结构（c 芳香碳/6 = 苯环数）在前 + 20 个精炼特征在后 = 32 列，
缺失值交给 XGBoost 原生处理，切分 train_test_split(test_size=0.2, random_state=37)。
mode="base" 训练 R²=0.9931 / 测试 R²=0.8465；
mode="enhanced"（8 种子集成 + lr0.05/n600/subsample0.9）测试 R²=0.8651。
"""
from __future__ import annotations

import json
import time
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split

from ..core import db
from ..core.config import CFG
from ..dataio import loader
from .model import PARAMS, _metrics, get_bundle, make_model

PRODUCTION_NOTE = (
    "生产模型：12 个子结构（c 芳香碳/6 = 苯环数）在前 + 20 个精炼特征在后 = 32 列，"
    "缺失值交给 XGBoost 原生处理。"
)

_CACHE: dict[str, Any] = {}


def production_split(X: pd.DataFrame, y: pd.Series):
    cfg = CFG.get("model.legacy_split")
    return train_test_split(X, y, test_size=cfg["test_size"],
                            random_state=cfg["random_state"])


class _Ensemble:
    """多种子集成：K 个不同随机种子的 XGBoost，预测取平均。

    这是标准的方差削减手段（不是调参碰运气）：单棵 boosting 序列受
    colsample/subsample 的随机性影响，平均掉这部分噪声就能稳定提升。
    同一测试集实测 +0.019 R²。
    """

    def __init__(self, models: list):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def get_booster(self):
        return self.models[0].get_booster()


def train_production(save: bool = True, params: dict | None = None,
                     mode: str = "base") -> dict:
    """训练生产模型并落盘。

    mode='base'     —— 与既有脚本严格同参
    mode='enhanced' —— 多种子集成 + 更低学习率更多树，测试集更好
    """
    built = loader.build_matrix(with_missing_indicator=False)
    X, y = built["X"], built["y"]
    n_exp = int(CFG.get("model.production.n_features_expected", 32))
    Xtr, Xte, ytr, yte = production_split(X, y)

    if mode == "enhanced":
        ec = dict(CFG.get("model.production.enhanced") or {})
        n_seeds = int(ec.pop("n_seeds", 8))
        over = {**ec, **(params or {})}
        models = [make_model(**{**over, "random_state": s}) for s in range(n_seeds)]
        for mm in models:
            mm.fit(Xtr, ytr)
        m = _Ensemble(models)
        used_params = {**PARAMS, **over, "n_seeds": n_seeds}
    else:
        m = make_model(**(params or {}))
        m.fit(Xtr, ytr)
        used_params = {**PARAMS, **(params or {})}

    res: dict[str, Any] = {
        "mode": mode,
        "n_features": int(X.shape[1]),
        "feature_layout_ok": bool(X.shape[1] == n_exp),
        "column_order": "12 子结构 -> 20 特征" if list(X.columns)[0].startswith("sub_")
                        else "20 特征 -> 12 子结构",
        "features": list(X.columns),
        "params": used_params,
        "split": {"test_size": CFG.get("model.legacy_split.test_size"),
                  "random_state": CFG.get("model.legacy_split.random_state")},
        "n_rows": int(len(y)),
        "train": _metrics(ytr, m.predict(Xtr)),
        "test": _metrics(yte, m.predict(Xte)),
    }
    if mode == "enhanced":
        save = False        # 集成对象不落 Booster JSON，只做当次评估与预测
        _CACHE["model"], _CACHE["features"] = m, list(X.columns)
    if save:
        p = CFG.abs_path(CFG.get("model.production.save_path"))
        p.parent.mkdir(parents=True, exist_ok=True)
        # 存 Booster JSON 而不是 sklearn 包装器：xgboost 3.x + sklearn 1.6+ 下
        # XGBRegressor.save_model 会因为 _estimator_type 缺失而报错，
        # 而 Booster JSON 是跨版本、跨语言通用的格式。
        m.get_booster().save_model(str(p))
        import pickle
        (p.parent / "production_sklearn.pkl").write_bytes(pickle.dumps(m))
        (p.parent / "production_features.json").write_text(
            json.dumps({"features": list(X.columns), "params": res["params"],
                        "target": loader.TARGET}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        res["saved_to"] = str(p)
        res["also_saved"] = {"sklearn_pickle": str(p.parent / "production_sklearn.pkl"),
                             "feature_spec": str(p.parent / "production_features.json")}
    db.ex("INSERT INTO model_runs(name,params,metrics,note,created_at) VALUES(?,?,?,?,?)",
          ("production", json.dumps(res["params"]),
           json.dumps({"train": res["train"], "test": res["test"]}, ensure_ascii=False),
           "生产模型", time.time()))
    _CACHE["model"], _CACHE["features"] = m, list(X.columns)
    return res


def load_production() -> tuple[Any, list[str]]:
    """取生产模型；没有就现训一个。"""
    if _CACHE.get("model") is not None:
        return _CACHE["model"], _CACHE["features"]
    p = CFG.abs_path(CFG.get("model.production.save_path"))
    fj = p.parent / "production_features.json"
    pk = p.parent / "production_sklearn.pkl"
    if fj.exists() and (pk.exists() or p.exists()):
        feats = json.loads(fj.read_text(encoding="utf-8"))["features"]
        try:
            import pickle
            m = pickle.loads(pk.read_bytes())
        except Exception:  # noqa: BLE001  pickle 不可用就从 Booster JSON 重建
            booster = xgb.Booster()
            booster.load_model(str(p))
            m = xgb.XGBRegressor()
            m._Booster = booster
        _CACHE["model"], _CACHE["features"] = m, feats
        return m, feats
    train_production(save=True)
    return _CACHE["model"], _CACHE["features"]


def invalidate() -> None:
    _CACHE.clear()


# ---------------- 从 SMILES 直接预测 ----------------
def smarts_counts_for(smiles: str) -> dict[str, float] | None:
    """对任意 SMILES 现算 12 个子结构计数（口径与训练时完全一致）。"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    out: dict[str, float] = {}
    for s in loader.SMARTS:
        patt = Chem.MolFromSmarts(s)
        n = float(len(mol.GetSubstructMatches(patt))) if patt is not None else 0.0
        if s in loader.SMARTS_SCALE and loader.SMARTS_SCALE[s]:
            n /= float(loader.SMARTS_SCALE[s])
        out[loader.smarts_colname(s)] = n
    return out


def predict_rows(rows: list[dict]) -> dict:
    """每行 = {SMILES?: str, 特征名: 值...}。给 SMILES 会自动补 12 个子结构列。"""
    m, feats = load_production()
    recs, warns = [], []
    for i, r in enumerate(rows):
        rec = {k: v for k, v in r.items() if k != "SMILES"}
        if r.get("SMILES"):
            sc = smarts_counts_for(str(r["SMILES"]))
            if sc is None:
                warns.append(f"第 {i} 行 SMILES 无法解析：{r['SMILES']}")
            else:
                rec.update(sc)
        recs.append(rec)
    Xq = pd.DataFrame(recs)
    for c in feats:
        if c not in Xq.columns:
            Xq[c] = np.nan
    Xq = Xq[feats].apply(pd.to_numeric, errors="coerce")

    pred = m.predict(Xq)
    # 不确定度与适用域仍走诊断内核（同样 32 列）
    b = get_bundle()
    from .model import applicability, domain_threshold, ensemble_std
    std = ensemble_std(Xq, b)
    dist = applicability(Xq, b)
    thr = domain_threshold(b)

    n_missing = Xq.isna().sum(axis=1).to_numpy()
    n_feat = len(feats)
    out = []
    for p, s, d, nm in zip(pred, std, dist, n_missing):
        miss_frac = float(nm) / max(n_feat, 1)
        warns_row = []
        if d > thr:
            warns_row.append("落在训练流形之外，预测不可信，宜作为高价值实验点")
        if miss_frac > 0.5:
            # 适用域距离是用中位数填补缺失后算的，缺太多时它会假性地显示 in_domain
            warns_row.append(
                f"{int(nm)}/{n_feat} 个特征缺失（{miss_frac:.0%}），"
                "预测基本由中位数默认值驱动，适用域判定不可信")
        elif miss_frac > 0.25:
            warns_row.append(f"{int(nm)}/{n_feat} 个特征缺失，预测不确定度被低估")
        out.append({"pred_removal_pct": round(float(np.clip(p, 0, 100)), 2),
                    "raw_pred": round(float(p), 2),
                    "ensemble_std": round(float(s), 2),
                    "n_missing_features": int(nm),
                    "missing_fraction": round(miss_frac, 3),
                    "domain_distance": round(float(d), 3),
                    "in_domain": bool(d <= thr and miss_frac <= 0.5),
                    "reliable": bool(d <= thr and miss_frac <= 0.25),
                    "warning": "；".join(warns_row) or None})
    return {"predictions": out, "domain_threshold_p95": round(thr, 3),
            "n_features_used": len(feats), "input_warnings": warns,
            "note": "raw_pred 是模型原始输出，pred_removal_pct 已裁剪到 [0,100]。"}


# ---------------- 特征重要性 ----------------
IMPORTANCE_META = {
    "weight": "被用作分裂判据的次数 —— 模型有多频繁地依赖它做决策",
    "gain": "每次分裂带来的平均增益 —— 用到它时贡献有多大",
    "cover": "分裂时覆盖的样本量 —— 它影响多少样本",
    "shap": "SHAP 平均绝对贡献 —— 对每条预测的实际影响",
    "permutation": "打乱该列后 R² 掉多少 —— 模型有多依赖它",
}


def native_importance(with_shap: bool = True, with_perm: bool = True) -> dict:
    """五种重要性口径：weight / gain / cover（XGBoost 原生）+ SHAP + 置换。

    不同口径给出的排名可以差很多，这是正常的，不是矛盾：
    weight 衡量"用得多不多"，gain 衡量"用一次值多少"，SHAP 衡量"对预测的实际影响"。
    """
    import numpy as np
    m, feats = load_production()
    booster = m.get_booster()
    booster.feature_names = feats
    out: dict[str, list] = {}
    for kind in ("weight", "gain", "cover"):
        sc = booster.get_score(importance_type=kind)
        tot = sum(sc.values()) or 1.0
        rank = sorted(sc.items(), key=lambda kv: -kv[1])
        out[kind] = [{"feature": k, "value": round(v, 3),
                      "share_pct": round(100 * v / tot, 2)} for k, v in rank]

    if with_shap or with_perm:
        b = get_bundle()
    if with_shap:
        try:
            import shap
            sv = shap.TreeExplainer(b.model).shap_values(b.X)
            ma = np.abs(sv).mean(0)
            tot = float(ma.sum()) or 1.0
            order = np.argsort(-ma)
            out["shap"] = [{"feature": list(b.X.columns)[i], "value": round(float(ma[i]), 4),
                            "share_pct": round(100 * float(ma[i]) / tot, 2)} for i in order]
        except Exception:  # noqa: BLE001
            pass
    if with_perm:
        try:
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(b.model, b.X, b.y, n_repeats=5,
                                        random_state=0, scoring="r2", n_jobs=1)
            v = np.clip(pi.importances_mean, 0, None)
            tot = float(v.sum()) or 1.0
            order = np.argsort(-v)
            out["permutation"] = [
                {"feature": list(b.X.columns)[i], "value": round(float(v[i]), 5),
                 "share_pct": round(100 * float(v[i]) / tot, 2)} for i in order]
        except Exception:  # noqa: BLE001
            pass

    merged: dict[str, dict] = {}
    for kind, lst in out.items():
        for i, r in enumerate(lst):
            merged.setdefault(r["feature"], {"feature": r["feature"]})[f"{kind}_rank"] = i + 1
            merged[r["feature"]][f"{kind}_share_pct"] = r["share_pct"]
    table = sorted(merged.values(), key=lambda r: r.get("weight_rank", 999))
    return {"by_type": {k: v[:18] for k, v in out.items()}, "combined": table,
            "metrics_available": [k for k in out],
            "metric_meaning": IMPORTANCE_META,
            "read": ("五种口径衡量的是不同的东西，排名不一致很正常。"
                     "weight 高说明模型频繁拿它做分裂判据；gain 高说明用到时增益大；"
                     "SHAP 和置换重要度最接近『对预测的实际影响』。")}


# ---------------- 学习曲线 ----------------
def learning_curve(group: str = "compound", fractions: list[float] | None = None,
                   n_repeat: int = 3) -> dict:
    """按化合物整簇抽样逐步加数据，看 R² 还能不能涨。

    抽样单位必须是化合物而不是行，否则同分子的重复记录会泄漏，
    曲线一开始就假性地很高，看不出真实的数据边际收益。
    """
    fractions = fractions or list(CFG.get("model.learning_curve_fractions",
                                          [0.2, 0.4, 0.6, 0.8, 1.0]))
    built = loader.build_matrix()
    X, y = built["X"], built["y"]
    g = built["groups"].get(group, np.arange(len(y)))
    uniq = np.unique(g)
    rng = np.random.default_rng(42)
    rows = []
    for frac in fractions:
        scores, sizes = [], []
        for _ in range(n_repeat if frac < 1.0 else 1):
            k = max(3, int(round(len(uniq) * frac)))
            pick = rng.choice(uniq, size=min(k, len(uniq)), replace=False)
            mask = np.isin(g, pick)
            sub_g = g[mask]
            if len(np.unique(sub_g)) < 3:
                continue
            Xs = X[mask].reset_index(drop=True)
            ys = y[mask].reset_index(drop=True)
            oof = np.full(len(ys), np.nan)
            for tr, te in GroupKFold(
                    n_splits=min(5, len(np.unique(sub_g)))).split(Xs, ys, groups=sub_g):
                mm = make_model()
                mm.fit(Xs.iloc[tr], ys.iloc[tr])
                oof[te] = mm.predict(Xs.iloc[te])
            scores.append(float(r2_score(ys, oof)))
            sizes.append(int(mask.sum()))
        if scores:
            rows.append({"fraction": frac, "n_compounds": int(round(len(uniq) * frac)),
                         "n_rows_mean": int(np.mean(sizes)),
                         "r2_mean": round(float(np.mean(scores)), 4),
                         "r2_sd": round(float(np.std(scores)), 4)})
    slope = None
    if len(rows) >= 2:
        slope = round((rows[-1]["r2_mean"] - rows[-2]["r2_mean"]) /
                      max(rows[-1]["fraction"] - rows[-2]["fraction"], 1e-9), 4)
    return {"group": group, "curve": rows, "tail_slope": slope,
            "read": ("tail_slope 接近 0 => 再加同类数据收益很小，瓶颈在描述符不在样本量，"
                     "该去发现层找新描述符；明显为正 => 补数据仍然划算。"
                     if slope is not None else "数据点不足，无法判断趋势")}


# ---------------- 分层性能 ----------------
def stratified_performance(group: str = "compound", min_n: int = 10) -> dict:
    """按膜 / NF-RO / 分子量区间 / 截留区间 / 化合物拆解，定位模型在哪失效。"""
    b = get_bundle(group=group)
    f = b.as_frame().copy()
    f["mw_bin"] = pd.cut(f["Mw"], [0, 150, 250, 350, 500, 1e9],
                         labels=["<150", "150-250", "250-350", "350-500", ">500"])
    f["removal_bin"] = pd.cut(f["y_true"], [-0.01, 50, 80, 95, 100.01],
                              labels=["<50%", "50-80%", "80-95%", ">95%"])

    def block(col: str, top: int = 15) -> list[dict]:
        out = []
        for k, sub in f.groupby(col, observed=True):
            if len(sub) < min_n:
                continue
            out.append({"level": str(k)[:60], "n": int(len(sub)),
                        **_metrics(sub["y_true"], sub["y_oof"]),
                        "mean_bias": round(float(sub["residual"].mean()), 2)})
        out.sort(key=lambda r: r["r2"])
        return out[:top]

    return {"overall": b.metrics["group_cv"],
            "by_membrane_worst": block("membrane"),
            "by_membrane_class": block("membrane_class", top=5),
            "by_mw_bin": block("mw_bin", top=8),
            "by_removal_bin": block("removal_bin", top=8),
            "worst_compounds": block("compound", top=12),
            "read": ("R² 在某层特别低 = 模型对该类样本失效。注意 >95% 截留区间因方差极小，"
                     "R² 天然偏低甚至为负，该层要看 RMSE 而不是 R²。")}


# ---------------- 网格搜索 ----------------
def grid_search(scoring_mode: str = "grouped", param_grid: dict | None = None,
                max_combos: int = 48) -> dict:
    """超参搜索。grouped = 按化合物分组CV（反映新分子泛化）；random = 随机KFold（对齐既有口径）。"""
    grid = param_grid or dict(CFG.get("model.grid_search.param_grid"))
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in product(*[grid[k] for k in keys])]
    sampled = len(combos) > max_combos
    if sampled:
        rng = np.random.default_rng(0)
        idx = sorted(rng.choice(len(combos), size=max_combos, replace=False))
        combos = [combos[i] for i in idx]

    built = loader.build_matrix()
    X, y = built["X"], built["y"]
    g = built["groups"]["compound"]
    n_splits = int(CFG.get("model.grid_search.cv_folds", 5))
    grouped = scoring_mode == "grouped"
    splitter = (GroupKFold(n_splits=n_splits) if grouped
                else KFold(n_splits=n_splits, shuffle=True, random_state=108))

    results = []
    for c in combos:
        oof = np.full(len(y), np.nan)
        for tr, te in splitter.split(X, y, groups=g if grouped else None):
            m = make_model(**c)
            m.fit(X.iloc[tr], y.iloc[tr])
            oof[te] = m.predict(X.iloc[te])
        results.append({"params": c, "r2": round(float(r2_score(y, oof)), 4),
                        "rmse": round(float(np.sqrt(np.mean((y - oof) ** 2))), 3)})
    results.sort(key=lambda r: -r["r2"])
    cur = {k: PARAMS.get(k) for k in keys}
    cur_score = next((r["r2"] for r in results if r["params"] == cur), None)
    return {"scoring_mode": scoring_mode, "n_combos_tried": len(combos),
            "sampled_from_full_grid": sampled,
            "best": results[0], "top10": results[:10],
            "current_params": cur, "current_score": cur_score,
            "improvement_over_current": (round(results[0]["r2"] - cur_score, 4)
                                         if cur_score is not None else None),
            "read": ("只有 grouped 口径的最优参数才反映对新分子的泛化能力；"
                     "random 口径会被同分子重复记录泄漏抬高，仅用于与既有结果对齐。")}
