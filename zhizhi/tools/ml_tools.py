"""量衡 LIANGHENG —— 模型层工具集。

对外只暴露"结论 + 少量数字"，大表落盘并返回指针，避免撑爆上下文。
"""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import CFG
from ..core.tools import P, obj, tool
from ..dataio import loader
from ..desc import store as dstore
from ..ml import model as M


def _dump(name: str, df: pd.DataFrame) -> str:
    p = CFG.logs_dir / f"{name}_{int(time.time())}.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return str(p)


def _bundle(descriptors: list[str] | None = None, drop: list[str] | None = None,
            group: str = "compound") -> M.Bundle:
    extra = dstore.active_extra(descriptors) if descriptors else None
    return M.get_bundle(extra=extra, drop=drop, group=group)


# =====================================================================
@tool("ml_data_health",
      "数据底座体检：样本量、化合物/膜/文献数、覆盖度、缺失结构、重复度。"
      "任何分析前先看这个。",
      obj({}), category="model")
def ml_data_health() -> dict:
    return loader.data_health()


@tool("ml_model_report",
      "模型总览：既有工作的随机切分口径 vs 按化合物/膜/文献分组的交叉验证，"
      "并给出泄漏幅度。理解模型真实外推能力的入口。",
      obj({}), category="model")
def ml_model_report() -> dict:
    return M.legacy_report()


@tool("ml_predict",
      "对给定条件预测截留率，附预测不确定度与适用域判定（域外会明确报警）。",
      obj({"rows": P("array", "待预测样本列表，每个是 {特征名: 值} 字典；"
                     "缺的特征留空即可（模型原生处理缺失）",
                     items={"type": "object"})}, ["rows"]), category="model")
def ml_predict(rows: list[dict]) -> dict:
    b = _bundle()
    Xq = pd.DataFrame(rows)
    for c in b.X.columns:
        if c not in Xq.columns:
            Xq[c] = np.nan
    Xq = Xq[b.X.columns].apply(pd.to_numeric, errors="coerce")
    pred = b.model.predict(Xq)
    std = M.ensemble_std(Xq, b)
    dist = M.applicability(Xq, b)
    thr = M.domain_threshold(b)
    return {"predictions": [
        {"pred_removal_pct": round(float(p), 2),
         "ensemble_std": round(float(s), 2),
         "domain_distance": round(float(d), 3),
         "in_domain": bool(d <= thr),
         "warning": None if d <= thr else "样本落在训练流形之外，预测不可信，"
                                          "应作为高价值实验点而非结论"}
        for p, s, d in zip(pred, std, dist)],
        "domain_threshold_p95": round(thr, 3),
        "model_group_cv_r2": b.metrics["group_cv"]["r2"]}


@tool("ml_residuals",
      "取分组交叉验证的 OOF 残差（不是训练残差！）。返回残差最大的样本、"
      "按化合物/膜/文献聚合的系统性偏差，以及残差与缺失模式的重合度检查。",
      obj({"group_by": P("string", "分组方式", enum=["compound", "membrane", "reference"]),
           "top_pct": P("number", "取绝对残差前多少比例，默认 0.2"),
           "top_n_list": P("integer", "详细列出的样本条数，默认 25")}),
      category="model")
def ml_residuals(group_by: str = "compound", top_pct: float = 0.2,
                 top_n_list: int = 25) -> dict:
    b = _bundle(group=group_by)
    f = b.as_frame()
    thr = float(np.quantile(f["abs_residual"], 1 - top_pct))
    top = f[f["abs_residual"] >= thr].sort_values("abs_residual", ascending=False)
    path = _dump(f"residuals_{group_by}", f)

    def agg(col: str, min_n: int = 3) -> list[dict]:
        g = f.groupby(col)["residual"].agg(["mean", "std", "count"])
        g = g[g["count"] >= min_n].sort_values("mean")
        out = []
        for side in (g.head(6), g.tail(6)):
            for k, r in side.iterrows():
                out.append({col: k, "mean_residual": round(float(r["mean"]), 2),
                            "sd": round(float(r["std"]), 2), "n": int(r["count"])})
        return out

    # 残差与缺失模式的重合度：|残差| 与各 _isna 指示位的点二列相关
    isna_cols = [c for c in b.X.columns if c.endswith("__isna")]
    miss_link = {}
    for c in isna_cols:
        v = b.X[c].to_numpy(float)
        if 0 < v.sum() < len(v):
            miss_link[c] = round(float(np.corrcoef(v, f["abs_residual"])[0, 1]), 3)
    miss_link = dict(sorted(miss_link.items(), key=lambda kv: -abs(kv[1]))[:6])

    return {
        "group_by": group_by,
        "cv_metrics": b.metrics["group_cv"],
        "residual_convention": "residual = y_true - y_oof；正=模型低估(实测截留更高)，负=模型高估",
        "abs_residual_threshold": round(thr, 2),
        "n_top": int(len(top)),
        "top_samples": top.head(top_n_list)[
            ["compound", "membrane", "membrane_class", "SMILES", "Mw",
             "y_true", "y_oof", "residual", "reference"]
        ].round(2).to_dict("records"),
        "systematic_by_compound": agg("compound"),
        "systematic_by_membrane": agg("membrane"),
        "systematic_by_reference": agg("reference", min_n=5),
        "residual_vs_missingness_corr": miss_link,
        "missingness_warning": ("若某 _isna 与 |残差| 相关系数明显偏离 0，"
                                "该残差簇更可能是数据缺口而非未建模机理，归因时必须先排除。"),
        "full_table_csv": path,
    }


@tool("ml_ablate",
      "语义分组消融：逐组剔除特征后看分组 CV 掉多少，并报告哪类分子退化最严重。"
      "用来发现『哪组特征在拖后腿 / 哪组根本没用』。",
      obj({"groups": P("array", "要消融的特征组名，留空则全部。可选："
                       "size_exclusion, hydrophobicity, electrostatics, abraham_lfer, "
                       "membrane_geom, operating, substructure",
                       items={"type": "string"}),
           "group_by": P("string", "CV 分组方式，默认 compound")}), category="model")
def ml_ablate(groups: list[str] | None = None, group_by: str = "compound") -> dict:
    all_groups = list((CFG.get("data.feature_groups") or {}).keys())
    groups = groups or all_groups
    base = _bundle(group=group_by)
    base_r2 = base.metrics["group_cv"]["r2"]
    base_f = base.as_frame()

    rows = []
    for g in groups:
        if g not in all_groups:
            continue
        b = M.get_bundle(drop=[f"__GROUP__:{g}"], group=group_by)
        d = b.as_frame()
        # 哪类分子退化最严重（按化合物聚合 |残差| 增幅）
        merged = base_f[["compound", "abs_residual"]].rename(
            columns={"abs_residual": "base"}).copy()
        merged["abl"] = d["abs_residual"].to_numpy()
        deg = (merged.groupby("compound")[["base", "abl"]].mean()
               .assign(delta=lambda x: x["abl"] - x["base"])
               .sort_values("delta", ascending=False))
        rows.append({
            "group": g,
            "features_removed": int(base.X.shape[1] - b.X.shape[1]),
            "r2_without": b.metrics["group_cv"]["r2"],
            "delta_r2": round(b.metrics["group_cv"]["r2"] - base_r2, 4),
            "worst_hit_compounds": [
                {"compound": k, "abs_res_increase": round(float(v), 2)}
                for k, v in deg["delta"].head(5).items()],
        })
    rows.sort(key=lambda r: r["delta_r2"])
    return {"baseline_r2": base_r2, "group_by": group_by, "ablation": rows,
            "read": "delta_r2 越负 = 该组越重要；接近 0 或为正 = 该组无用甚至在拖后腿。"}


@tool("ml_extrapolate",
      "外推压力测试：留一化合物 / 留一膜 / 留一文献，模拟『新分子、新膜、新课题组』"
      "三种真实使用场景。并给出适用域覆盖率。",
      obj({"modes": P("array", "要跑的模式", items={"type": "string",
           "enum": ["compound", "membrane", "reference", "membrane_class"]})}),
      category="model")
def ml_extrapolate(modes: list[str] | None = None) -> dict:
    modes = modes or ["compound", "membrane", "reference"]
    out: dict[str, Any] = {}
    for m in modes:
        if m == "membrane_class":
            # NF 训练 -> RO 测试，反之亦然（最硬的外推）
            built = loader.build_matrix()
            X, y = built["X"], built["y"]
            cls = built["groups"]["membrane_class"]
            res = {}
            for tr_c, te_c in (("NF", "RO"), ("RO", "NF")):
                tr, te = cls == tr_c, cls == te_c
                if tr.sum() < 50 or te.sum() < 20:
                    continue
                mdl = M.make_model()
                mdl.fit(X[tr], y[tr])
                res[f"{tr_c}->{te_c}"] = M._metrics(y[te], mdl.predict(X[te]))
            out[m] = res
        else:
            b = M.get_bundle(group=m)
            out[m] = b.metrics["group_cv"]
    base = M.get_bundle(group="compound")
    dist = M.applicability(base.X, base)
    return {"extrapolation": out,
            "in_sample_reference": base.metrics["in_sample"],
            "domain_distance_quantiles": {
                q: round(float(np.percentile(dist, q)), 3) for q in (50, 75, 90, 95, 99)},
            "read": ("三种分组 R² 的落差揭示不同的失效来源："
                     "留一化合物差 = 分子描述符不够；留一膜差 = 膜描述符不够；"
                     "留一文献差 = 存在未记录的实验协议变量。")}


@tool("ml_explain",
      "SHAP 解释：全局重要性、交互作用、指定子群的分层 SHAP、单样本归因。",
      obj({"scope": P("string", "global=全局 | interaction=交互 | subgroup=分层 | local=单样本",
                      enum=["global", "interaction", "subgroup", "local"]),
           "subgroup_query": P("string", "scope=subgroup 时的 pandas query，"
                               "可用列 compound/membrane/membrane_class/Mw/y_true"),
           "row_id": P("integer", "scope=local 时的样本 row_id"),
           "top_k": P("integer", "返回前 k 个特征，默认 15")}), category="model")
def ml_explain(scope: str = "global", subgroup_query: str = "",
               row_id: int = 0, top_k: int = 15) -> dict:
    import shap
    b = _bundle()
    expl = shap.TreeExplainer(b.model)
    sv = expl.shap_values(b.X)
    names = list(b.X.columns)

    if scope == "global":
        imp = np.abs(sv).mean(0)
        order = np.argsort(-imp)[:top_k]
        # 方向性：SHAP 与特征值的相关号
        signs = []
        for i in order:
            v = b.X.iloc[:, i].to_numpy(float)
            ok = ~np.isnan(v)
            c = np.corrcoef(v[ok], sv[ok, i])[0, 1] if ok.sum() > 10 else np.nan
            signs.append(None if np.isnan(c) else ("升高→截留升高" if c > 0 else "升高→截留降低"))
        return {"scope": "global",
                "top_features": [{"feature": names[i], "mean_abs_shap": round(float(imp[i]), 3),
                                  "direction": s} for i, s in zip(order, signs)],
                "base_value": round(float(expl.expected_value), 2)}

    if scope == "interaction":
        # 交互矩阵在 2102×39 上可算，但只回传 top 对
        sub = b.X.sample(min(400, len(b.X)), random_state=0)
        iv = shap.TreeExplainer(b.model).shap_interaction_values(sub)
        mat = np.abs(iv).mean(0)
        np.fill_diagonal(mat, 0)
        pairs = []
        idx = np.dstack(np.unravel_index(np.argsort(-mat, axis=None), mat.shape))[0]
        seen = set()
        for i, j in idx:
            if (j, i) in seen or i == j:
                continue
            seen.add((i, j))
            pairs.append({"pair": [names[i], names[j]],
                          "mean_abs_interaction": round(float(mat[i, j]), 3)})
            if len(pairs) >= top_k:
                break
        return {"scope": "interaction", "top_interactions": pairs,
                "note": "在 400 样本子集上计算以控制耗时。"}

    if scope == "subgroup":
        f = b.as_frame()
        mask = f.eval(subgroup_query).to_numpy() if subgroup_query else np.ones(len(f), bool)
        if mask.sum() < 5:
            return {"error": f"子群样本太少 ({int(mask.sum())})，换个 query"}
        imp_all = np.abs(sv).mean(0)
        imp_sub = np.abs(sv[mask]).mean(0)
        order = np.argsort(-imp_sub)[:top_k]
        return {"scope": "subgroup", "query": subgroup_query, "n": int(mask.sum()),
                "mean_residual_in_subgroup": round(float(f.loc[mask, "residual"].mean()), 2),
                "features": [{"feature": names[i],
                              "shap_subgroup": round(float(imp_sub[i]), 3),
                              "shap_global": round(float(imp_all[i]), 3),
                              "enrichment": round(float(imp_sub[i] / (imp_all[i] + 1e-9)), 2)}
                             for i in order]}

    f = b.as_frame()
    hit = f.index[f["row_id"] == row_id]
    if len(hit) == 0:
        return {"error": f"row_id {row_id} 不存在"}
    i = int(hit[0])
    order = np.argsort(-np.abs(sv[i]))[:top_k]
    return {"scope": "local", "row_id": row_id,
            "compound": f.loc[i, "compound"], "membrane": f.loc[i, "membrane"],
            "y_true": round(float(f.loc[i, "y_true"]), 2),
            "y_oof": round(float(f.loc[i, "y_oof"]), 2),
            "contributions": [{"feature": names[j], "value": None if pd.isna(b.X.iloc[i, j])
                               else round(float(b.X.iloc[i, j]), 4),
                               "shap": round(float(sv[i, j]), 3)} for j in order]}


@tool("ml_counterfactual",
      "反事实分析。两条通道：(a) synthetic=在物理约束下搜索最小特征改动以达到目标截留；"
      "(b) matched=从真实数据里找结构高度相似但截留差异大的分子对（写论文更有说服力）。",
      obj({"mode": P("string", "synthetic | matched", enum=["synthetic", "matched"]),
           "row_id": P("integer", "synthetic 模式的起始样本"),
           "target_removal": P("number", "synthetic 模式的目标截留率 %"),
           "mutable": P("array", "允许改动的特征名，留空则用操作条件+膜性质",
                        items={"type": "string"}),
           "min_tanimoto": P("number", "matched 模式的最小结构相似度，默认 0.6"),
           "min_gap": P("number", "matched 模式的最小截留差，默认 25")}), category="model")
def ml_counterfactual(mode: str = "matched", row_id: int = 0,
                      target_removal: float = 90.0, mutable: list[str] | None = None,
                      min_tanimoto: float = 0.6, min_gap: float = 25.0) -> dict:
    b = _bundle()
    f = b.as_frame()

    if mode == "matched":
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        RDLogger.DisableLog("rdApp.*")
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        agg = (f.groupby(["membrane", "compound", "SMILES"])["y_true"]
               .mean().reset_index())
        out = []
        for mb, sub in agg.groupby("membrane"):
            if len(sub) < 2:
                continue
            mols = [Chem.MolFromSmiles(s) for s in sub["SMILES"]]
            fps = [gen.GetFingerprint(m) if m else None for m in mols]
            arr = sub.reset_index(drop=True)
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    if fps[i] is None or fps[j] is None:
                        continue
                    t = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    gap = abs(arr.loc[i, "y_true"] - arr.loc[j, "y_true"])
                    if t >= min_tanimoto and gap >= min_gap:
                        out.append({"membrane": mb, "tanimoto": round(t, 3),
                                    "removal_gap": round(float(gap), 1),
                                    "A": {"compound": arr.loc[i, "compound"],
                                          "SMILES": arr.loc[i, "SMILES"],
                                          "removal": round(float(arr.loc[i, "y_true"]), 1)},
                                    "B": {"compound": arr.loc[j, "compound"],
                                          "SMILES": arr.loc[j, "SMILES"],
                                          "removal": round(float(arr.loc[j, "y_true"]), 1)}})
        out.sort(key=lambda d: (-d["tanimoto"], -d["removal_gap"]))
        return {"mode": "matched", "n_pairs": len(out), "pairs": out[:20],
                "read": ("结构极相似却截留差异巨大的真实分子对 = 现有描述符语言的裂缝所在，"
                         "是新描述符最该解释的对象，也是 L3 判别实验的天然候选。")}

    hit = f.index[f["row_id"] == row_id]
    if len(hit) == 0:
        return {"error": f"row_id {row_id} 不存在"}
    i = int(hit[0])
    base_row = b.X.iloc[i]
    default_mutable = [c for c in b.X.columns if c in
                       (CFG.get("data.feature_groups.operating", []) +
                        CFG.get("data.feature_groups.membrane_geom", []) +
                        ["MB contact angle (°)", "MB zeta potential", "pH"])]
    mut = [c for c in (mutable or default_mutable) if c in b.X.columns]
    if not mut:
        return {"error": "没有可改动的特征"}
    lo = b.X[mut].quantile(0.02)
    hi = b.X[mut].quantile(0.98)
    rng = np.random.default_rng(7)
    best = None
    cand = pd.DataFrame([base_row] * 4000).reset_index(drop=True)
    for c in mut:
        cand[c] = rng.uniform(lo[c], hi[c], size=len(cand))
    pred = b.model.predict(cand)
    okmask = np.abs(pred - target_removal) < 2.0
    if okmask.any():
        span = (hi - lo).replace(0, 1.0)
        cost = (np.abs(cand.loc[okmask, mut] - base_row[mut]) / span).sum(axis=1)
        k = cost.idxmin()
        best = {"achieved_pred": round(float(pred[list(cand.index).index(k)]), 2),
                "changes": {c: {"from": None if pd.isna(base_row[c]) else round(float(base_row[c]), 4),
                                "to": round(float(cand.loc[k, c]), 4)}
                            for c in mut if abs(float(cand.loc[k, c]) -
                                                float(base_row[c] if not pd.isna(base_row[c]) else 0)) > 1e-9},
                "normalized_cost": round(float(cost.min()), 3)}
    return {"mode": "synthetic", "row_id": row_id,
            "compound": f.loc[i, "compound"], "membrane": f.loc[i, "membrane"],
            "y_true": round(float(f.loc[i, "y_true"]), 2),
            "target": target_removal, "solution": best,
            "note": ("未找到解说明在可调操作窗口内达不到目标 —— 这本身是结论。"
                     if best is None else
                     "合成反事实只在模型适用域内成立，须与 matched 真实对照互相印证。")}


@tool("ml_mixed_effects",
      "分层（混合效应）模型：把膜和文献作为随机效应，检验某个描述符的固定效应"
      "是否在扣除膜间/课题组间差异后依然稳健。Abraham 五参数 LFER 的正确打开方式。",
      obj({"fixed": P("array", "固定效应特征名列表", items={"type": "string"}),
           "group": P("string", "随机效应分组：membrane 或 reference",
                      enum=["membrane", "reference"])}, ["fixed"]), category="model")
def ml_mixed_effects(fixed: list[str], group: str = "membrane") -> dict:
    import statsmodels.formula.api as smf
    b = _bundle()
    f = b.as_frame()
    cols = [c for c in fixed if c in b.X.columns]
    unknown = [c for c in fixed if c not in b.X.columns]
    if not cols:
        return {"error": f"没有可用的固定效应列。未知: {unknown}",
                "available": list(b.X.columns)}
    d = b.X[cols].copy()
    d["y"] = b.y.to_numpy()
    d["grp"] = f[group].to_numpy()
    d = d.dropna()
    if len(d) < 60:
        return {"error": f"完整样本仅 {len(d)} 行，不足以拟合分层模型",
                "hint": "换用缺失率更低的特征，或减少固定效应个数"}
    safe = {c: f"v{i}" for i, c in enumerate(cols)}
    d = d.rename(columns=safe)
    formula = "y ~ " + " + ".join(safe.values())
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        try:
            res = smf.mixedlm(formula, d, groups=d["grp"]).fit(reml=True, method="lbfgs")
        except Exception as e:  # noqa: BLE001
            return {"error": f"拟合失败: {e}"}
    conv_msgs = [str(x.message)[:160] for x in caught
                 if "Convergence" in type(x.message).__name__ or "converge" in str(x.message).lower()]
    converged = bool(getattr(res, "converged", True)) and not conv_msgs
    inv = {v: k for k, v in safe.items()}
    coefs = []
    for name, val in res.params.items():
        if name in inv:
            coefs.append({"feature": inv[name], "coef": round(float(val), 4),
                          "p": round(float(res.pvalues[name]), 5),
                          "significant": bool(res.pvalues[name] < 0.05)})
    out = {"n_used": int(len(d)), "n_groups": int(d["grp"].nunique()),
           "random_effect": group, "coefficients": coefs,
           "group_variance": round(float(res.cov_re.iloc[0, 0]), 4),
           "residual_variance": round(float(res.scale), 4),
           "converged": converged,
           "note": ("注意：本分析对完整观测子集拟合（缺失行被丢弃），"
                    f"从 {len(b.y)} 行降到 {len(d)} 行，结论只对该子集成立。"),
           "read": ("group_variance 远大于 0 说明膜/课题组间存在系统差异，"
                    "任何忽略它的朴素回归都会给出误导性的系数。")}
    if not converged:
        out["convergence_warnings"] = conv_msgs
        out["WARNING"] = ("★ 模型未收敛（常见于 group_variance 贴近 0 或完整样本太少）。"
                          "此时系数与 p 值不可信，**不得**用它下任何结论，"
                          "只能说明该子集数据不足以支撑分层模型。")
        for c in coefs:
            c["significant"] = None
    return out


@tool("ml_add_descriptor",
      "★ 描述符检验闭环的核心。把一个已计算好的新描述符加入特征集并重训，"
      "报告 ΔR²(分组CV, bootstrap CI)、扣除共线性后的偏 R²、置换重要度、"
      "y-scrambling 负对照、以及它是否修好了目标残差簇。",
      obj({"name": P("string", "描述符名（须已通过 disc_compute_descriptor 计算并入库）"),
           "target_subgroup": P("string", "该描述符本来要解释的子群 pandas query，"
                                "如 \"membrane_class=='NF' and Mw<250\"；留空则跳过定向检验"),
           "group_by": P("string", "CV 分组，默认 compound")}, ["name"]), category="model")
def ml_add_descriptor(name: str, target_subgroup: str = "",
                      group_by: str = "compound") -> dict:
    vals = dstore.load_values(name)
    if not vals:
        return {"error": f"描述符 {name} 没有已计算的值，先调用 disc_compute_descriptor"}

    base = M.get_bundle(group=group_by)
    new = M.get_bundle(extra={name: vals}, group=group_by, refit=True)
    if name not in new.X.columns:
        return {"error": "描述符未成功并入特征矩阵"}

    r2_b, r2_n = base.metrics["group_cv"]["r2"], new.metrics["group_cv"]["r2"]
    d_r2 = r2_n - r2_b

    # 1) 冗余度：与既有特征的最大 |相关|
    v = new.X[name].to_numpy(float)
    ok = ~np.isnan(v)
    cover = float(ok.mean())
    corrs = {}
    for c in base.X.columns:
        u = base.X[c].to_numpy(float)
        m = ok & ~np.isnan(u)
        if m.sum() > 30 and np.std(u[m]) > 0 and np.std(v[m]) > 0:
            corrs[c] = float(np.corrcoef(u[m], v[m])[0, 1])
    top_corr = sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:5]
    max_abs_r = abs(top_corr[0][1]) if top_corr else 0.0

    # 2) bootstrap CI of ΔR²（对 OOF 预测做 bootstrap，按化合物整簇重采样）
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(11)
    comp = base.meta["compound"].to_numpy()
    uniq = np.unique(comp)
    yb, ob, on = base.y.to_numpy(), base.oof, new.oof
    deltas = []
    for _ in range(int(CFG.get("discovery.n_bootstrap", 200))):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(comp == c) for c in pick])
        try:
            deltas.append(r2_score(yb[idx], on[idx]) - r2_score(yb[idx], ob[idx]))
        except Exception:  # noqa: BLE001
            continue
    ci = (round(float(np.percentile(deltas, 2.5)), 4),
          round(float(np.percentile(deltas, 97.5)), 4)) if deltas else (None, None)
    p_boot = float(np.mean(np.asarray(deltas) <= 0)) if deltas else 1.0

    # 3) y-scrambling 负对照：打乱描述符值后 ΔR² 的分布
    scr = []
    smis = list(vals.keys())
    for s in range(int(CFG.get("discovery.n_yscramble", 20))):
        perm = rng.permutation(list(vals.values()))
        fake = dict(zip(smis, [float(x) for x in perm]))
        bb = M.get_bundle(extra={f"{name}__scr": fake}, group=group_by,
                          refit=True, cache=False)
        scr.append(bb.metrics["group_cv"]["r2"] - r2_b)
    scr_arr = np.asarray(scr) if scr else np.zeros(1)
    p_scramble = float(np.mean(scr_arr >= d_r2))

    # 4) 置换重要度（在新模型上）
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(new.model, new.X, new.y, n_repeats=5,
                                random_state=0, scoring="r2", n_jobs=1)
    rank = int(np.argsort(-pi.importances_mean).tolist().index(
        list(new.X.columns).index(name))) + 1

    # 5) 定向检验：目标残差簇有没有被修好
    targeted = None
    if target_subgroup:
        fb, fn = base.as_frame(), new.as_frame()
        try:
            mask = fb.eval(target_subgroup).to_numpy()
        except Exception as e:  # noqa: BLE001
            return {"error": f"target_subgroup 无法解析: {e}"}
        if mask.sum() >= 5:
            targeted = {
                "query": target_subgroup, "n": int(mask.sum()),
                "mean_abs_residual_before": round(float(fb.loc[mask, "abs_residual"].mean()), 2),
                "mean_abs_residual_after": round(float(fn.loc[mask, "abs_residual"].mean()), 2),
                "mean_bias_before": round(float(fb.loc[mask, "residual"].mean()), 2),
                "mean_bias_after": round(float(fn.loc[mask, "residual"].mean()), 2)}
            targeted["fixed"] = bool(
                targeted["mean_abs_residual_after"] < targeted["mean_abs_residual_before"] * 0.9)

    # 6) 多重比较：这里做保守的 Bonferroni 单点闸门（分母 = 历史检验过的描述符数 + 本次）。
    #    真正的 BH-FDR 排序校正在 val_l1_battery 里对全体描述符统一做。
    n_tests = max(dstore.n_tested() + 1, 1)
    alpha = float(CFG.get("discovery.fdr_alpha", 0.10))
    p_comb = max(p_boot, p_scramble)
    bonferroni_pass = bool(p_comb * n_tests <= alpha)

    thr = float(CFG.get("discovery.pass_delta_r2", 0.005))
    redundant = max_abs_r >= float(CFG.get("discovery.redundancy_r_threshold", 0.90))
    verdict = ("REDUNDANT" if redundant else
               "PASS" if (d_r2 >= thr and (ci[0] or -1) > 0 and p_scramble < 0.05
                          and bonferroni_pass)
               else "WEAK" if d_r2 >= thr else "FAIL")

    metrics = {
        "coverage": round(cover, 3),
        "r2_base": r2_b, "r2_with": r2_n, "delta_r2": round(d_r2, 4),
        "delta_r2_ci95": ci, "p_bootstrap": round(p_boot, 4),
        "yscramble_delta_r2_mean": round(float(scr_arr.mean()), 4),
        "yscramble_delta_r2_p95": round(float(np.percentile(scr_arr, 95)), 4),
        "p_yscramble": round(p_scramble, 4),
        "max_abs_corr_with_existing": round(max_abs_r, 3),
        "most_correlated": [{"feature": k, "r": round(v, 3)} for k, v in top_corr],
        "permutation_importance_rank": rank,
        "n_features_total": int(new.X.shape[1]),
        "targeted_test": targeted,
        "fdr": {"n_tests_so_far": n_tests, "alpha": alpha,
                "p_combined": round(p_comb, 4), "method": "Bonferroni gate",
                "pass": bonferroni_pass},
        "verdict": verdict,
    }
    dstore.set_status(name, {"PASS": "passed", "REDUNDANT": "redundant"}.get(verdict, "tested"),
                      metrics)
    metrics["read"] = {
        "PASS": "通过全部检验：有增量、CI 不跨零、优于打乱对照、经多重比较校正仍成立。",
        "WEAK": "有增量但未通过负对照或多重比较闸门，不能宣称为新知识，可挂起等更多数据。",
        "FAIL": "无增量。这也是结论：该假设在现有数据上不成立。",
        "REDUNDANT": f"与既有特征相关系数 {max_abs_r:.2f}，是旧信息的换皮，不算新描述符。",
    }[verdict]
    return metrics


@tool("ml_data_qc",
      "★ 数据完整性核查：用 RDKit 从 SMILES 反算分子量/式，与表中报告的 Mw 对照，"
      "揪出 SMILES 错标（错标会直接污染 12 个子结构特征，并伪装成『机理发现』）。"
      "同时检查重复行、目标值越界、单点组合。归因任何残差前必须先跑这个。",
      obj({"tolerance_pct": P("number", "Mw 相对容差，默认 0.02")}), category="model")
def ml_data_qc(tolerance_pct: float = 0.02) -> dict:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")

    d = loader.load_raw()
    cmp_col = loader.GROUP_COLS["compound"]
    u = (d.groupby([cmp_col, "SMILES"])["Compound Mw (g/mol)"]
         .first().reset_index())
    issues, invalid = [], []
    for _, r in u.iterrows():
        smi = str(r["SMILES"])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append({"compound": r[cmp_col], "SMILES": smi})
            continue
        mw = float(Descriptors.MolWt(mol))
        rep = r["Compound Mw (g/mol)"]
        if pd.isna(rep):
            continue
        rep = float(rep)
        if abs(mw - rep) <= max(1.0, tolerance_pct * rep):
            continue
        # 区分「只是水合物/盐」与「根本是另一个分子」
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        kind = "wrong_molecule"
        if len(frags) > 1:
            main = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            if abs(float(Descriptors.MolWt(main)) - rep) <= max(1.0, tolerance_pct * rep):
                kind = "hydrate_or_salt"
        issues.append({"compound": r[cmp_col], "SMILES": smi,
                       "formula_from_smiles": rdMolDescriptors.CalcMolFormula(mol),
                       "mw_from_smiles": round(mw, 2), "mw_reported": round(rep, 2),
                       "delta": round(mw - rep, 2), "kind": kind})

    wrong = [i for i in issues if i["kind"] == "wrong_molecule"]
    names = [i["compound"] for i in wrong]
    affected = int(d[d[cmp_col].isin(names)].shape[0])

    dup = d.duplicated(subset=[cmp_col, loader.GROUP_COLS["membrane"],
                               "Pressure (kPa)", "pH",
                               "Initial concentration of compound (mg/L)",
                               "Measurement time (min)", loader.TARGET]).sum()
    y = pd.to_numeric(d[loader.TARGET], errors="coerce")
    return {
        "n_unique_compound_smiles_pairs": int(len(u)),
        "n_invalid_smiles": len(invalid),
        "invalid_smiles": invalid[:10],
        "n_mw_mismatch": len(issues),
        "n_wrong_molecule": len(wrong),
        "n_hydrate_or_salt": len(issues) - len(wrong),
        "rows_affected_by_wrong_molecule": affected,
        "pct_rows_affected": round(100 * affected / len(d), 1),
        "wrong_molecule_list": wrong,
        "hydrate_or_salt_list": [i for i in issues if i["kind"] == "hydrate_or_salt"],
        "n_exact_duplicate_rows": int(dup),
        "target_out_of_range": int(((y < 0) | (y > 100)).sum()),
        "verdict": (f"{len(wrong)} 个化合物的 SMILES 与报告 Mw 对不上且不是水合物/盐差异，"
                    f"波及 {affected} 行（{round(100*affected/len(d),1)}%）。"
                    "这些行的 12 个子结构特征全部错误，其残差不可用于机理归因，"
                    "必须先修正 SMILES 或将其排除。"),
        "action": "在发现层做残差归因前，先把 wrong_molecule_list 里的化合物标记为『数据可疑』。",
    }


@tool("ml_list_features",
      "列出当前建模用到的全部特征名与语义分组，供其它工具引用。",
      obj({}), category="model")
def ml_list_features() -> dict:
    b = _bundle()
    return {"n_features": int(b.X.shape[1]), "features": list(b.X.columns),
            "semantic_groups": CFG.get("data.feature_groups"),
            "target": loader.TARGET,
            "registered_descriptors": [
                {"name": r["name"], "status": r["status"]} for r in dstore.listing()]}
