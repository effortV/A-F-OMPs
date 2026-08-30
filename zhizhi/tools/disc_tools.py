"""格物 GEWU —— 发现层工具集（本项目核心）。

三引擎 + 描述符生成检验闭环。所有"新知识"必须过预注册、负对照、FDR、查重四关。
"""
from __future__ import annotations

import functools
import hashlib
import json
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd

from ..core import db
from ..core.config import CFG
from ..core.tools import (P, obj, report_tool_progress, tool,
                          tool_cancel_requested)
from ..dataio import loader
from ..desc import primitives as prim
from ..desc import store as dstore
from ..ml import model as M
from ..sandbox import runner


def _new_card_id(prefix: str = "C") -> str:
    return f"{prefix}{time.strftime('%m%d')}-{uuid.uuid4().hex[:6]}"


@functools.lru_cache(maxsize=1)
def _qc_suspicious() -> frozenset:
    """SMILES 与报告 Mw 对不上的化合物 —— 其残差不可用于机理归因。"""
    from .ml_tools import ml_data_qc
    try:
        return frozenset(x["compound"] for x in ml_data_qc()["wrong_molecule_list"])
    except Exception:  # noqa: BLE001
        return frozenset()


# ==================== 引擎 1：残差考古 ====================
@tool("disc_residual_clusters",
      "★ 引擎1 残差考古。取分组 OOF 残差最大的样本，在【标准化特征空间 ⊕ Morgan 指纹】上"
      "聚类，返回每个簇的特征画像、代表分子、涉及膜与文献、残差符号，"
      "并自动标注：该簇是否与数据质量问题/缺失模式重合（必须先排除才能归因为机理）。",
      obj({"top_pct": P("number", "取绝对残差前多少比例，默认 0.2"),
           "n_clusters": P("integer", "簇数，默认 6"),
           "group_by": P("string", "CV 分组，默认 compound"),
           "exclude_qc_suspicious": P("boolean", "是否剔除 SMILES 错标的化合物，默认 true"),
           "with_literature": P("boolean", "★ 是否为每个簇自动向文献层取证（默认 true）："
                                "用簇的代表分子+主导特征去检索原文段落、查该因素的历史"
                                "效应方向分布。数据方向与文献方向不一致时会自动标红。"),
           "search_web": P("boolean", "文献层查不到时是否联网补检索，默认 false（慢）")}),
      category="discovery", long_running=True)
def disc_residual_clusters(top_pct: float | None = None, n_clusters: int | None = None,
                           group_by: str = "compound",
                           exclude_qc_suspicious: bool = True,
                           with_literature: bool = True,
                           search_web: bool = False) -> dict:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    RDLogger.DisableLog("rdApp.*")

    top_pct = float(top_pct or CFG.get("discovery.residual_top_pct", 0.2))
    n_clusters = int(n_clusters or CFG.get("discovery.n_clusters", 6))
    b = M.get_bundle(group=group_by)
    f = b.as_frame()
    bad = _qc_suspicious()
    f["qc_suspicious"] = f["compound"].isin(bad)

    pool = f[~f["qc_suspicious"]] if exclude_qc_suspicious else f
    thr = float(np.quantile(pool["abs_residual"], 1 - top_pct))
    sel = pool[pool["abs_residual"] >= thr].copy()
    if len(sel) < n_clusters * 3:
        return {"error": f"高残差样本仅 {len(sel)} 条，不足以聚 {n_clusters} 簇"}

    idx = sel.index.to_numpy()
    A = b.X.iloc[idx].to_numpy(float)
    med = np.nanmedian(b.X.to_numpy(float), axis=0)
    A = np.where(np.isnan(A), med, A)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1
    Z = (A - mu) / sd

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fps = []
    for s in sel["SMILES"]:
        m = Chem.MolFromSmiles(s)
        fps.append(np.array(gen.GetFingerprint(m), dtype=float) if m
                   else np.zeros(1024))
    F = np.vstack(fps)
    F = PCA(n_components=min(12, F.shape[0] - 1, F.shape[1])).fit_transform(F)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)

    ZZ = np.hstack([Z, F])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(ZZ)
    sel["cluster"] = km.labels_

    feat_names = list(b.X.columns)
    base_mean = b.X.mean(numeric_only=True)
    mechanism_features = list(CFG.get("data.mechanism_features", [
        "ΦS", "ΦD", "∆Gs-m (J·m-2)"]))
    out = []
    for c in range(n_clusters):
        g = sel[sel["cluster"] == c]
        if len(g) == 0:
            continue
        gi = g.index.to_numpy()
        sub = b.X.iloc[gi]
        # 与全局均值差异最大的特征（标准化尺度）
        diff = ((sub.mean(numeric_only=True) - base_mean) / (b.X.std(numeric_only=True) + 1e-9))
        diff = diff.dropna().sort_values(key=abs, ascending=False)
        mechanism_prof = [{"feature": k,
                           "cluster_mean": round(float(sub[k].mean()), 4),
                           "global_mean": round(float(base_mean[k]), 4),
                           "z_shift": round(float(diff.get(k, 0.0)), 2)}
                          for k in mechanism_features if k in sub.columns]
        other_diff = diff.drop(labels=[k for k in mechanism_features if k in diff.index])
        prof = [{"feature": k, "cluster_mean": round(float(sub[k].mean()), 4),
                 "global_mean": round(float(base_mean[k]), 4),
                 "z_shift": round(float(v), 2)} for k, v in other_diff.head(8).items()]
        exemplars = (g.reindex(g["abs_residual"].sort_values(ascending=False).index)
                     .drop_duplicates("compound").head(6))
        out.append({
            "cluster": c, "n": int(len(g)),
            "mean_residual": round(float(g["residual"].mean()), 2),
            "sd_residual": round(float(g["residual"].std()), 2),
            "direction": "模型系统性高估（实测截留低于预测）" if g["residual"].mean() < 0
                         else "模型系统性低估（实测截留高于预测）",
            "n_compounds": int(g["compound"].nunique()),
            "n_membranes": int(g["membrane"].nunique()),
            "membrane_class_mix": g["membrane_class"].value_counts().to_dict(),
            "top_membranes": g["membrane"].value_counts().head(4).to_dict(),
            "top_references": {k[:70]: int(v) for k, v in
                               g["reference"].value_counts().head(3).items()},
            "single_reference_dominated": bool(
                g["reference"].value_counts().iloc[0] / len(g) > 0.6),
            "mw_range": [round(float(g["Mw"].min()), 1), round(float(g["Mw"].max()), 1)],
            "mechanism_profile": mechanism_prof,
            "feature_profile": prof,
            "exemplars": exemplars[["compound", "SMILES", "membrane", "y_true",
                                    "y_oof", "residual"]].round(2).to_dict("records"),
        })
    out.sort(key=lambda d: -abs(d["mean_residual"]))

    # ---- 文献层取证：每个簇自动去查原文和历史方向 ----
    if with_literature:
        for c in out:
            c["literature"] = _cluster_literature(c, search_web=search_web)

    return {
        "group_by": group_by, "cv_r2": b.metrics["group_cv"]["r2"],
        "abs_residual_threshold": round(thr, 2), "n_selected": int(len(sel)),
        "qc_excluded_compounds": sorted(bad)[:30] if exclude_qc_suspicious else [],
        "n_qc_excluded_rows": int(f["qc_suspicious"].sum()) if exclude_qc_suspicious else 0,
        "clusters": out,
        "attribution_protocol": [
            "对每个簇必须四选一或组合归因：随机噪声 / 缺条件变量 / 测量协议差异 / 未建模机理。",
            "single_reference_dominated=true 的簇优先归因为『测量协议差异』，不要直接当机理。",
            "归因为『未建模机理』时，必须写出命题："
            "『现有 2D 特征语言无法表达 ___，因为 ___』，并给出可计算的描述符草案。",
            "机理特征固定为 ΦS、ΦD、∆Gs-m (J·m-2)；其它列只能称为模型驱动特征。",
            "每个簇的 literature 字段已经替你查好了文献：passages 是原文段落"
            "（引用必须逐字带上），direction_counts 是该因素的历史效应方向分布，"
            "conflict_with_data=true 表示数据方向和文献方向相反 —— 那是最有价值的信号。",
        ],
    }


def _cluster_literature(cluster: dict, search_web: bool = False) -> dict:
    """给一个残差簇自动做文献取证：原文段落 + 该因素的历史方向分布。"""
    from .lit_tools import lit_claims, lit_expand_search, lit_search
    mols = "、".join(e["compound"] for e in cluster.get("exemplars", [])[:4])
    mechanism_profile = cluster.get("mechanism_profile", [])
    feats = [f["feature"] for f in mechanism_profile]
    mbs = "、".join(list(cluster.get("top_membranes", {}))[:3])
    query = (f"rejection of {mols} by {mbs} membrane; "
             f"effect of {', '.join(feats)} on retention")
    out: dict[str, Any] = {"auto_query": query, "driver_features": feats}
    try:
        out["passages"] = [
            {k: (v[:600] if k == "text" else v) for k, v in p_.items()
             if k in ("title", "year", "page", "section", "text")}
            for p_ in lit_search(query, top_k=4)["passages"]]
    except Exception as e:  # noqa: BLE001
        out["passages"] = []
        out["search_error"] = str(e)[:150]

    # 主导特征的历史方向分布，并与数据方向对照
    lead = max(mechanism_profile, key=lambda x: abs(float(x.get("z_shift", 0)))) \
        if mechanism_profile else None
    probe = _feature_to_descriptor(lead["feature"]) if lead else ""
    if probe:
        try:
            cl = lit_claims(probe, limit=20)
            out["descriptor_probed"] = probe
            out["direction_counts"] = cl["direction_counts"]
            out["claims_sample"] = cl["claims"][:4]
            up, dn = cl["direction_counts"].get("up", 0), cl["direction_counts"].get("down", 0)
            data_dir = "down" if cluster["mean_residual"] < 0 else "up"
            lit_dir = "up" if up > dn else ("down" if dn > up else "mixed")
            out["literature_direction"] = lit_dir
            out["data_direction"] = data_dir
            out["conflict_with_data"] = bool(
                lit_dir != "mixed" and up + dn >= 3 and lit_dir != data_dir)
        except Exception as e:  # noqa: BLE001
            out["claims_error"] = str(e)[:150]

    if search_web and not out.get("passages"):
        try:
            out["web_expansion"] = lit_expand_search(query, max_papers=12)
        except Exception as e:  # noqa: BLE001
            out["web_error"] = str(e)[:150]
    return out


_FEATURE_DESCRIPTOR_MAP = {
    "compound size (nm)": "molecular size", "Compound log K ow": "log Kow",
    "Compound charge": "charge", "MaxPartialCharge": "partial charge",
    "MB zeta potential": "zeta potential", "MB contact angle (°)": "contact angle",
    "pH": "pH", "Pressure (kPa)": "pressure", "WS (mg/L)": "solubility",
    "pKa1 ": "pKa", "pKa2": "pKa", "ΦS": "steric hindrance",
    "ΦD": "dielectric exclusion", "∆Gs-m (J·m-2)": "free energy",
    "Diffusion coefficient (cm2·s-1)": "diffusion",
    "Initial concentration of compound (mg/L)": "feed concentration",
    "Measurement time (min)": "operating time", "E": "excess molar refraction",
    "A": "hydrogen bond acidity", "Density (g·cm-3)": "density",
}


def _feature_to_descriptor(feature: str) -> str:
    """把建模特征名映射成文献里常用的说法，好去查历史主张。"""
    if feature in _FEATURE_DESCRIPTOR_MAP:
        return _FEATURE_DESCRIPTOR_MAP[feature]
    if feature.startswith("sub_"):
        return feature.replace("sub_", "").replace("_", " ")
    return feature.split("(")[0].strip()


# ==================== 引擎 2：图谱覆盖分析 ====================
@tool("disc_coverage_map",
      "★ 引擎2 图谱覆盖分析。289 化合物 × 51 膜的组合矩阵实际只填了约 6%。"
      "对空白格按【预测不确定度 × 外推新颖度 × 假设判别力 × 可得性】打分，"
      "输出高价值空白组合清单，直接可交给 L3 实验设计。",
      obj({"top_n": P("integer", "返回多少个高价值空白，默认 25"),
           "membrane_min_data": P("integer", "只考虑至少有这么多条数据的膜，默认 15"),
           "compound_min_data": P("integer", "只考虑至少有这么多条数据的化合物，默认 3"),
           "prefer_anomalous": P("boolean", "是否优先选残差异常化合物所在的空白，默认 true"),
           "with_literature": P("boolean", "★ 是否对 Top 空白组合查文献（默认 true）："
                                "数据集里空白 ≠ 文献里没人做过。已有报道的会被标出来，"
                                "让你优先做真正的空白。"),
           "check_top_k": P("integer", "查文献的组合数，默认 10")}),
      category="discovery", long_running=True)
def disc_coverage_map(top_n: int = 25, membrane_min_data: int = 15,
                      compound_min_data: int = 3,
                      prefer_anomalous: bool = True,
                      with_literature: bool = True,
                      check_top_k: int = 10) -> dict:
    b = M.get_bundle()
    f = b.as_frame()
    bad = _qc_suspicious()

    mb_counts = f["membrane"].value_counts()
    cp_counts = f["compound"].value_counts()
    membranes = [m for m in mb_counts.index if mb_counts[m] >= membrane_min_data]
    compounds = [c for c in cp_counts.index
                 if cp_counts[c] >= compound_min_data and c not in bad]
    filled = set(zip(f["compound"], f["membrane"]))

    # 每张膜的代表性条件（取该膜出现最多的操作条件行）
    # reindex 到完整列集：全 NaN 的列会被 median() 丢掉，不补会导致拼接时缺列
    cols_all = list(b.X.columns)
    mb_rows: dict[str, pd.Series] = {}
    for m in membranes:
        sub = b.X[f["membrane"].to_numpy() == m]
        mb_rows[m] = sub.median(numeric_only=True).reindex(cols_all)
    cp_rows: dict[str, pd.Series] = {}
    for c in compounds:
        sub = b.X[f["compound"].to_numpy() == c]
        cp_rows[c] = sub.median(numeric_only=True).reindex(cols_all)

    mol_cols = [c for c in b.X.columns
                if c.startswith("sub_") or c in
                ("compound size (nm)", "Compound charge", "Compound log K ow",
                 "Density (g·cm-3)", "Diffusion coefficient (cm2·s-1)", "pKa1 ",
                 "pKa2", "WS (mg/L)", "MaxPartialCharge", "E", "A")
                or c.endswith("__isna")]
    mem_cols = [c for c in b.X.columns if c not in mol_cols]

    rows, keys = [], []
    for c in compounds:
        for m in membranes:
            if (c, m) in filled:
                continue
            r = mb_rows[m].copy()
            for col in mol_cols:
                if col in cp_rows[c]:
                    r[col] = cp_rows[c][col]
            rows.append(r)
            keys.append((c, m))
    if not rows:
        return {"error": "没有满足条件的空白组合，放宽 min_data 阈值"}
    Q = pd.DataFrame(rows)[b.X.columns]

    pred = b.model.predict(Q)
    unc = M.ensemble_std(Q, b)
    dist = M.applicability(Q, b)
    thr = M.domain_threshold(b)

    resid_by_cmp = f.groupby("compound")["residual"].mean()
    anom = {c: abs(float(resid_by_cmp.get(c, 0.0))) for c in compounds}
    amax = max(anom.values()) or 1.0

    recs = []
    for (c, m), p, u, d in zip(keys, pred, unc, dist):
        # 判别力代理：预测落在中间区间（40-90%）时最能区分竞争假设，
        # 因为两端（全截留/全透过）任何机理都给同样答案
        discr = float(np.exp(-((p - 70.0) / 28.0) ** 2))
        nov = float(min(d / max(thr, 1e-6), 2.0))
        anomaly = (anom.get(c, 0.0) / amax) if prefer_anomalous else 0.5
        feas = 1.0 / (1.0 + float(np.log1p(max(d - thr, 0.0))))
        score = (u / (unc.std() + 1e-9)) * 0.35 + discr * 0.25 + nov * 0.2 \
            + anomaly * 0.15 + feas * 0.05
        recs.append({"compound": c, "membrane": m,
                     "predicted_removal_pct": round(float(p), 1),
                     "prediction_uncertainty": round(float(u), 2),
                     "domain_distance": round(float(d), 2),
                     "in_domain": bool(d <= thr),
                     "discriminating_power": round(discr, 3),
                     "compound_mean_residual": round(float(resid_by_cmp.get(c, 0.0)), 2),
                     "value_score": round(float(score), 3)})
    recs.sort(key=lambda r: -r["value_score"])

    # ---- 文献层交叉核对：数据集里空白 ≠ 文献里没人做过 ----
    n_studied = 0
    if with_literature and recs:
        from .lit_tools import lit_search
        for r in recs[:check_top_k]:
            try:
                hits = lit_search(f"rejection of {r['compound']} by {r['membrane']} "
                                  f"nanofiltration reverse osmosis membrane",
                                  top_k=3)["passages"]
                strong = [h for h in hits
                          if r["compound"].lower()[:12] in (h.get("text") or "").lower()]
                r["literature"] = {
                    "n_hits": len(hits),
                    "likely_already_studied": bool(strong),
                    "top_hit": (f"{hits[0]['title'][:70]} ({hits[0]['year']}) "
                                f"p.{hits[0]['page']}" if hits else None),
                    "evidence": [{"title": h["title"][:70], "year": h["year"],
                                  "page": h["page"], "text": (h.get("text") or "")[:280]}
                                 for h in strong[:2]]}
                n_studied += bool(strong)
            except Exception as e:  # noqa: BLE001
                r["literature"] = {"error": str(e)[:120]}

    return {
        "matrix": {"n_compounds_considered": len(compounds),
                   "n_membranes_considered": len(membranes),
                   "possible_cells": len(compounds) * len(membranes),
                   "filled_cells": int(sum(1 for c in compounds for m in membranes
                                           if (c, m) in filled)),
                   "blank_cells": len(recs),
                   "global_coverage_pct": loader.data_health()["coverage_pct"]},
        "scoring": ("value = 0.35·预测不确定度 + 0.25·判别力(预测落在40-90%中间带) "
                    "+ 0.20·外推新颖度 + 0.15·该化合物残差异常度 + 0.05·可行性"),
        "high_value_blanks": recs[:top_n],
        "literature_crosscheck": {
            "checked": min(check_top_k, len(recs)) if with_literature else 0,
            "likely_already_studied": n_studied,
            "read": ("likely_already_studied=true 的组合虽然不在 dataset.xlsx 里，"
                     "但文献中很可能已有报道 —— 优先做文献也查不到的那些，"
                     "或者先把已有文献的数据补录进来。")},
        "caveat": ("空白格的特征向量是用『该膜条件中位数 + 该化合物分子特征』拼出来的，"
                   "是合成样本；in_domain=false 的项预测不可信，"
                   "但恰恰是信息量最大的实验点。"),
    }


# ==================== 引擎 3：跨学科迁移 ====================
CROSS_SYS = """你是跨学科概念迁移专家。任务：从给定外领域中提炼它解释"选择性"的核心概念，
映射到 NF/RO 膜对有机微污染物的截留问题上，产出**可证伪**的假设。

硬性要求（缺任何一项该次扫描判为无效）：
1. donor_concept：该领域的核心概念，一句话说清它在原领域解释什么；
2. mapping：概念如何映射到膜截留（对应关系要具体到物理量，不能是比喻）；
3. falsifiable_prediction：写成"若假设成立，则在 ___ 条件下应观察到 ___；
   若观察到 ___ 则假设被证伪"；
4. computable_descriptor：给出**可以用 RDKit + 构象系综算出来的**描述符定义，
   包含名字、计算步骤、预期符号（升高时截留升高还是降低）；
5. discriminating_test：一个能把该假设与"现有尺寸排阻+疏水分配"标准解释分开的检验，
   要说明两种解释在什么条件下给出**相反**的预测；
6. why_not_already_known：为什么现有 20 个特征表达不了这个量。

已知可用的构象原语（你的描述符必须能用它们实现）：
%s

输出 JSON：
{"donor_domain": str, "donor_concept": str, "mapping": str,
 "falsifiable_prediction": str,
 "computable_descriptor": {"name": str, "definition": str, "steps": [str],
                           "expected_sign": "up|down", "primitives_used": [str]},
 "discriminating_test": str, "why_not_already_known": str,
 "risk": str, "confidence": 0-1}"""


PROPOSE_SYS = """你在为一个膜分离知识发现系统挑选**外领域**做概念迁移。

不要局限于任何预设清单。你的任务是：看当前数据里的异常和文献里的矛盾，
反推「哪个外领域已经解决过结构类似的问题」，提出 3-5 个候选供体领域。

挑选原则（按重要性排序）：
1. **机制可类比**：该领域也在研究"某种介质如何区别对待不同分子"，
   哪怕物理载体完全不同（生物膜、色谱柱、催化剂孔、土壤、皮肤角质层、血脑屏障…）。
2. **离膜科学越远越好**：近领域（其它膜工艺、水处理单元）容易产出已知复现。
   优先跨到化学之外：物理、生物、材料、地学、甚至工程学的其它分支。
3. **该领域有成熟的定量描述符**：能落到 RDKit / 构象系综算得出来的量，
   否则迁移过来无法检验。
4. **能解释给定的异常**：明确说清它可能解释哪一条残差异常或文献矛盾。

已经扫过的领域（避免重复，除非你能给出全新角度）：
%s

输出 JSON：
{"candidates": [
   {"domain": str,
    "why_this_domain": str,
    "which_anomaly_it_might_explain": str,
    "core_selectivity_concept": str,
    "expected_descriptor_family": str,
    "distance_from_membrane_science": "near|medium|far",
    "promise": 0-1}],
 "recommended": str,
 "reasoning": str}"""


def _normalize_domain_proposal(raw: Any) -> dict:
    """Accept the harmless JSON shape variations produced by different models.

    The documented shape is ``{"candidates": [...]}``, but providers may return
    the candidate array directly even in JSON mode.  Normalize those variants at
    the tool boundary so callers always receive one stable dictionary schema.
    """
    payload = dict(raw) if isinstance(raw, dict) else {}
    if isinstance(raw, list):
        candidates: Any = raw
    else:
        candidates = (payload.get("candidates") or payload.get("candidate_domains")
                      or payload.get("domains"))
        if candidates is None and payload.get("domain"):
            candidates = [payload]

    if isinstance(candidates, dict):
        if candidates.get("domain"):
            candidates = [candidates]
        else:
            candidates = list(candidates.values())
    if not isinstance(candidates, list):
        candidates = []

    normalized = []
    for item in candidates:
        if isinstance(item, str) and item.strip():
            normalized.append({"domain": item.strip()})
        elif isinstance(item, dict) and str(item.get("domain") or "").strip():
            normalized.append(dict(item))
    if not normalized:
        raise ValueError("候选领域 JSON 中没有可用的 domain")

    recommended = payload.get("recommended")
    if isinstance(recommended, dict):
        recommended = recommended.get("domain")
    if not isinstance(recommended, str) or not recommended.strip():
        def promise(candidate: dict) -> float:
            try:
                return float(candidate.get("promise") or 0)
            except (TypeError, ValueError):
                return 0.0
        recommended = max(normalized, key=promise)["domain"]

    payload["candidates"] = normalized
    payload["recommended"] = recommended.strip()
    payload.setdefault("reasoning", "")
    return payload


@tool("disc_propose_domains",
      "★ 让 Agent 自己提出值得跨界的外领域（不限于预设轮转池）。"
      "会结合当前残差异常与文献矛盾反推「哪个领域已经解决过结构类似的问题」，"
      "并优先推荐离膜科学远、但机制可类比、且有成熟定量描述符的领域。",
      obj({"context": P("string", "当前的残差发现/矛盾摘要；留空会自动从数据库现取")}),
      category="discovery")
def disc_propose_domains(context: str = "", max_seconds: float | None = None) -> dict:
    from ..core.llm import LLM
    scanned = _scanned_domains()
    if not context:
        context = _auto_context()
    system = PROPOSE_SYS % (json.dumps(sorted(scanned), ensure_ascii=False)
                            or "（还没扫过）")
    user = f"当前数据异常与文献矛盾：\n{context[:8000]}"
    if max_seconds is None:
        # 用户单独点击“先看它想扫哪些领域”：恢复原来的完整生成方式。
        j = LLM("gewu").ask_json(system, user, temperature=0.75)
    else:
        # 深扫内部的自动提域只受总耗时预算约束，不限制输出 token。
        j = LLM("gewu", model=CFG.llm_model, fallbacks=[CFG.llm_model],
                usage_kind="crossdomain_domain_proposal").ask_json(
            system, user, temperature=0.75, thinking=False,
            request_timeout=max(10.0, float(max_seconds)), attempts=1)
    j = _normalize_domain_proposal(j)
    j["already_scanned"] = sorted(scanned)
    j["next"] = "挑一个 domain 传给 disc_crossdomain_scan 做深扫"
    return j


def _scanned_domains() -> set[str]:
    out = set()
    for r in db.q("SELECT payload FROM cards WHERE engine='E3_crossdomain'"):
        try:
            d = db.jdict(r["payload"]).get("donor_domain")
            if d:
                out.add(d)
        except Exception:  # noqa: BLE001
            continue
    for r in db.q("SELECT content FROM memory WHERE kind='crossdomain_scanned'"):
        out.add(r["content"])
    return out


def _auto_context() -> str:
    """从库里现取残差异常 + 文献矛盾，作为提域和扫描的上下文。"""
    parts = []
    try:
        rc = disc_residual_clusters(n_clusters=4)
        for c in rc.get("clusters", [])[:3]:
            ex = "、".join(e["compound"] for e in c["exemplars"][:3])
            parts.append(f"[残差簇{c['cluster']}] n={c['n']} 平均残差{c['mean_residual']:+.1f} "
                         f"({c['direction']})，代表分子：{ex}；"
                         f"特征偏离：{', '.join(f['feature'] for f in c['feature_profile'][:3])}")
    except Exception:  # noqa: BLE001
        pass
    for r in db.q("SELECT descriptor, side_a, side_b FROM contradictions LIMIT 5"):
        parts.append(f"[矛盾] {r['descriptor']}：A={str(r['side_a'])[:90]} / "
                     f"B={str(r['side_b'])[:90]}")
    parts.append("现有 20 个特征里没有任何真 3D 量，分子被当作刚性球处理。")
    return "\n".join(parts)


@tool("disc_crossdomain_scan",
      "★ 引擎3 跨学科扫描。三种模式：auto_propose=Agent 先自主提候选领域再深扫（默认，"
      "不限于预设池）；rotate=从轮转池取一个没扫过的；manual=你指定任意领域（自由文本）。"
      "必须产出可证伪预测 + 可计算描述符 + 判别性检验，否则判为无效扫描。"
      "联网查重只读取文献元数据，不同步下载 PDF；完整文献扩充可另行后台排队。",
      obj({"domain": P("string", "手动指定的外领域（任意文本，如「晶体工程」「血脑屏障渗透」）；"
                       "填了就等于 manual 模式"),
           "mode": P("string", "auto_propose | rotate | manual",
                     enum=["auto_propose", "rotate", "manual"]),
           "context": P("string", "当前残差发现/矛盾；留空自动从数据库现取"),
           "auto_novelty_check": P("boolean", "★ 扫完自动查重（本地语料 + OpenAlex），"
                                   "默认 true。查到已有报道会降级为 rediscovery。"),
           "search_web": P("boolean", "查重时是否联网读取 OpenAlex 元数据，默认 true；"
                           "不会下载 PDF，也不会阻塞等待文献学习"),
           "expand_literature": P("boolean", "是否把完整文献扩充转入独立后台任务，"
                                  "默认 false；开启后立即返回任务编号")}),
      category="discovery", long_running=True)
def disc_crossdomain_scan(domain: str = "", mode: str = "", context: str = "",
                          auto_novelty_check: bool = True,
                          search_web: bool = True,
                          expand_literature: bool = False) -> dict:
    from ..core.llm import LLM
    started = time.monotonic()
    max_seconds = max(60.0, float(
        CFG.get("discovery.cross_domain_max_seconds", 1200)))

    def remaining() -> float:
        return max(0.0, max_seconds - (time.monotonic() - started))

    pool = list(CFG.get("discovery.cross_domain_pool") or [])
    scanned = _scanned_domains()
    domain = str(domain or "").strip()
    mode = str(mode or ("manual" if domain else
                        CFG.get("discovery.cross_domain_mode", "auto_propose")))
    proposal = None

    def result(status: str, stage: str, payload: dict | None = None,
               error: str = "", missing_fields: list[str] | None = None,
               partial: bool | None = None) -> dict:
        """Return one stable envelope for success, invalid, timeout and errors."""
        out = dict(payload or {})
        out.update({
            "status": status,
            "scan_valid": status == "success",
            "missing_fields": list(missing_fields or []),
            "domain": domain or str(out.get("donor_domain") or ""),
            "mode": mode,
            "stage": stage,
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "partial": (status != "success" if partial is None else partial),
        })
        if error:
            out["error"] = error
        if proposal:
            out["domain_proposal"] = proposal
        return out

    def cancelled(stage: str) -> dict | None:
        if tool_cancel_requested():
            return result("cancelled", stage, {"cancelled": True})
        return None

    report_tool_progress("整理残差、文献矛盾与已扫描领域", 0.04)
    if not context:
        context = _auto_context()
    stopped = cancelled("context")
    if stopped:
        return stopped

    if mode == "auto_propose" and not domain:
        report_tool_progress("V4-Pro 正在选择最有信息量的外领域", 0.16)
        try:
            proposal = disc_propose_domains(
                context, max_seconds=min(
                    float(CFG.get("discovery.domain_proposal_timeout", 55)),
                    max(10.0, remaining() - 20.0)))
        except Exception as exc:  # noqa: BLE001
            return result(
                "timeout" if "timeout" in str(exc).lower() else "error",
                "domain_proposal",
                error=f"自主提域失败: {type(exc).__name__}: {exc}")
        domain = (proposal.get("recommended")
                  or (proposal.get("candidates") or [{}])[0].get("domain") or "")
    stopped = cancelled("domain_selected")
    if stopped:
        return stopped
    if not domain:
        remaining_domains = [d for d in pool if d not in scanned]
        if not (remaining_domains or pool):
            return result("error", "domain_selection",
                          error="未获得可扫描的外领域")
        domain = (remaining_domains or pool)[0]
        mode = "rotate"

    report_tool_progress(f"V4-Pro 正在深扫：{domain}", 0.34)
    sys_p = CROSS_SYS % json.dumps(prim.AVAILABLE, ensure_ascii=False, indent=1)
    user = f"外领域：{domain}\n\n目标问题：NF/RO 膜对有机微污染物的截留选择性。"
    if context:
        user += f"\n\n当前已发现的数据异常/矛盾（迁移应尽量解释它们）：\n{context[:6000]}"
    try:
        j = LLM("gewu", model=CFG.llm_model, fallbacks=[CFG.llm_model],
                usage_kind="crossdomain_scan").ask_json(
            sys_p, user, temperature=0.6, thinking=True,
            request_timeout=min(
                float(CFG.get("discovery.cross_domain_llm_timeout", 1000)),
                max(10.0, remaining() - 12.0)),
            attempts=1)
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        is_timeout = ("timeout" in text.lower() or "timed out" in text.lower()
                      or isinstance(exc, TimeoutError))
        return result(
            "timeout" if is_timeout else "error", "crossdomain_reasoning",
            error=f"跨学科主推理失败: {type(exc).__name__}: {exc}")
    stopped = cancelled("crossdomain_reasoning")
    if stopped:
        return stopped

    if isinstance(j, list):
        j = next((dict(item) for item in j if isinstance(item, dict)), {})
    elif isinstance(j, dict):
        for wrapper in ("result", "scan", "hypothesis"):
            nested = j.get(wrapper)
            if isinstance(nested, dict) and not j.get("donor_concept"):
                j = dict(nested)
                break
    else:
        j = {}

    required = ["donor_concept", "mapping", "falsifiable_prediction",
                "computable_descriptor", "discriminating_test",
                "why_not_already_known"]
    missing = [k for k in required if not j.get(k)]
    j["already_scanned_domains"] = sorted(scanned)
    if missing:
        return result("invalid", "schema_validation", j,
                      error="模型已完成推理，但输出缺少必需字段",
                      missing_fields=missing)
    j = result("success", "crossdomain_reasoning", j, partial=False)

    # ---- 文献层：快速查重与完整文献扩充彻底解耦 ----
    concept = (j.get("donor_concept") or "")[:300]
    pred = (j.get("falsifiable_prediction") or "")[:400]
    novelty_statement = f"{concept}。应用于 NF/RO 膜截留：{pred}"
    if auto_novelty_check and concept:
        from .lit_tools import lit_novelty_check
        try:
            report_tool_progress("查重：只读取本地语料与 OpenAlex 元数据", 0.60)
            j["novelty_check"] = lit_novelty_check(
                novelty_statement, search_web=search_web,
                max_seconds=max(8.0, remaining() - 5.0))
            j["novelty"] = j["novelty_check"].get("verdict")
            j["novelty_check_completed"] = bool(
                j["novelty_check"].get("completed")
                or j["novelty_check"].get("cached"))
        except Exception as e:  # noqa: BLE001
            j["novelty_check"] = {"error": str(e)[:180]}
            j["novelty_check_completed"] = False
    stopped = cancelled("novelty_check")
    if stopped:
        return {**j, **stopped}

    if expand_literature and concept:
        from .lit_tools import queue_literature_expansion
        try:
            report_tool_progress("完整文献扩充正在转入独立后台任务", 0.88)
            j["literature_expansion_task"] = queue_literature_expansion(
                f"{domain} {concept} membrane separation selectivity",
                max_papers=15)
        except Exception as e:  # noqa: BLE001
            j["literature_expansion_task"] = {"error": str(e)[:180]}

    # 记下来，下次自主提域会避开
    report_tool_progress("保存扫描记录；当前结果已可用于预注册和落卡", 0.96)
    db.ex("INSERT INTO memory(agent,kind,content,created_at) VALUES('gewu',"
          "'crossdomain_scanned',?,strftime('%s','now'))", (domain,))
    j.update(result("success", "complete", j, partial=False))
    if j.get("novelty_check_completed"):
        j["next_step"] = ("查重已完成，禁止重复调用 lit_novelty_check；直接 disc_prereg 预注册 -> "
                          "disc_compute_descriptor 实现 -> ml_add_descriptor 检验")
    else:
        j["next_step"] = ("查重未完成；确认查重结果后 disc_prereg 预注册 -> "
                          "disc_compute_descriptor 实现 -> ml_add_descriptor 检验")
    return j


@tool("disc_list_primitives",
      "列出 3D 构象描述符原语库（写描述符代码时可直接用 prim.xxx 调用）。",
      obj({}), category="discovery")
def disc_list_primitives() -> dict:
    return {"module": "prim（沙箱内已自动注入）", "functions": prim.AVAILABLE,
            "usage_example": (
                "def compute(smiles):\n"
                "    cs = prim.min_cross_section(smiles)\n"
                "    hb = prim.intramolecular_hbonds(smiles)\n"
                "    if cs is None or hb is None: return None\n"
                "    return cs['spread_nm'] * (1 + hb['mean'])"),
            "contract": "compute(smiles) 必须返回 float 或 None；不得 import 白名单外的模块。",
            "note": ("现有 20 个特征里没有任何真 3D 量（min/max projection 只是 2D 投影），"
                     "这是描述符语言最大的空白。")}


# ==================== 描述符闭环 ====================
@tool("disc_prereg",
      "★ 预注册。在**看到检验结果之前**把假设、描述符定义、检验协议、"
      "成功阈值、证伪条件写死并做哈希存档。事后修改判定标准会被记录并标红。"
      "这是这套系统能不能算科学的分水岭 —— 任何描述符检验前必须先做这一步。",
      obj({"descriptor_name": P("string", "描述符名（英文、下划线、可作列名）"),
           "hypothesis": P("string", "假设陈述"),
           "engine": P("string", "来源引擎 E1_residual|E2_coverage|E3_contradiction|E3_crossdomain"),
           "expected_sign": P("string", "预期方向 up|down"),
           "target_subgroup": P("string", "它本来要解释的残差子群 pandas query"),
           "success_threshold": P("object", "成功阈值，如 {\"delta_r2\": 0.005, "
                                  "\"targeted_abs_residual_drop_pct\": 10}"),
           "falsification_condition": P("string", "什么结果算证伪")},
          ["descriptor_name", "hypothesis", "engine", "expected_sign",
           "falsification_condition"]), category="discovery")
def disc_prereg(descriptor_name: str, hypothesis: str, engine: str,
                expected_sign: str, falsification_condition: str,
                target_subgroup: str = "", success_threshold: dict | None = None) -> dict:
    if db.q1("SELECT name FROM descriptors WHERE name=?", (descriptor_name,)):
        return {"error": f"描述符 {descriptor_name} 已存在，换个名字或用 disc_card_get 查看"}
    proto = {"descriptor_name": descriptor_name, "hypothesis": hypothesis,
             "engine": engine, "expected_sign": expected_sign,
             "target_subgroup": target_subgroup,
             "success_threshold": success_threshold or
             {"delta_r2": CFG.get("discovery.pass_delta_r2", 0.005),
              "delta_r2_ci_lower_gt": 0.0, "p_yscramble_lt": 0.05,
              "max_abs_corr_with_existing_lt": CFG.get("discovery.redundancy_r_threshold", 0.9)},
             "falsification_condition": falsification_condition,
             "registered_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    blob = json.dumps(proto, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha256(blob.encode()).hexdigest()[:32]
    cid = _new_card_id("D")
    db.ex("INSERT INTO cards(id,kind,engine,title,statement,payload,prereg,prereg_hash,"
          "status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,'proposed',?,?)",
          (cid, "descriptor", engine, f"描述符假设：{descriptor_name}", hypothesis,
           json.dumps({"descriptor_name": descriptor_name}, ensure_ascii=False),
           blob, h, time.time(), time.time()))
    dstore.register(descriptor_name, hypothesis, "", proto, card_id=cid, status="proposed")
    p = CFG.cards_dir / f"prereg_{descriptor_name}.json"
    p.write_text(json.dumps({"hash": h, **proto}, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return {"card_id": cid, "prereg_hash": h, "file": str(p), "protocol": proto,
            "next": "disc_compute_descriptor 实现代码 -> ml_add_descriptor 检验"}


@tool("disc_compute_descriptor",
      "★ 在沙箱里执行你写的描述符代码，对全部 299 个 SMILES 求值并入库。"
      "返回有效率、方差、与既有 20 特征的最大相关系数（>0.9 直接判为旧信息换皮）。",
      obj({"name": P("string", "描述符名（须先 disc_prereg 预注册）"),
           "code": P("string", "Python 代码，必须定义 compute(smiles) -> float|None；"
                     "可直接用 prim.* 原语；只能 import numpy/math/rdkit"),
           "timeout_s": P("integer", "超时秒数，默认 1200")},
          ["name", "code"]), category="discovery", long_running=True)
def disc_compute_descriptor(name: str, code: str, timeout_s: int = 1200) -> dict:
    reg = db.q1("SELECT * FROM descriptors WHERE name=?", (name,))
    if not reg:
        return {"error": f"{name} 未预注册。先调用 disc_prereg。"}
    smis = loader.unique_smiles()
    res = runner.run_descriptor(code, smis, timeout=timeout_s)
    if not res.get("ok"):
        return res

    vals = {k: v for k, v in res["values"].items() if v is not None}
    stats = res["stats"]
    if stats["n_valid"] < 30:
        dstore.set_status(name, "failed", {"reason": "有效值太少", **stats})
        return {**res, "verdict": "FAIL", "reason": "有效值不足 30 个，描述符不可用"}
    if stats.get("n_unique", 0) < 5:
        dstore.set_status(name, "failed", {"reason": "几乎为常数", **stats})
        return {**res, "verdict": "FAIL", "reason": "取值几乎是常数，无区分度"}

    dstore.save_values(name, vals)
    spec = db.jdict(reg["spec"])
    dstore.register(name, reg["hypothesis"], code, spec,
                    card_id=reg["card_id"], status="computed")

    # 冗余度预检
    built = loader.build_matrix(extra={name: vals})
    X = built["X"]
    v = X[name].to_numpy(float)
    ok = ~np.isnan(v)
    corrs = {}
    for c in X.columns:
        if c == name:
            continue
        u = X[c].to_numpy(float)
        m = ok & ~np.isnan(u)
        if m.sum() > 30 and np.std(u[m]) > 0 and np.std(v[m]) > 0:
            corrs[c] = float(np.corrcoef(u[m], v[m])[0, 1])
    top = sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:5]
    max_r = abs(top[0][1]) if top else 0.0
    redundant = max_r >= float(CFG.get("discovery.redundancy_r_threshold", 0.9))
    if redundant:
        dstore.set_status(name, "redundant", {"max_abs_corr": max_r})
    return {"name": name, "stats": stats, "n_errors": res["n_errors"],
            "error_sample": res.get("error_sample"),
            "max_abs_corr_with_existing": round(max_r, 3),
            "most_correlated": [{"feature": k, "r": round(x, 3)} for k, x in top],
            "verdict": "REDUNDANT" if redundant else "READY",
            "next": ("与既有特征高度相关，这是旧信息换皮，不要继续检验。" if redundant else
                     f"调用 ml_add_descriptor(name='{name}', target_subgroup=预注册里的子群) 做正式检验。")}


# ==================== 卡片 ====================
@tool("disc_create_card",
      "创建发现卡片。任何要上报给人的发现都必须落成卡片，"
      "并且 novelty 字段必须来自 lit_novelty_check 的判定，不能自己拍。",
      obj({"kind": P("string", "discovery|descriptor|contradiction|blank_spot|crossdomain"),
           "engine": P("string", "E1_residual|E2_coverage|E3_contradiction|E3_crossdomain"),
           "title": P("string", "简短标题"),
           "statement": P("string", "命题陈述（要可证伪）"),
           "novelty": P("string", "rediscovery|in_field_new|cross_domain_new|novel"),
           "payload": P("object", "证据与细节 JSON：**必须同时含数据侧量化结果和文献原文引语**"),
           "force": P("boolean", "跳过证据校验（仅在确知无文献可查时用），默认 false")},
          ["kind", "engine", "title", "statement", "novelty"]), category="discovery")
def disc_create_card(kind: str, engine: str, title: str, statement: str,
                     novelty: str, payload: dict | None = None,
                     force: bool = False) -> dict:
    # 机理类卡片必须同时带数据侧证据和文献原文引语，缺一边直接拒
    chk = disc_evidence_check(payload or {}, kind)
    if not chk["ok"] and not force:
        return {"error": "证据不足，卡片未创建", "check": chk,
                "how_to_fix": ("三个引擎的返回值里已经带了文献证据："
                               "引擎1 每个簇的 literature.passages、"
                               "引擎2 每个空白的 literature.evidence、"
                               "引擎3 的 novelty_check。把原文引语和数据侧量化结果"
                               "一起写进 payload；确实没有文献证据时"
                               "（如引擎2 的覆盖空白）请把 kind 设为 blank_spot。")}
    cid = _new_card_id("C")
    db.ex("INSERT INTO cards(id,kind,engine,title,statement,novelty,payload,status,"
          "created_at,updated_at) VALUES(?,?,?,?,?,?,?,'proposed',?,?)",
          (cid, kind, engine, title, statement, novelty,
           json.dumps(payload or {}, ensure_ascii=False), time.time(), time.time()))
    (CFG.cards_dir / f"{cid}.json").write_text(
        json.dumps({"id": cid, "kind": kind, "engine": engine, "title": title,
                    "statement": statement, "novelty": novelty,
                    "payload": payload or {}}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return {"card_id": cid, "status": "proposed",
            "note": "卡片已进入人工审阅队列。下一步可交给验真做 L1/L2/L3 验证设计。"}


@tool("disc_update_card",
      "更新卡片：写入 L1 结果 / L2 方案 / L3 方案 / 状态 / 补充证据。",
      obj({"card_id": P("string", "卡片 id"),
           "status": P("string", "proposed|tested|passed|refuted|parked"),
           "l1_result": P("object", "L1 验证结果"),
           "l2_plan": P("string", "L2 MD 方案"),
           "l3_plan": P("string", "L3 实验方案"),
           "payload_merge": P("object", "要并入 payload 的补充内容")},
          ["card_id"]), category="discovery")
def disc_update_card(card_id: str, status: str = "", l1_result: dict | None = None,
                     l2_plan: str = "", l3_plan: str = "",
                     payload_merge: dict | None = None) -> dict:
    r = db.q1("SELECT * FROM cards WHERE id=?", (card_id,))
    if not r:
        return {"error": "卡片不存在"}
    sets, args = ["updated_at=?"], [time.time()]
    if status:
        sets.append("status=?")
        args.append(status)
    if l1_result is not None:
        # 只接受 dict。曾经有人传空字符串进来，落库成 '""'，
        # 读回时 json.loads 得到 str，后面 .get() 直接崩。
        if not isinstance(l1_result, dict):
            return {"error": "l1_result 必须是 JSON 对象（dict），"
                             f"收到 {type(l1_result).__name__}"}
        sets.append("l1_result=?")
        args.append(json.dumps(l1_result, ensure_ascii=False))
    if l2_plan:
        sets.append("l2_plan=?")
        args.append(l2_plan)
    if l3_plan:
        sets.append("l3_plan=?")
        args.append(l3_plan)
    if payload_merge:
        cur = db.jdict(r["payload"])
        cur.update(payload_merge)
        sets.append("payload=?")
        args.append(json.dumps(cur, ensure_ascii=False))
    args.append(card_id)
    db.ex(f"UPDATE cards SET {','.join(sets)} WHERE id=?", args)
    return {"card_id": card_id, "updated": True}


@tool("disc_list_cards",
      "列出发现卡片，可按状态/引擎/新颖性过滤。",
      obj({"status": P("string", "过滤状态"), "engine": P("string", "过滤引擎"),
           "limit": P("integer", "默认 30")}), category="discovery")
def disc_list_cards(status: str = "", engine: str = "", limit: int = 30) -> dict:
    sql, args = "SELECT * FROM cards", []
    w = []
    if status:
        w.append("status=?")
        args.append(status)
    if engine:
        w.append("engine=?")
        args.append(engine)
    if w:
        sql += " WHERE " + " AND ".join(w)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    rows = db.rows_to_dicts(db.q(sql, args))
    out = []
    for r in rows:
        out.append({"id": r["id"], "kind": r["kind"], "engine": r["engine"],
                    "title": r["title"], "novelty": r["novelty"],
                    "status": r["status"], "has_prereg": bool(r["prereg_hash"]),
                    "has_l1": bool(r["l1_result"]), "has_l2": bool(r["l2_plan"]),
                    "has_l3": bool(r["l3_plan"]),
                    "review": r["review"]})
    counts = {r["status"]: r["c"] for r in
              db.q("SELECT status, COUNT(*) c FROM cards GROUP BY status")}
    return {"n": len(out), "status_counts": counts, "cards": out}


@tool("disc_card_get", "取单张卡片全文（含预注册协议、L1/L2/L3）。",
      obj({"card_id": P("string", "卡片 id")}, ["card_id"]), category="discovery")
def disc_card_get(card_id: str) -> dict:
    r = db.q1("SELECT * FROM cards WHERE id=?", (card_id,))
    if not r:
        return {"error": "不存在"}
    d = dict(r)
    for k in ("payload", "prereg", "l1_result"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except Exception:  # noqa: BLE001
                pass
    return d


# ==================== 建卡证据校验 ====================
@tool("disc_evidence_check",
      "建卡前自查：检查一份 payload 是否同时具备数据侧与文献侧证据。"
      "disc_create_card 内部也会调它，机理类卡片缺文献引语会被拒。",
      obj({"payload": P("object", "准备写进卡片的证据 JSON"),
           "kind": P("string", "卡片类型")}, ["payload"]), category="discovery")
def disc_evidence_check(payload: dict, kind: str = "discovery") -> dict:
    blob = json.dumps(payload or {}, ensure_ascii=False)
    has_data = any(k in blob for k in
                   ("residual", "delta_r2", "r2", "shap", "ablation", "cv_",
                    "coverage", "prediction_uncertainty"))
    has_lit = any(k in blob for k in ("quote", "引语", "paper", "doi", "p.", "文献"))
    need_lit = kind in ("discovery", "descriptor", "contradiction", "crossdomain")
    ok = has_data and (has_lit or not need_lit)
    return {"ok": ok, "has_data_evidence": has_data, "has_literature_evidence": has_lit,
            "literature_required_for_this_kind": need_lit,
            "hint": ("通过" if ok else
                     "机理类卡片必须同时含数据侧证据和文献原文引语。"
                     "三个引擎的返回值里已经带了文献段落，把引语写进 payload 即可。"
                     if need_lit and not has_lit else
                     "缺数据侧证据：payload 里要有残差/ΔR²/SHAP 之类的量化结果。")}
