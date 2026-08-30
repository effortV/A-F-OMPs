"""数据底座：dataset.xlsx -> 建模矩阵。

设计决定（写死在这里，任何模块都别再各自造轮子）：
1. 缺失值不插补，原样交给 XGBoost 原生处理；缺失率 >5% 的列额外生成 _isna 指示位。
   —— 插补会凭空造数据、污染残差，而残差是发现层的全部输入。
2. 12 个子结构 SMARTS 计数按既有工作口径计算（'c' 芳香碳计数 /6 折算为苯环数）。
3. 膜身份默认不进模型（只用膜的物性参数），避免模型记住膜编号而非学膜性质。
"""
from __future__ import annotations

import functools
import hashlib
import re
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import CFG

TARGET = CFG.get("data.target")
FEATURES: list[str] = list(CFG.get("data.features"))
SMARTS: list[str] = list(CFG.get("data.smarts"))
SMARTS_SCALE: dict[str, float] = dict(CFG.get("data.smarts_scale") or {})
GROUP_COLS: dict[str, str] = dict(CFG.get("data.group_cols"))


# XGBoost 不接受列名含 [ ] <，故给每个 SMARTS 一个可读的安全别名
SMARTS_LABEL: dict[str, str] = {
    "F": "sub_F",
    "c": "sub_aromRing",            # 芳香碳计数/6 ≈ 苯环数
    "[OH]": "sub_OH",
    "S(=O)(=O)[OH]": "sub_SO3H",
    "C(=O)[OH]": "sub_COOH",
    "[NH2]": "sub_NH2",
    "C(=O)N[#6]": "sub_amide",
    "[#6]O[#6]": "sub_ether",
    "S(=O)(=O)[#6]": "sub_sulfone",
    "C(=O)O[#6]": "sub_ester",
    "Cl": "sub_Cl",
    "[CH3]": "sub_CH3",
}


def smarts_colname(s: str) -> str:
    if s in SMARTS_LABEL:
        return SMARTS_LABEL[s]
    return "sub_" + re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")


SMARTS_COLS = [smarts_colname(s) for s in SMARTS]


@functools.lru_cache(maxsize=1)
def load_raw() -> pd.DataFrame:
    """读取 dataset.xlsx（双行表头，取第二行为列名），只保留有编号的数据行。"""
    df = pd.read_excel(CFG.dataset_path, header=1)
    df = df[df["number"].notna()].reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    # 统一分组列的字符串化
    for key, col in GROUP_COLS.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


@functools.lru_cache(maxsize=1)
def _smarts_table() -> pd.DataFrame:
    """按唯一 SMILES 计算 12 个子结构计数（带磁盘缓存）。"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    df = load_raw()
    smis = sorted(set(df["SMILES"].dropna().astype(str)))
    key = hashlib.md5(("|".join(smis) + "||" + "|".join(SMARTS)).encode()).hexdigest()[:16]
    cache = CFG.cache_dir / f"smarts_{key}.parquet"
    if cache.exists():
        try:
            return pd.read_parquet(cache)
        except Exception:  # noqa: BLE001
            pass

    patterns = [(s, Chem.MolFromSmarts(s)) for s in SMARTS]
    rows = []
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        rec: dict[str, Any] = {"SMILES": smi}
        for s, patt in patterns:
            n = 0.0
            if mol is not None and patt is not None:
                n = float(len(mol.GetSubstructMatches(patt)))
                if s in SMARTS_SCALE and SMARTS_SCALE[s]:
                    n = n / float(SMARTS_SCALE[s])
            rec[smarts_colname(s)] = n
        rows.append(rec)
    out = pd.DataFrame(rows)
    try:
        out.to_parquet(cache, index=False)
    except Exception:  # noqa: BLE001
        pass
    return out


def build_matrix(extra: dict[str, dict[str, float]] | None = None,
                 drop: list[str] | None = None,
                 keep_only: list[str] | None = None,
                 use_membrane_onehot: bool | None = None,
                 with_substructure: bool = True,
                 with_missing_indicator: bool | None = None) -> dict[str, Any]:
    """构造 X / y / groups。

    默认口径：12 个子结构在前 + 20 个特征在后 = 32 列，不加缺失指示位。
    列顺序与既有脚本 np.hstack(arr1, arr3) 一致。

    extra: {描述符名: {SMILES: 值}} —— 发现层新描述符按 SMILES 映射进来。
    drop:  要剔除的特征名（消融用），支持特征组名 __GROUP__:name。
    with_missing_indicator: None 表示读配置 data.use_missing_indicator（默认 false）。
    """
    df = load_raw()
    sub = _smarts_table()
    d = df.merge(sub, on="SMILES", how="left")

    cols: list[str] = [c for c in FEATURES if c in d.columns]
    feat_block = d[cols].apply(pd.to_numeric, errors="coerce").copy()
    sub_block = pd.DataFrame(
        {c: pd.to_numeric(d[c], errors="coerce").fillna(0.0) for c in SMARTS_COLS},
        index=d.index) if with_substructure else pd.DataFrame(index=d.index)

    # 列顺序必须与既有脚本 np.hstack(arr1_子结构, arr3_特征) 一致：
    # XGBoost 的 colsample_bytree 按列位置采样，顺序变了指标就复现不了。
    if with_substructure and bool(CFG.get("data.substructure_first", True)):
        X = pd.concat([sub_block, feat_block], axis=1)
    else:
        X = pd.concat([feat_block, sub_block], axis=1)

    if with_missing_indicator is None:
        with_missing_indicator = bool(CFG.get("data.use_missing_indicator", False))
    if with_missing_indicator:
        thr = float(CFG.get("data.missing_indicator_threshold", 0.05))
        for c in cols:
            if X[c].isna().mean() > thr:
                X[f"{c}__isna"] = X[c].isna().astype(float)

    onehot = CFG.get("model.use_membrane_onehot", False) if use_membrane_onehot is None \
        else use_membrane_onehot
    if onehot:
        oh = pd.get_dummies(d[GROUP_COLS["membrane"]], prefix="MB").astype(float)
        X = pd.concat([X, oh], axis=1)

    added: list[str] = []
    if extra:
        smi = d["SMILES"].astype(str)
        for name, mapping in extra.items():
            X[name] = smi.map(mapping).astype(float)
            added.append(name)

    if keep_only is not None:
        X = X[[c for c in X.columns if c in keep_only]]
    if drop:
        expanded: set[str] = set()
        groups_cfg = CFG.get("data.feature_groups") or {}
        for item in drop:
            if item.startswith("__GROUP__:"):
                gname = item.split(":", 1)[1]
                for f in groups_cfg.get(gname, []):
                    if f == "__SMARTS__":
                        expanded.update(SMARTS_COLS)
                    else:
                        expanded.add(f)
                        expanded.add(f"{f}__isna")
            else:
                expanded.add(item)
                expanded.add(f"{item}__isna")
        X = X[[c for c in X.columns if c not in expanded]]

    y = pd.to_numeric(d[TARGET], errors="coerce")
    ok = y.notna().to_numpy()

    groups = {name: d[col].astype(str).to_numpy() for name, col in GROUP_COLS.items()
              if col in d.columns}
    meta = pd.DataFrame({
        "row_id": d["row_id"],
        "compound": d[GROUP_COLS["compound"]].astype(str),
        "membrane": d[GROUP_COLS["membrane"]].astype(str),
        "membrane_class": d[GROUP_COLS["membrane_class"]].astype(str),
        "reference": d[GROUP_COLS["reference"]].astype(str),
        "SMILES": d["SMILES"].astype(str),
        "CAS": d["CAS"].astype(str),
        "Mw": pd.to_numeric(d["Compound Mw (g/mol)"], errors="coerce"),
    })

    return {"X": X[ok].reset_index(drop=True),
            "y": y[ok].reset_index(drop=True),
            "groups": {k: v[ok] for k, v in groups.items()},
            "meta": meta[ok].reset_index(drop=True),
            "feature_names": list(X.columns),
            "added_descriptors": added,
            "n_dropped_rows": int((~ok).sum())}


def unique_smiles() -> list[str]:
    return sorted(set(load_raw()["SMILES"].dropna().astype(str)))


def compound_smiles_map() -> dict[str, str]:
    d = load_raw()
    return dict(zip(d[GROUP_COLS["compound"]].astype(str), d["SMILES"].astype(str)))


def data_health() -> dict:
    """数据体检：规模、缺失结构、覆盖度、重复。发现层的第一手证据。"""
    d = load_raw()
    n = len(d)
    miss = {c: round(float(d[c].isna().mean()), 4) for c in FEATURES if c in d.columns}
    miss = dict(sorted(miss.items(), key=lambda kv: -kv[1]))
    cmp_col, mb_col = GROUP_COLS["compound"], GROUP_COLS["membrane"]
    pairs = d.groupby([cmp_col, mb_col]).size()
    n_cmp, n_mb = d[cmp_col].nunique(), d[mb_col].nunique()
    per_cmp = d.groupby(cmp_col)[mb_col].nunique()
    complete = int(d[[c for c in FEATURES if c in d.columns] + [TARGET]].dropna().shape[0])
    return {
        "n_rows": n,
        "n_compounds": int(n_cmp),
        "n_smiles": int(d["SMILES"].nunique()),
        "n_membranes": int(n_mb),
        "membrane_class": d[GROUP_COLS["membrane_class"]].value_counts().to_dict(),
        "n_references": int(d[GROUP_COLS["reference"]].nunique()),
        "filled_pairs": int(len(pairs)),
        "possible_pairs": int(n_cmp * n_mb),
        "coverage_pct": round(100 * len(pairs) / (n_cmp * n_mb), 2),
        "max_replicates_per_pair": int(pairs.max()),
        "median_replicates_per_pair": float(pairs.median()),
        "compounds_on_single_membrane": int((per_cmp == 1).sum()),
        "median_membranes_per_compound": float(per_cmp.median()),
        "rows_complete_on_20_features": complete,
        "missing_rate_by_feature": miss,
        "target_stats": {k: round(float(v), 3) for k, v in
                         pd.to_numeric(d[TARGET], errors="coerce").describe().items()},
        "note": ("缺失值不插补，XGBoost 原生处理；缺失率>5% 的列自动加 _isna 指示位。"
                 "若某残差簇与某列缺失模式高度重合，那是数据缺口而非机理。"),
    }


def save_health_report() -> Path:
    rep = data_health()
    p = CFG.logs_dir / "data_health.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
