"""3D 构象描述符原语库。

现有 20 个特征里**一个真正的 3D 量都没有**（min/max projection 是 2D 投影），
这是描述符语言里最大的空白。本模块提供构象系综层面的可计算原语，
供发现层自己写的描述符代码直接调用（沙箱里以 `prim` 名字注入）。

所有函数以 SMILES 为入口，构象系综带磁盘缓存。
"""
from __future__ import annotations

import hashlib
import math
import pickle
from functools import lru_cache
from typing import Any

import numpy as np

from ..core.config import CFG

_CACHE_DIR = CFG.cache_dir / "conformers"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VDW = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 16: 1.80,
       17: 1.75, 35: 1.85, 53: 1.98, 11: 2.27, 19: 2.75}
POLAR = {7, 8, 9, 15, 16}
RT = 0.001987 * 298.15          # kcal/mol


def _mol_key(smiles: str, n_conf: int) -> str:
    return hashlib.md5(f"{smiles}|{n_conf}".encode()).hexdigest()[:16]


@lru_cache(maxsize=512)
def ensemble(smiles: str, n_conf: int | None = None) -> dict[str, Any] | None:
    """生成 ETKDGv3 构象系综 + MMFF94s 优化，返回坐标、能量、元素、半径。

    返回 None 表示分子无法处理（发现层的描述符必须能容忍 None）。
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    from rdkit.Chem import rdMolDescriptors
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        return None
    heavy = mol0.GetNumHeavyAtoms()
    if n_conf is None:
        # 按柔性自适应：刚性分子多采样是浪费，柔性分子少采样会漏掉折叠态。
        # 再按分子尺寸压一层顶：ETKDG 在大环/大分子上极慢（红霉素类 57 重原子 ×30 构象
        # 能跑十几分钟），必须封顶，否则一个分子就把整批描述符计算拖死。
        cap = int(CFG.get("discovery.conformers", 30))
        if heavy > 45:
            cap = min(cap, 6)
        elif heavy > 35:
            cap = min(cap, 10)
        elif heavy > 28:
            cap = min(cap, 16)
        n_rot = int(rdMolDescriptors.CalcNumRotatableBonds(mol0))
        n_conf = int(min(cap, max(4, 4 * n_rot + 4)))
    n_conf = int(n_conf)
    key = _mol_key(smiles, n_conf)
    cache = _CACHE_DIR / f"{key}.pkl"
    if cache.exists():
        try:
            return pickle.loads(cache.read_bytes())
        except Exception:  # noqa: BLE001
            pass

    mol = Chem.AddHs(mol0)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 0xC0FFEE
    ps.pruneRmsThresh = 0.3
    ps.useSmallRingTorsions = True
    ps.maxIterations = 200          # 封顶嵌入尝试次数，避免大环上无限重试
    ps.numThreads = 0               # 0 = 用满可用核心
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=ps)
    if len(ids) == 0:
        ps.useRandomCoords = True
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=max(4, n_conf // 4), params=ps)
    if len(ids) == 0:
        return None
    energies: list[float] = []
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(
            mol, mmffVariant="MMFF94s",
            maxIters=300 if heavy > 35 else 600, numThreads=0)
        energies = [float(e) for _, e in res]
    except Exception:  # noqa: BLE001
        energies = [0.0] * len(ids)

    coords = np.stack([mol.GetConformer(i).GetPositions() for i in ids])
    Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    radii = np.array([VDW.get(int(z), 1.7) for z in Z])
    # MMFF 部分电荷（算偶极用）
    charges = np.zeros(len(Z))
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if props is not None:
            charges = np.array([props.GetMMFFPartialCharge(i) for i in range(len(Z))])
    except Exception:  # noqa: BLE001
        pass

    out = {"smiles": smiles, "coords": coords, "Z": Z, "radii": radii,
           "energies": np.array(energies, float), "charges": charges,
           "molblock": Chem.MolToMolBlock(mol, confId=int(ids[0])),
           "n_conf": len(ids)}
    try:
        cache.write_bytes(pickle.dumps(out))
    except Exception:  # noqa: BLE001
        pass
    return out


def boltzmann_weights(energies: np.ndarray, temperature: float = 298.15,
                      polar_penalty: np.ndarray | None = None,
                      dielectric: float = 78.0,
                      lam_base: float = 0.10) -> np.ndarray:
    """构象权重。

    dielectric 是**代理**而非真实溶剂化计算：以 polar_penalty（极性 SASA）为
    溶剂化稳定项，介电越高越偏好极性外露构象。用于近似"水相 vs 低介电孔内"的
    构象布居差异。任何基于它的结论都必须标注为 proxy，并由 L2(MD/xtb) 复核。
    """
    e = np.asarray(energies, float)
    e = e - np.nanmin(e)
    if polar_penalty is not None and dielectric > 1.0:
            # lam_base 取极性原子溶剂化的量级 (~0.1 kcal/mol/Å²)；(1-1/ε) 是 Born 因子
        lam = lam_base * (1.0 - 1.0 / dielectric)
        e = e - lam * np.asarray(polar_penalty, float)
        e = e - e.min()
    w = np.exp(-e / (0.001987 * temperature))
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


# ---------------- 表面积 ----------------
def _sasa_per_atom(smiles: str, conf_idx: int) -> np.ndarray | None:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdFreeSASA
    ens = ensemble(smiles)
    if ens is None:
        return None
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(ens["coords"][conf_idx]):
        conf.SetAtomPosition(i, xyz.tolist())
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    try:
        radii = rdFreeSASA.classifyAtoms(mol)
        rdFreeSASA.CalcSASA(mol, radii)
        return np.array([float(a.GetProp("SASA")) if a.HasProp("SASA") else 0.0
                         for a in mol.GetAtoms()])
    except Exception:  # noqa: BLE001
        _ = AllChem
        return None


def sasa_profile(smiles: str) -> dict | None:
    """每个构象的 总SASA / 极性SASA(3D PSA) / 疏水SASA。"""
    ens = ensemble(smiles)
    if ens is None:
        return None
    Z = ens["Z"]
    polar_mask = np.isin(Z, list(POLAR))
    # 极性 H（连在 N/O 上）也算极性面积
    total, polar = [], []
    for ci in range(ens["n_conf"]):
        sa = _sasa_per_atom(smiles, ci)
        if sa is None:
            continue
        total.append(float(sa.sum()))
        polar.append(float(sa[polar_mask].sum()))
    if not total:
        return None
    total = np.array(total)
    polar = np.array(polar)
    return {"total_sasa": total, "polar_sasa": polar,
            "apolar_sasa": total - polar, "energies": ens["energies"][:len(total)]}


# ---------------- 几何 ----------------
def _rotations(n: int = 64) -> np.ndarray:
    """球面近似均匀采样方向（Fibonacci 球）。"""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def _min_enclosing_circle_r(pts: np.ndarray, rad: np.ndarray,
                            iters: int = 300) -> float:
    """含 vdW 半径的最小外接圆半径（Badoiu-Clarkson 迭代，(1+eps) 近似）。"""
    return float(_min_enclosing_circle_batch(pts[None, :, :], rad, iters)[0])


def _min_enclosing_circle_batch(pts: np.ndarray, rad: np.ndarray,
                                iters: int = 300) -> np.ndarray:
    """批量版：pts (M, N, 2)，一次算 M 个投影的最小外接圆半径。

    对全部 M 个投影同时迭代，把 Python 层循环从 M*iters 降到 iters ——
    min_cross_section 要跑 构象数 × 取向数 个投影，不向量化会慢一个量级。
    """
    M = pts.shape[0]
    c = pts.mean(axis=1)                                   # (M,2)
    rows = np.arange(M)
    for i in range(iters):
        d = np.linalg.norm(pts - c[:, None, :], axis=2) + rad[None, :]
        j = np.argmax(d, axis=1)                           # (M,)
        c = c + (pts[rows, j] - c) / (i + 2.0)
    return (np.linalg.norm(pts - c[:, None, :], axis=2) + rad[None, :]).max(axis=1)


def min_cross_section(smiles: str, n_dir: int = 64) -> dict | None:
    """★ 构象系综上的**最小穿孔截面**。

    对每个构象、每个取向 d，把原子(含 vdW 半径)投影到 ⟂d 平面，
    取该投影的最小外接圆直径 = 该取向下能通过的最小圆柱孔径。
    对取向取最小 = 该构象的最优穿孔姿态；再对系综取最小/玻尔兹曼平均。

    这正是现有 'compound size (nm)' 与 Stokes 半径无法表达的量：
    柔性分子可以侧身穿孔。
    """
    ens = ensemble(smiles)
    if ens is None:
        return None
    dirs = _rotations(n_dir)
    # 为每个方向构造 ⟂d 的正交基（向量化）
    a = np.where(np.abs(dirs[:, :1]) < 0.9,
                 np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    u = np.cross(dirs, a)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(dirs, u)
    basis = np.stack([u, v], axis=2)                       # (D, 3, 2)

    coords = ens["coords"]                                 # (C, N, 3)
    proj = np.einsum("cnk,dkj->cdnj", coords, basis)       # (C, D, N, 2)
    C, D_, N, _ = proj.shape
    radii = _min_enclosing_circle_batch(proj.reshape(C * D_, N, 2), ens["radii"])
    per_conf = 2.0 * radii.reshape(C, D_).min(axis=1)      # 每构象取最优穿孔取向
    w = boltzmann_weights(ens["energies"][:len(per_conf)])
    return {"per_conformer_nm": per_conf / 10.0,
            "min_nm": float(per_conf.min() / 10.0),
            "boltzmann_nm": float((per_conf * w).sum() / 10.0),
            "max_nm": float(per_conf.max() / 10.0),
            "spread_nm": float((per_conf.max() - per_conf.min()) / 10.0)}


def shape_descriptors(smiles: str) -> dict | None:
    """回转半径、不对称度、主惯量比 NPR1/NPR2、离心率 —— 形状各向异性。"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors3D
    ens = ensemble(smiles)
    if ens is None:
        return None
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    vals: dict[str, list[float]] = {"rg": [], "npr1": [], "npr2": [],
                                    "asph": [], "ecc": []}
    for ci in range(ens["n_conf"]):
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, xyz in enumerate(ens["coords"][ci]):
            conf.SetAtomPosition(i, xyz.tolist())
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
        try:
            vals["rg"].append(float(Descriptors3D.RadiusOfGyration(mol)))
            vals["npr1"].append(float(Descriptors3D.NPR1(mol)))
            vals["npr2"].append(float(Descriptors3D.NPR2(mol)))
            vals["asph"].append(float(Descriptors3D.Asphericity(mol)))
            vals["ecc"].append(float(Descriptors3D.Eccentricity(mol)))
        except Exception:  # noqa: BLE001
            continue
    if not vals["rg"]:
        return None
    w = boltzmann_weights(ens["energies"][:len(vals["rg"])])
    out = {}
    for k, v in vals.items():
        arr = np.array(v)
        out[f"{k}_mean"] = float((arr * w).sum())
        out[f"{k}_min"] = float(arr.min())
        out[f"{k}_max"] = float(arr.max())
        out[f"{k}_spread"] = float(arr.max() - arr.min())
    return out


def intramolecular_hbonds(smiles: str, dist_cut: float = 2.6,
                          angle_cut: float = 120.0) -> dict | None:
    """★ 每个构象的分子内氢键数（几何判据）。变色龙假设的直接观测量。"""
    from rdkit import Chem
    ens = ensemble(smiles)
    if ens is None:
        return None
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    donors = []      # (H_idx, D_idx)
    for a in mol.GetAtoms():
        if a.GetAtomicNum() in (7, 8):
            for nb in a.GetNeighbors():
                if nb.GetAtomicNum() == 1:
                    donors.append((nb.GetIdx(), a.GetIdx()))
    acceptors = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8)]
    if not donors or not acceptors:
        return {"per_conformer": np.zeros(ens["n_conf"]), "mean": 0.0, "max": 0.0,
                "frac_with_imhb": 0.0}
    bonded = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()}
    bonded |= {(j, i) for i, j in bonded}
    counts = []
    for coords in ens["coords"]:
        n = 0
        for h, d in donors:
            for acc in acceptors:
                if acc == d or (d, acc) in bonded:
                    continue
                v1 = coords[acc] - coords[h]
                r = np.linalg.norm(v1)
                if r > dist_cut:
                    continue
                v2 = coords[d] - coords[h]
                cosang = float(np.dot(v1, v2) / (r * np.linalg.norm(v2) + 1e-9))
                ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
                if ang > angle_cut:
                    n += 1
        counts.append(n)
    counts = np.array(counts, float)
    w = boltzmann_weights(ens["energies"][:len(counts)])
    return {"per_conformer": counts, "mean": float((counts * w).sum()),
            "max": float(counts.max()), "frac_with_imhb": float((counts > 0).mean())}


def dipole(smiles: str) -> dict | None:
    """MMFF 电荷 + 构象坐标的偶极矩（Debye），按玻尔兹曼加权。"""
    ens = ensemble(smiles)
    if ens is None:
        return None
    q = ens["charges"]
    mus = []
    for coords in ens["coords"]:
        c = coords - coords.mean(0)
        mu = (q[:, None] * c).sum(0)
        mus.append(float(np.linalg.norm(mu) * 4.803))   # e·Å -> Debye
    mus = np.array(mus)
    w = boltzmann_weights(ens["energies"][:len(mus)])
    return {"mean_debye": float((mus * w).sum()), "min_debye": float(mus.min()),
            "max_debye": float(mus.max()), "spread_debye": float(mus.max() - mus.min())}


def chameleonicity(smiles: str) -> dict | None:
    """★ 分子变色龙指数（药物化学迁移概念）。

    比较"水相"(高介电，偏好极性外露)与"孔内低介电"(偏好折叠藏极性)两种
    玻尔兹曼布居下的 3D 极性表面积，相对落差即变色龙指数。

    注意：介电效应用 polar-SASA 溶剂化代理实现，是 proxy 不是隐式溶剂 QM。
    任何据此得到的结论必须走 L2（xtb/CPCM 或 MD）复核。
    """
    prof = sasa_profile(smiles)
    if prof is None:
        return None
    psa, e = prof["polar_sasa"], prof["energies"]
    w_water = boltzmann_weights(e, polar_penalty=psa, dielectric=78.0)
    w_pore = boltzmann_weights(e, polar_penalty=psa, dielectric=2.0)
    psa_w = float((psa * w_water).sum())
    psa_p = float((psa * w_pore).sum())
    denom = max(psa_w, 1e-6)
    return {"psa3d_water_A2": psa_w, "psa3d_lowdielectric_A2": psa_p,
            "delta_psa_A2": psa_w - psa_p,
            "chameleonicity": float((psa_w - psa_p) / denom),
            "psa_range_A2": float(psa.max() - psa.min()),
            "is_proxy": True}


def conformer_entropy(smiles: str) -> float | None:
    ens = ensemble(smiles)
    if ens is None:
        return None
    w = boltzmann_weights(ens["energies"])
    w = w[w > 1e-12]
    return float(-(w * np.log(w)).sum())


def flexibility(smiles: str) -> dict | None:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    cs = min_cross_section(smiles)
    return {"n_rotatable": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "n_rings": int(rdMolDescriptors.CalcNumRings(mol)),
            "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
            "tpsa_2d": float(Descriptors.TPSA(mol)),
            "cross_section_spread_nm": cs["spread_nm"] if cs else None}


AVAILABLE = {
    "ensemble": "构象系综 (coords/energies/Z/radii/charges)",
    "min_cross_section": "★ 最小穿孔截面 (nm)：min/boltzmann/max/spread",
    "shape_descriptors": "Rg / NPR1 / NPR2 / 不对称度 / 离心率，均含系综 spread",
    "intramolecular_hbonds": "★ 分子内氢键数：mean / max / frac_with_imhb",
    "sasa_profile": "每构象 总/极性/疏水 SASA",
    "chameleonicity": "★ 变色龙指数（水相 vs 低介电孔内 3D PSA 落差，proxy）",
    "dipole": "偶极矩 (Debye) 系综统计",
    "conformer_entropy": "构象熵（玻尔兹曼权重的 Shannon 熵）",
    "flexibility": "可旋转键、环数、Fsp3、2D TPSA、截面可变幅度",
    "boltzmann_weights": "自定义能量/介电代理下的构象权重",
}
