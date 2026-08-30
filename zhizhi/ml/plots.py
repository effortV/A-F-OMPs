"""模型诊断图：预测-实测散点、残差分布、残差 vs 特征、学习曲线。

统一走 matplotlib，输出 PNG + SVG（SVG 可直接进论文）。
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..core.config import CFG  # noqa: E402

# Windows 常见中文字体，找不到就退回英文标签
for _f in ("Microsoft YaHei", "SimHei", "DejaVu Sans"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:  # noqa: BLE001
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


_SEQ = itertools.count()


def _save(fig, stem: str) -> dict:
    """文件名必须唯一。用秒级时间戳会让同一秒内产出的多张图重名，
    进而在 Streamlit 里撞 download_button 的 key —— 加一个进程内自增序号。"""
    d = CFG.figures_dir
    d.mkdir(parents=True, exist_ok=True)
    tag = f"{int(time.time())}_{next(_SEQ):04d}"
    png = d / f"{stem}_{tag}.png"
    svg = d / f"{stem}_{tag}.svg"
    fig.savefig(png)
    fig.savefig(svg)
    plt.close(fig)
    return {"png": str(png), "svg": str(svg)}


def parity_plot(y_true, y_pred, title: str = "预测 vs 实测",
                subtitle: str = "") -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(y_true, y_pred, s=12, alpha=0.45, edgecolors="none", color="#2563EB")
    lo, hi = -3, 103
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    for band in (10, 20):
        ax.fill_between([lo, hi], [lo - band, hi - band], [lo + band, hi + band],
                        color="grey", alpha=0.07)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("实测截留率 (%)")
    ax.set_ylabel("预测截留率 (%)")
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title, fontsize=11)
    ax.text(0.04, 0.95, f"R² = {r2:.4f}\nRMSE = {rmse:.2f}\nn = {len(y_true)}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CBD5E1"))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)
    return _save(fig, "parity")


def residual_plots(y_true, y_pred, feature_values=None,
                   feature_name: str = "") -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    res = y_true - y_pred
    n = 3 if feature_values is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.0))

    axes[0].hist(res, bins=45, color="#2563EB", alpha=0.75)
    axes[0].axvline(0, color="k", ls="--", lw=1)
    axes[0].set_xlabel("残差 = 实测 − 预测 (百分点)")
    axes[0].set_ylabel("样本数")
    axes[0].set_title(f"残差分布  中位={np.median(res):.2f}  sd={res.std():.2f}",
                      fontsize=10)
    axes[0].grid(alpha=0.2)

    axes[1].scatter(y_pred, res, s=10, alpha=0.4, edgecolors="none", color="#0F766E")
    axes[1].axhline(0, color="k", ls="--", lw=1)
    axes[1].set_xlabel("预测截留率 (%)")
    axes[1].set_ylabel("残差 (百分点)")
    axes[1].set_title("残差 vs 预测值（看异方差）", fontsize=10)
    axes[1].grid(alpha=0.2)

    if feature_values is not None:
        v = np.asarray(feature_values, float)
        ok = ~np.isnan(v)
        axes[2].scatter(v[ok], res[ok], s=10, alpha=0.4, edgecolors="none",
                        color="#B45309")
        axes[2].axhline(0, color="k", ls="--", lw=1)
        axes[2].set_xlabel(feature_name)
        axes[2].set_ylabel("残差 (百分点)")
        axes[2].set_title(f"残差 vs {feature_name}（看漏掉的趋势）", fontsize=10)
        axes[2].grid(alpha=0.2)
    return _save(fig, "residuals")


def learning_curve_plot(curve: list[dict]) -> dict:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    x = [c["fraction"] * 100 for c in curve]
    y = [c["r2_mean"] for c in curve]
    e = [c["r2_sd"] for c in curve]
    ax.errorbar(x, y, yerr=e, marker="o", capsize=3, color="#2563EB")
    ax.set_xlabel("使用的化合物比例 (%)")
    ax.set_ylabel("分组交叉验证 R²")
    ax.set_title("学习曲线（按化合物整簇抽样，避免重复记录泄漏）", fontsize=10)
    ax.grid(alpha=0.25)
    return _save(fig, "learning_curve")


def importance_plot(items: list[dict], value_key: str = "share_pct",
                    label_key: str = "feature", top: int = 18,
                    title: str = "特征重要性") -> dict:
    items = items[:top][::-1]
    fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.30 * len(items))))
    ax.barh([str(i[label_key])[:34] for i in items],
            [i[value_key] for i in items], color="#2563EB", alpha=0.85)
    ax.set_xlabel(value_key)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, "importance")


def stratified_plot(rows: list[dict], title: str) -> dict:
    rows = [r for r in rows if r.get("r2") is not None][::-1]
    if not rows:
        return {}
    fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.32 * len(rows))))
    labels = [f"{r['level'][:28]} (n={r['n']})" for r in rows]
    vals = [r["r2"] for r in rows]
    colors = ["#DC2626" if v < 0 else "#2563EB" for v in vals]
    ax.barh(labels, vals, color=colors, alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("分组交叉验证 R²")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, "stratified")


def list_figures(limit: int = 30) -> list[str]:
    d: Path = CFG.figures_dir
    if not d.exists():
        return []
    fs = sorted(d.glob("*.png"), key=lambda p: -p.stat().st_mtime)
    return [str(p) for p in fs[:limit]]
