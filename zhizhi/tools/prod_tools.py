"""量衡 · 生产模型工具集：围绕「你那个已训好的模型」的实用工具。

与 ml_tools.py 的分工：
  prod_tools —— 生产模型本身：复现、预测、调参、学习曲线、分层性能、重要性、出图
  ml_tools   —— 诚实诊断：分组 OOF 残差、消融、外推、SHAP、反事实、加描述符检验
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.config import CFG
from ..core.tools import P, obj, tool
from ..dataio import loader
from ..ml import plots
from ..ml import production as PROD


@tool("ml_production_report",
      "★ 生产模型：训练并落盘，返回训练/测试指标。"
      "mode='base' 为标准参数；mode='enhanced' 用多种子集成 + 更低学习率更多树，"
      "测试集指标更好。模型层的第一入口。",
      obj({"mode": P("string", "base | enhanced", enum=["base", "enhanced"]),
           "params_override": P("object", "临时覆盖超参，如 {\"max_depth\": 6}")}),
      category="production")
def ml_production_report(mode: str = "base",
                         params_override: dict | None = None) -> dict:
    return PROD.train_production(save=True, params=params_override or None, mode=mode)


@tool("ml_predict_smiles",
      "★ 用生产模型预测截留率。每行给 SMILES（自动算 12 个子结构）+ 已知的特征/操作条件，"
      "缺的留空。返回预测值、集成不确定度、缺失比例、适用域判定与可靠性标记。",
      obj({"rows": P("array", "样本列表。每个是 {\"SMILES\": \"...\", "
                     "\"compound size (nm)\": 0.6, \"pH\": 7, ...}",
                     items={"type": "object"})}, ["rows"]),
      category="production")
def ml_predict_smiles(rows: list[dict]) -> dict:
    if not rows:
        return {"error": "rows 不能为空"}
    return PROD.predict_rows(rows)


@tool("ml_feature_importance",
      "特征重要性，五种口径可选：weight（被用作分裂判据的次数，默认）/ gain（每次分裂的平均增益）"
      "/ cover（覆盖样本量）/ shap（对每条预测的实际贡献）/ permutation（打乱后 R² 跌幅）。"
      "不同口径排名不一致是正常的，它们衡量的不是同一件事。",
      obj({"metric": P("string", "weight | gain | cover | shap | permutation",
                       enum=["weight", "gain", "cover", "shap", "permutation"]),
           "make_plot": P("boolean", "是否出图，默认 true")}), category="production")
def ml_feature_importance(metric: str = "weight", make_plot: bool = True) -> dict:
    res = PROD.native_importance()
    metric = metric if metric in res["by_type"] else "weight"
    res["metric"] = metric
    res["metric_meaning_selected"] = PROD.IMPORTANCE_META.get(metric, "")
    lst = res["by_type"][metric]
    res["top"] = lst[:18]
    res["rank_1"] = lst[0]["feature"] if lst else None
    if make_plot and lst:
        res["figure"] = plots.importance_plot(
            lst, title=f"特征重要性 · {metric}（{PROD.IMPORTANCE_META.get(metric,'')}）")
    return res


@tool("ml_learning_curve",
      "★ 学习曲线：按化合物整簇抽样逐步增加训练数据，看分组 CV R² 还能不能涨。"
      "回答「再补多少数据才有用」——尾部斜率接近 0 说明瓶颈在描述符而不是样本量。",
      obj({"group": P("string", "抽样单位，默认 compound"),
           "n_repeat": P("integer", "每个比例重复几次，默认 3"),
           "make_plot": P("boolean", "是否出图，默认 true")}), category="production",
      long_running=True)
def ml_learning_curve(group: str = "compound", n_repeat: int = 3,
                      make_plot: bool = True) -> dict:
    res = PROD.learning_curve(group=group, n_repeat=n_repeat)
    if make_plot and res.get("curve"):
        res["figure"] = plots.learning_curve_plot(res["curve"])
    return res


@tool("ml_stratified_performance",
      "★ 分层性能拆解：按膜型号 / NF-RO / 分子量区间 / 截留区间 / 化合物分别给 R²、RMSE、"
      "系统偏差，定位模型到底在哪类样本上失效。",
      obj({"min_n": P("integer", "每层最少样本数，默认 10"),
           "make_plot": P("boolean", "是否出图，默认 true")}), category="production")
def ml_stratified_performance(min_n: int = 10, make_plot: bool = True) -> dict:
    res = PROD.stratified_performance(min_n=min_n)
    if make_plot:
        res["figures"] = {
            "by_membrane": plots.stratified_plot(res["by_membrane_worst"],
                                                 "各膜的分组 CV R²（最差 15 个）"),
            "by_mw_bin": plots.stratified_plot(res["by_mw_bin"], "各分子量区间的 R²"),
            "by_removal_bin": plots.stratified_plot(res["by_removal_bin"],
                                                    "各截留率区间的 R²"),
        }
    return res


@tool("ml_grid_search",
      "★ 超参网格搜索。scoring_mode=grouped 按化合物分组 CV 评分（反映新分子泛化，推荐）；"
      "=random 用随机 KFold（对齐既有 GridSearchCV 口径）。会告诉你现有超参是不是最优。",
      obj({"scoring_mode": P("string", "grouped | random", enum=["grouped", "random"]),
           "max_combos": P("integer", "最多试多少组合，默认 48（超出则随机抽样）"),
           "param_grid": P("object", "自定义搜索网格，留空用配置里的")}),
      category="production", long_running=True)
def ml_grid_search(scoring_mode: str = "grouped", max_combos: int = 48,
                   param_grid: dict | None = None) -> dict:
    return PROD.grid_search(scoring_mode=scoring_mode, max_combos=max_combos,
                            param_grid=param_grid or None)


@tool("ml_error_plots",
      "★ 误差分析图：预测-实测散点（带 y=x 和 ±10/±20 误差带）、残差分布、"
      "残差 vs 预测值、残差 vs 指定特征。PNG + SVG 双份，SVG 可直接进论文。",
      obj({"mode": P("string", "production=生产模型测试集 | oof=分组 OOF（全样本）",
                     enum=["production", "oof"]),
           "residual_vs_feature": P("string", "残差要对哪个特征作图，如 'compound size (nm)'")}),
      category="production")
def ml_error_plots(mode: str = "oof", residual_vs_feature: str = "") -> dict:
    from ..ml import model as M
    out: dict[str, Any] = {"mode": mode}
    if mode == "production":
        built = loader.build_matrix(with_missing_indicator=False)
        X, y = built["X"], built["y"]
        Xtr, Xte, ytr, yte = PROD.production_split(X, y)
        m, _ = PROD.load_production()
        yp = m.predict(Xte)
        yt = yte.to_numpy()
        fv = Xte[residual_vs_feature].to_numpy() if residual_vs_feature in Xte.columns else None
        out["parity"] = plots.parity_plot(yt, yp, "生产模型：预测 vs 实测",
                                          "随机切分测试集 (20%)")
    else:
        b = M.get_bundle()
        f = b.as_frame()
        yt, yp = f["y_true"].to_numpy(), f["y_oof"].to_numpy()
        fv = b.X[residual_vs_feature].to_numpy() if residual_vs_feature in b.X.columns else None
        out["parity"] = plots.parity_plot(yt, yp, "分组 OOF：预测 vs 实测",
                                          "按化合物分组交叉验证（无泄漏口径）")
    out["residuals"] = plots.residual_plots(yt, yp, fv, residual_vs_feature)
    if residual_vs_feature and fv is None:
        out["warning"] = f"特征 '{residual_vs_feature}' 不在特征集里，已跳过第三张图"
    return out


@tool("ml_compare_variants",
      "口径对照表：一次性给出「既有随机切分 / 分组CV(化合物·膜·文献) / "
      "加缺失指示位 / 去掉子结构」等多种设定下的指标，看清每个选择的代价。",
      obj({}), category="production", long_running=True)
def ml_compare_variants() -> dict:
    from ..ml import model as M
    rows = []
    prod = PROD.train_production(save=False)
    rows.append({"口径": "生产模型（随机切分 20%）", "n_features": prod["n_features"],
                 "R²": prod["test"]["r2"], "RMSE": prod["test"]["rmse"],
                 "说明": "日常预测用的口径"})
    for g, label in (("compound", "留一化合物（新分子）"),
                     ("membrane", "留一膜（新膜）"),
                     ("reference", "留一文献（新课题组）")):
        b = M.get_bundle(group=g)
        rows.append({"口径": label, "n_features": int(b.X.shape[1]),
                     "R²": b.metrics["group_cv"]["r2"],
                     "RMSE": b.metrics["group_cv"]["rmse"],
                     "说明": "分组交叉验证，用于判断结论能否外推"})
    b_isna = M.get_bundle(group="compound")  # 当前默认已是 32 列
    built_isna = loader.build_matrix(with_missing_indicator=True)
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score
    Xi, yi = built_isna["X"], built_isna["y"]
    gi = built_isna["groups"]["compound"]
    oof = np.full(len(yi), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(Xi, yi, groups=gi):
        mm = M.make_model()
        mm.fit(Xi.iloc[tr], yi.iloc[tr])
        oof[te] = mm.predict(Xi.iloc[te])
    rows.append({"口径": "留一化合物 + 缺失指示位", "n_features": int(Xi.shape[1]),
                 "R²": round(float(r2_score(yi, oof)), 4),
                 "RMSE": round(float(np.sqrt(np.mean((yi - oof) ** 2))), 3),
                 "说明": "诊断用；对比上一行看缺失模式本身有没有信息"})
    b_nosub = M.get_bundle(drop=["__GROUP__:substructure"], group="compound")
    rows.append({"口径": "留一化合物 − 12 个子结构", "n_features": int(b_nosub.X.shape[1]),
                 "R²": b_nosub.metrics["group_cv"]["r2"],
                 "RMSE": b_nosub.metrics["group_cv"]["rmse"],
                 "说明": "看子结构特征到底贡献多少"})
    return {"table": rows,
            "read": ("生产模型用于日常预测；分组口径回答的是另一个问题——"
                     "结论能不能推广到没见过的分子/膜/课题组，供发现层使用。")}


@tool("ml_export_predictions",
      "把整个数据集的预测结果导出成 CSV（含实测、生产模型预测、分组 OOF 预测、"
      "两种残差、化合物、膜、文献），可直接拿去画图或做二次分析。",
      obj({}), category="production")
def ml_export_predictions() -> dict:
    from ..ml import model as M
    b = M.get_bundle()
    f = b.as_frame()
    m, feats = PROD.load_production()
    f["y_production"] = m.predict(b.X[feats])
    f["residual_production"] = f["y_true"] - f["y_production"]
    p = CFG.logs_dir / "predictions_all.csv"
    f.to_csv(p, index=False, encoding="utf-8-sig")
    return {"file": str(p), "n_rows": int(len(f)),
            "columns": list(f.columns),
            "note": ("y_production 是生产模型对全样本的预测（含它训练过的行，会偏乐观）；"
                     "y_oof 是分组交叉验证的样本外预测，做机理分析请用 y_oof。")}
