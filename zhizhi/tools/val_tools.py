"""验真 YANZHEN —— 验证层工具集。

验证调度 + 三层验证操作手册：
  L1 ML 数据验证（纯计算，全自动跑）
  L2 MD 分子动力学（只出可执行方案，不跑）
  L3 实验验证（判别性设计，不做无差别扩样）
"""
from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np
import pandas as pd

from ..core import db
from ..core.config import CFG
from ..core.llm import LLM
from ..core.tools import P, obj, tool
from ..dataio import loader
from ..desc import store as dstore
from ..ml import model as M


def _card(card_id: str) -> dict | None:
    r = db.q1("SELECT * FROM cards WHERE id=?", (card_id,))
    return dict(r) if r else None


# ==================== 调度 ====================
@tool("val_schedule",
      "验证调度：读卡片，按【信息增益 / 成本】给出 L1→L2→L3 的路径建议。"
      "不是每张卡都往下走 —— L1 就证伪的不浪费 MD 机时和实验耗材。",
      obj({"card_id": P("string", "卡片 id，留空则对所有 proposed 卡片排队")}),
      category="validation")
def val_schedule(card_id: str = "") -> dict:
    cards = [_card(card_id)] if card_id else db.rows_to_dicts(
        db.q("SELECT * FROM cards WHERE status IN ('proposed','tested') "
             "ORDER BY created_at DESC LIMIT 20"))
    cards = [c for c in cards if c]
    if not cards:
        return {"error": "没有待验证的卡片"}
    out = []
    for c in cards:
        has_desc = bool(db.q1("SELECT name FROM descriptors WHERE card_id=?", (c["id"],)))
        l1 = db.jdict(c["l1_result"]) or None
        if c["kind"] == "descriptor" and not has_desc:
            # 描述符卡但描述符行已不存在（被清理或改名）—— 不能当普通命题往 L3 送
            nxt, why = "孤儿卡，需清理或重建描述符", (
                "这张卡声明自己是描述符假设，但 descriptors 表里找不到对应记录，"
                "无法做任何检验。请重新 disc_prereg + disc_compute_descriptor，"
                "或把这张卡驳回。")
        elif not c["prereg_hash"] and has_desc:
            nxt, why = "补做预注册", "描述符类卡片没有预注册协议，检验结果不可采信"
        elif has_desc and not l1:
            nxt, why = "L1", "描述符已就绪但未跑自动验证电池，成本最低先跑"
        elif l1 and l1.get("verdict") == "FAIL":
            nxt, why = "结案(证伪)", "L1 已证伪，不投入 L2/L3；负结果本身写进卡片"
        elif l1 and l1.get("verdict") in ("WEAK",):
            nxt, why = "挂起或补数据", "有增量但没过负对照/FDR，等更多数据再议，不上实验"
        elif l1 and l1.get("verdict") == "PASS" and not c["l2_plan"]:
            nxt, why = "L2", "L1 通过，需要机理层面独立证据；先跑便宜的 xtb 路线"
        elif c["l2_plan"] and not c["l3_plan"]:
            nxt, why = "L3", "机理方案已就绪，设计判别性实验做终局检验"
        elif not has_desc:
            nxt, why = "L3(直接)", "非描述符类命题（如覆盖空白/矛盾调和），直接设计判别实验"
        else:
            nxt, why = "结案", "三层齐备，交人工审卡"
        out.append({"card_id": c["id"], "title": c["title"], "status": c["status"],
                    "novelty": c["novelty"], "next_step": nxt, "reason": why,
                    "l1_verdict": (l1 or {}).get("verdict")})
    return {"n": len(out), "queue": out,
            "principle": "成本序：L1(分钟级/免费) < L2(机时/可先用 xtb 廉价路线) < L3(周级/耗材)。"
                         "只有在上一层通过后才投下一层。"}


# ==================== L1 ====================
@tool("val_l1_battery",
      "★ L1 自动验证电池（全自动跑，几分钟）。对一个描述符依次执行："
      "分组CV ΔR² + bootstrap CI、y-scrambling 负对照、语义分组消融、"
      "留一文献外推、外部留出文献集、定向亚组检验、混合效应稳健性、"
      "以及跨所有历史描述符的 BH-FDR 多重比较校正。",
      obj({"descriptor_name": P("string", "描述符名"),
           "card_id": P("string", "关联卡片 id（结果会写回卡片）"),
           "target_subgroup": P("string", "定向检验子群 query，留空则读预注册里的")},
          ["descriptor_name"]), category="validation", long_running=True)
def val_l1_battery(descriptor_name: str, card_id: str = "",
                   target_subgroup: str = "") -> dict:
    from .ml_tools import ml_add_descriptor

    reg = db.q1("SELECT * FROM descriptors WHERE name=?", (descriptor_name,))
    if not reg:
        return {"error": f"描述符 {descriptor_name} 未注册"}
    spec = db.jdict(reg["spec"])
    target_subgroup = target_subgroup or spec.get("target_subgroup", "")
    card_id = card_id or (reg["card_id"] or "")

    core = ml_add_descriptor(descriptor_name, target_subgroup=target_subgroup)
    if "error" in core:
        return core

    vals = dstore.load_values(descriptor_name)
    extra = {descriptor_name: vals}

    # 留一文献外推 + 外部留出文献集
    base_ref = M.get_bundle(group="reference")
    new_ref = M.get_bundle(extra=extra, group="reference")
    built = loader.build_matrix(extra=extra)
    refs = pd.Series(built["groups"]["reference"])
    top_refs = refs.value_counts()
    holdout = list(top_refs.index[:int(CFG.get("validation.l1_external_holdout_refs", 1))])
    mask = refs.isin(holdout).to_numpy()
    ext: dict[str, Any] = {"holdout_reference": [h[:80] for h in holdout],
                           "n_holdout_rows": int(mask.sum())}
    if 20 < mask.sum() < len(mask) - 100:
        Xb = loader.build_matrix()["X"]
        y = built["y"]
        for tag, XX in (("without_descriptor", Xb), ("with_descriptor", built["X"])):
            m = M.make_model()
            m.fit(XX[~mask], y[~mask])
            ext[tag] = M._metrics(y[mask], m.predict(XX[mask]))
        ext["delta_r2_on_holdout"] = round(
            ext["with_descriptor"]["r2"] - ext["without_descriptor"]["r2"], 4)

    # 混合效应稳健性（膜为随机效应）
    from .ml_tools import ml_mixed_effects
    try:
        mixed = ml_mixed_effects([descriptor_name, "compound size (nm)",
                                  "Compound log K ow"], group="membrane")
    except Exception as e:  # noqa: BLE001
        mixed = {"error": str(e)}

    # 语义分组消融对照：加了新描述符后，哪些老特征组变得可有可无
    from .ml_tools import ml_ablate
    abl = ml_ablate()

    # 跨所有描述符的 BH-FDR
    allrows = [r for r in dstore.listing() if r["status"] in ("tested", "passed", "failed")]
    ps = []
    for r in allrows:
        m = db.jdict(r["metrics"])
        p = m.get("fdr", {}).get("p_combined")
        if p is not None:
            ps.append((r["name"], float(p)))
    fdr_table = []
    if ps:
        ps.sort(key=lambda kv: kv[1])
        n = len(ps)
        alpha = float(CFG.get("discovery.fdr_alpha", 0.1))
        crit = [(i + 1) / n * alpha for i in range(n)]
        passed_upto = -1
        for i, (_, p) in enumerate(ps):
            if p <= crit[i]:
                passed_upto = i
        for i, (nm, p) in enumerate(ps):
            fdr_table.append({"descriptor": nm, "p": round(p, 5),
                              "bh_critical": round(crit[i], 5),
                              "survives_bh": i <= passed_upto})

    verdict = core["verdict"]
    if verdict == "PASS" and ext.get("delta_r2_on_holdout") is not None \
            and ext["delta_r2_on_holdout"] < 0:
        verdict = "WEAK"
        core["downgrade_reason"] = "外部留出文献集上增益为负，降级为 WEAK"

    result = {
        "descriptor": descriptor_name, "verdict": verdict,
        "core_test": core,
        "leave_one_reference_out": {
            "r2_without": base_ref.metrics["group_cv"]["r2"],
            "r2_with": new_ref.metrics["group_cv"]["r2"],
            "delta_r2": round(new_ref.metrics["group_cv"]["r2"]
                              - base_ref.metrics["group_cv"]["r2"], 4)},
        "external_holdout": ext,
        "mixed_effects_membrane_random": mixed,
        "ablation_after_adding": abl["ablation"][:4],
        "fdr_across_all_descriptors": fdr_table,
        "checks_run": ["分组CV ΔR²", "bootstrap CI", "y-scrambling 负对照",
                       "置换重要度", "定向亚组检验", "留一文献外推",
                       "外部留出文献集", "混合效应稳健性", "语义分组消融",
                       "BH-FDR 多重比较校正"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if card_id:
        db.ex("UPDATE cards SET l1_result=?, status=?, updated_at=? WHERE id=?",
              (json.dumps(result, ensure_ascii=False),
               {"PASS": "tested", "WEAK": "tested", "FAIL": "refuted",
                "REDUNDANT": "refuted"}.get(verdict, "tested"),
               time.time(), card_id))
    return result


@tool("val_power_analysis",
      "L3 实验重复数估算：给定预期效应量和测量标准差，算每组需要多少个独立重复。",
      obj({"effect_size_pct": P("number", "预期两组截留率差值（百分点）"),
           "sd_pct": P("number", "单次测量标准差（百分点），LC-MS/MS 截留率典型 2-5"),
           "alpha": P("number", "显著性水平，默认 0.05"),
           "power": P("number", "统计功效，默认 0.8")},
          ["effect_size_pct", "sd_pct"]), category="validation")
def val_power_analysis(effect_size_pct: float, sd_pct: float,
                       alpha: float = 0.05, power: float = 0.8) -> dict:
    from scipy import stats
    if sd_pct <= 0 or effect_size_pct == 0:
        return {"error": "effect_size 和 sd 必须为正"}
    d = abs(effect_size_pct) / sd_pct
    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    n = 2 * ((za + zb) / d) ** 2
    # t 分布小样本修正。n 必须 >=2，否则自由度 2n-2 会降到 0 使 t.ppf 返回 NaN
    n_ceil = max(2, int(math.ceil(n)))
    for _ in range(20):
        df = 2 * n_ceil - 2
        ta = stats.t.ppf(1 - alpha / 2, df)
        tb = stats.t.ppf(power, df)
        n_new = max(2, int(math.ceil(2 * ((ta + tb) / d) ** 2)))
        if n_new == n_ceil:
            break
        n_ceil = n_new
    mde = (za + zb) * sd_pct * math.sqrt(2 / max(n_ceil, 2))
    return {"cohens_d": round(d, 3), "n_per_group": max(n_ceil, 3),
            "total_runs": max(n_ceil, 3) * 2,
            "minimum_detectable_effect_at_n3_pct": round(
                (za + zb) * sd_pct * math.sqrt(2 / 3), 2),
            "mde_at_computed_n_pct": round(mde, 2),
            "scope": "仅用于 L3 湿实验的两组差异设计，不用于 ML 模型验证或 L2 MD。",
            "assumptions": ["两组相互独立且样本量相同", "两组方差近似相等",
                            "双侧检验", "输入的是独立实验重复，不是同一样品的技术复测"],
            "read": (f"要以 {int(power*100)}% 把握在 α={alpha} 下检出 "
                     f"{effect_size_pct} 个百分点的差异，每组需 {max(n_ceil,3)} 次独立重复。"
                     "若效应量小于测量噪声的 1.5 倍，就不要做这个实验 —— 换判别力更强的设计。")}


# ==================== L2 ====================
L2_SYS = """你是分子模拟方案设计专家，要为一个膜分离机理假设写一份**可直接交给学生执行**的
MD 验证方案。不要跑计算，只出方案。方案必须具体到能照做。

必须包含以下小节，且给出具体数值与命令级细节：

1. 目标与判别点：这个 MD 到底要测哪个物理量，测出什么值支持假设、什么值证伪假设。
2. 廉价先行路线（必须给）：用 CREST/GFN2-xTB + CPCM 双介电（ε=78 水 vs ε≈2-4 孔内）
   做构象系综重加权，单分子 CPU 小时级。给出具体命令行、关键参数、以及判据。
   —— 这条路线是让课题组"今天就能开始"的，务必写实。
3. 全 MD 路线：
   3.1 膜模型构建：MPD/TMC 交联聚酰胺片段，Packmol 装配，交联度扫描（如 60/70/80%），
       用自由体积/孔径分布（PSD, 探针半径法）反算并校准到目标膜
       （NF270 有效孔半径约 0.42 nm，NF90 约 0.27 nm，具体以你的假设涉及的膜为准）。
   3.2 力场与水模型：溶质 GAFF2（RESP 或 AM1-BCC 电荷），膜 GAFF/CHARMM，
       水 TIP4P/2005；说明为什么这样选。
   3.3 平衡流程：能量最小化 -> NVT 退火 -> NPT 压缩 -> 生产，给温度/压力/时长/步长。
   3.4 采样与观测量：给出反应坐标（CV）的具体选择。若假设涉及构象变化，
       CV 必须是二维 (z 轴穿孔坐标, Rg 或分子内氢键数)，说明伞形采样窗口数、
       力常数、每窗时长、WHAM 解析。
   3.5 具体观测量清单：孔内 vs 体相的构象布居、分子内氢键数、局部介电常数估计、
       孔内扩散系数、分配自由能、PMF 势垒高度。
4. 预期数值与证伪条件：给出量化判据（如"若孔内 Rg 分布与体相的 KS 检验 p>0.05，
   则构象改变假设在该体系被证伪"）。
5. 计算成本估计：核时、墙钟时间、可并行度。
6. 常见坑：至少 3 条（如交联度对 PSD 的敏感性、伞形窗口重叠不足、
   有限尺寸效应、聚合物弛豫时间尺度不够）。
7. 分析脚本要点：列出要写哪些分析脚本、用什么工具（MDAnalysis / gmx 命令 / WHAM）。

用中文写，专业术语保留英文。输出 Markdown，不要 JSON。
直接写最终方案本身，不要写你的推敲过程（不要出现"我们可以说""或者……但……"这类措辞），
不要写开场白，第一行就是 `## 1. 目标与判别点`。"""


@tool("val_l2_md_protocol",
      "★ L2 分子动力学验证方案（只出手册，不跑计算）。包含廉价 xtb 先行路线 + 全 MD 路线、"
      "力场、平衡流程、二维伞形采样 CV、量化证伪判据、成本估计、常见坑、分析脚本要点。",
      obj({"card_id": P("string", "卡片 id（会自动带入假设与证据）"),
           "hypothesis": P("string", "若不给 card_id，直接给假设文本"),
           "membranes": P("array", "涉及的膜，如 ['NF270','NF90']",
                          items={"type": "string"}),
           "molecules": P("array", "涉及的分子（名字或 SMILES）", items={"type": "string"})}),
      category="validation", long_running=True)
def val_l2_md_protocol(card_id: str = "", hypothesis: str = "",
                       membranes: list[str] | None = None,
                       molecules: list[str] | None = None) -> dict:
    c = _card(card_id) if card_id else None
    if c and not hypothesis:
        hypothesis = c["statement"]
    if not hypothesis:
        return {"error": "需要 card_id 或 hypothesis"}
    ctx = f"假设：{hypothesis}\n"
    if c:
        ctx += f"卡片来源引擎：{c['engine']}\n新颖性判定：{c['novelty']}\n"
        if c["payload"]:
            ctx += f"证据摘要：{c['payload'][:3000]}\n"
        if c["l1_result"]:
            l1 = db.jdict(c["l1_result"])
            ctx += (f"L1 结论：{l1.get('verdict')}；"
                    f"ΔR²={l1.get('core_test', {}).get('delta_r2')}\n")
    if membranes:
        ctx += f"涉及膜：{', '.join(membranes)}\n"
    if molecules:
        ctx += f"涉及分子：{', '.join(molecules)}\n"

    # 关思维链：开着的话模型会把"我们可以说…或者…"这类推敲过程写进正文，
    md = LLM("yanzhen").ask(L2_SYS, ctx, temperature=0.3,
                            thinking=False)
    path = CFG.cards_dir / f"L2_{card_id or 'adhoc'}_{int(time.time())}.md"
    path.write_text(f"# L2 分子动力学验证方案\n\n> 假设：{hypothesis}\n\n{md}",
                    encoding="utf-8")
    if card_id:
        db.ex("UPDATE cards SET l2_plan=?, updated_at=? WHERE id=?",
              (md, time.time(), card_id))
    return {"card_id": card_id, "file": str(path), "protocol_markdown": md,
            "note": "本层只出方案不跑计算；廉价 xtb 路线可以立刻开始。"}


# ==================== L3 ====================
L3_SYS = """你是膜分离实验设计专家。为一个假设设计**判别性实验**（discriminating experiment）。

核心原则：只做能把 H1（新假设）和 H0（现有尺寸排阻 + 疏水分配标准解释）分开的实验。
**不做无差别扩样**。如果一个实验在 H1 和 H0 下预测相同，它就没有价值，不要写进来。

必须包含：
1. H1 与 H0 的并列陈述，以及它们在什么条件下给出**相反或最大差异**的预测（这是全篇核心）。
2. 判别性设计（给 2-3 个，按判别力排序），每个包含：
   - 设计类型（如：等体积不同柔性分子对 / 同分子跨极性梯度膜 / pH 摆动开关分子内氢键 /
     共溶剂调孔内介电 / 温度梯度分离焓熵）
   - 具体分子（给名字 + CAS + 为什么选它 + 标准品是否易得）
   - 具体膜（型号 + 为什么选它）
   - 完整操作条件：压力、错流速度、温度、pH、离子强度（背景电解质及浓度）、
     进水浓度、预压时间、取样时间点、回收率核算方式
   - 分析方法：LC-MS/MS 离子对/色谱柱建议、定量限要求、内标策略
   - **预期效应量**（H1 与 H0 预测的截留率差，给具体百分点）
   - **事先写死的判定规则**（如"若 A 分子在 NF270 上的截留比 B 低 >8 个百分点，
     且该差异在 pH 3 时消失，则支持 H1"）
3. 对照组与阴性对照：必须有。说明每个对照排除的是什么替代解释。
4. 重复数与功效：引用给出的功效分析结果，明确每组 n。
5. 混杂因素与规避：吸附/浓差极化/膜老化/批次差异/温度漂移，各写具体规避措施。
6. 失败模式：如果结果既不支持 H1 也不支持 H0，最可能的原因是什么，怎么补救。
7. 工作量估计：总实验数、机时、耗材、大致周期。

用中文写，术语保留英文。输出 Markdown，不要 JSON。
直接写最终方案本身，不要写你的推敲过程，不要写开场白，第一行就是 `## 1. H1 与 H0 的对立点`。"""


@tool("val_l3_experiment_design",
      "★ L3 判别性实验设计。自动带入：引擎2 的高价值空白组合、"
      "真实结构相似但截留差异大的匹配分子对、功效分析。"
      "只设计能分开 H1/H0 的实验，不做无差别扩样。",
      obj({"card_id": P("string", "卡片 id"),
           "hypothesis": P("string", "若不给 card_id，直接给假设"),
           "expected_effect_pct": P("number", "预期效应量（百分点），默认 10"),
           "measurement_sd_pct": P("number", "测量标准差（百分点），默认 3"),
           "include_blanks": P("boolean", "是否带入高价值空白组合，默认 true"),
           "include_matched_pairs": P("boolean", "是否带入匹配分子对，默认 true")}),
      category="validation", long_running=True)
def val_l3_experiment_design(card_id: str = "", hypothesis: str = "",
                             expected_effect_pct: float = 10.0,
                             measurement_sd_pct: float = 3.0,
                             include_blanks: bool = True,
                             include_matched_pairs: bool = True) -> dict:
    c = _card(card_id) if card_id else None
    if c and not hypothesis:
        hypothesis = c["statement"]
    if not hypothesis:
        return {"error": "需要 card_id 或 hypothesis"}

    ctx = f"假设 H1：{hypothesis}\n\n"
    if c and c["payload"]:
        ctx += f"支撑证据：{c['payload'][:2500]}\n\n"
    if include_blanks:
        from .disc_tools import disc_coverage_map
        try:
            cov = disc_coverage_map(top_n=12)
            ctx += ("【引擎2 给出的高价值空白组合（该化合物×该膜从未测过）】\n"
                    + json.dumps(cov.get("high_value_blanks", []), ensure_ascii=False)[:3500]
                    + "\n\n")
        except Exception:  # noqa: BLE001
            pass
    if include_matched_pairs:
        from .ml_tools import ml_counterfactual
        try:
            mp = ml_counterfactual(mode="matched", min_tanimoto=0.55, min_gap=20)
            ctx += ("【真实数据里结构极相似却截留差异极大的分子对（现有描述符解释不了）】\n"
                    + json.dumps(mp.get("pairs", [])[:10], ensure_ascii=False)[:3000] + "\n\n")
        except Exception:  # noqa: BLE001
            pass
    pw = val_power_analysis(expected_effect_pct, measurement_sd_pct)
    ctx += f"【功效分析结果】\n{json.dumps(pw, ensure_ascii=False)}\n"

    md = LLM("yanzhen").ask(L3_SYS, ctx, temperature=0.35,
                            thinking=False)
    path = CFG.cards_dir / f"L3_{card_id or 'adhoc'}_{int(time.time())}.md"
    path.write_text(f"# L3 判别性实验设计\n\n> 假设：{hypothesis}\n\n{md}", encoding="utf-8")
    if card_id:
        db.ex("UPDATE cards SET l3_plan=?, updated_at=? WHERE id=?",
              (md, time.time(), card_id))
    return {"card_id": card_id, "file": str(path), "power_analysis": pw,
            "design_markdown": md}


@tool("val_export_workorder",
      "把一张卡片的完整验证链（命题 + 预注册 + L1 结果 + L2 方案 + L3 设计）"
      "导出成一份可交付的 Markdown 验证工单。",
      obj({"card_id": P("string", "卡片 id")}, ["card_id"]), category="validation")
def val_export_workorder(card_id: str) -> dict:
    c = _card(card_id)
    if not c:
        return {"error": "卡片不存在"}
    l1 = db.jdict(c["l1_result"]) or None
    pre = db.jdict(c["prereg"]) or None
    pay = db.jdict(c["payload"])

    lines = [f"# 验证工单 {c['id']}：{c['title']}", "",
             f"- 来源引擎：`{c['engine']}`　类型：`{c['kind']}`",
             f"- 新颖性判定：**{c['novelty']}**　当前状态：**{c['status']}**",
             f"- 创建时间：{time.strftime('%Y-%m-%d %H:%M', time.localtime(c['created_at']))}",
             "", "## 1. 命题", "", f"> {c['statement']}", ""]
    if pay:
        lines += ["## 2. 支撑证据", "", "```json",
                  json.dumps(pay, ensure_ascii=False, indent=2)[:6000], "```", ""]
    if pre:
        lines += ["## 3. 预注册协议（结果产生前锁定）", "",
                  f"- 哈希：`{c['prereg_hash']}`", "", "```json",
                  json.dumps(pre, ensure_ascii=False, indent=2), "```", ""]
    else:
        lines += ["## 3. 预注册协议", "", "> ⚠ 本卡片没有预注册协议，"
                  "任何统计结论都应视为探索性而非确证性。", ""]
    if l1:
        core = l1.get("core_test") if isinstance(l1.get("core_test"), dict) else {}
        lines += ["## 4. L1 ML 数据验证", "",
                  f"**判定：{l1.get('verdict')}**", "",
                  "| 指标 | 值 |", "|---|---|",
                  f"| 基线分组CV R² | {core.get('r2_base')} |",
                  f"| 加入描述符后 R² | {core.get('r2_with')} |",
                  f"| ΔR² | {core.get('delta_r2')} |",
                  f"| ΔR² 95% CI | {core.get('delta_r2_ci95')} |",
                  f"| y-scrambling p | {core.get('p_yscramble')} |",
                  f"| 与既有特征最大相关 | {core.get('max_abs_corr_with_existing')} |",
                  f"| 置换重要度排名 | {core.get('permutation_importance_rank')} |",
                  f"| 外部留出集 ΔR² | {l1.get('external_holdout', {}).get('delta_r2_on_holdout')} |",
                  "", f"已执行检查：{'、'.join(l1.get('checks_run', []))}", ""]
    if c["l2_plan"]:
        lines += ["## 5. L2 分子动力学方案", "", c["l2_plan"], ""]
    if c["l3_plan"]:
        lines += ["## 6. L3 判别性实验设计", "", c["l3_plan"], ""]
    if c["review"]:
        lines += ["## 7. 人工审阅批注", "", f"> {c['review']}", ""]

    md = "\n".join(lines)
    path = CFG.cards_dir / f"workorder_{card_id}.md"
    path.write_text(md, encoding="utf-8")
    return {"card_id": card_id, "file": str(path), "n_chars": len(md),
            "markdown": md[:4000] + ("\n\n…[完整内容见文件]" if len(md) > 4000 else "")}


@tool("val_negative_results",
      "列出所有被证伪/降级的命题。负结果是资产：它界定了知识边界，也防止重复踩坑。",
      obj({}), category="validation")
def val_negative_results() -> dict:
    rows = db.rows_to_dicts(db.q(
        "SELECT id,title,statement,engine,novelty,l1_result FROM cards "
        "WHERE status IN ('refuted','parked') ORDER BY updated_at DESC LIMIT 30"))
    out = []
    for r in rows:
        l1 = db.jdict(r["l1_result"])
        core = l1.get("core_test") if isinstance(l1.get("core_test"), dict) else {}
        out.append({"card_id": r["id"], "title": r["title"],
                    "statement": r["statement"][:300],
                    "verdict": l1.get("verdict"),
                    "delta_r2": core.get("delta_r2"),
                    "why": core.get("read", "")})
    failed_desc = [r for r in dstore.listing() if r["status"] in ("failed", "redundant")]
    return {"n_refuted_cards": len(out), "refuted": out,
            "n_failed_descriptors": len(failed_desc),
            "failed_descriptors": [{"name": r["name"], "status": r["status"],
                                    "hypothesis": (r["hypothesis"] or "")[:200]}
                                   for r in failed_desc],
            "read": "这些是已经排除的解释。写论文时它们进讨论章节，做新假设时不要重复。"}
