"""致知 ZHIZHI 冒烟测试：不调 LLM，只验证所有非 LLM 工具链路是否通。

    python smoke_test.py            跑全部
    python smoke_test.py --fast     跳过耗时项（消融、SHAP 交互、覆盖矩阵）
"""
from __future__ import annotations

import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

FAST = "--fast" in sys.argv
PASS, FAIL = [], []


def check(name: str, fn, slow: bool = False):
    if slow and FAST:
        print(f"  --   {name} (跳过)")
        return None
    t0 = time.time()
    try:
        r = fn()
        if isinstance(r, dict) and "error" in r:
            FAIL.append((name, r["error"]))
            print(f"  FAIL {name}: {r['error']}")
            return r
        PASS.append(name)
        print(f"  OK   {name}  [{time.time()-t0:.1f}s]")
        return r
    except Exception as e:  # noqa: BLE001
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return None


def main() -> None:
    from zhizhi.core import db
    db.init()

    print("\n[0] 新机制：模型路由 / 去重 / 并行 Agent")
    from zhizhi.core.config import CFG
    from zhizhi.lit import dedup, kg
    check("模型路由配置", lambda: {
        "ok": CFG.llm_model == "deepseek-ai/DeepSeek-V4-Pro"
              and CFG.literature_preprocess_model == "Pro/deepseek-ai/DeepSeek-V3.2"
              and CFG.get("data.mechanism_features")
              == ["ΦS", "ΦD", "∆Gs-m (J·m-2)"]
    } if CFG.llm_model == "deepseek-ai/DeepSeek-V4-Pro"
         and CFG.literature_preprocess_model == "Pro/deepseek-ai/DeepSeek-V3.2"
         and CFG.get("data.mechanism_features") == ["ΦS", "ΦD", "∆Gs-m (J·m-2)"]
         else {"error": "模型路由配置不符合预期"})
    check("论文身份归一", lambda: {
        "doi": dedup.normalize_doi("https://doi.org/10.1000/ABC"),
        "same_title": dedup.normalize_title("A title: test")
                      == dedup.normalize_title("A TITLE — TEST"),
    } if dedup.normalize_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
         and dedup.normalize_title("A title: test") == dedup.normalize_title("A TITLE — TEST")
         else {"error": "论文身份归一失败"})
    check("膜实体别名归一", lambda: {"name": kg.canonical_name("Membrane", "NF-270")}
          if kg.node_id("Membrane", "NF270") == kg.node_id("Membrane", "NF 270")
          == kg.node_id("Membrane", "NF-270") else {"error": "NF270 别名未合并"})

    def jobs_parallel_check():
        from unittest.mock import patch
        from zhizhi.core import jobs

        class FakeAgent:
            def run(self, session_id, prompt, thinking=False):
                time.sleep(0.08)
                yield {"type": "delta", "text": prompt}
                yield {"type": "text", "text": prompt}
                yield {"type": "done", "steps": 0}

        with patch.object(jobs, "_agent_factory", return_value=FakeAgent()):
            a = jobs.submit("bowen", f"smoke-a-{time.time_ns()}", "A")
            b = jobs.submit("gewu", f"smoke-b-{time.time_ns()}", "B")
            deadline = time.time() + 5
            while time.time() < deadline:
                ja, jb = jobs.get(a["job"]["id"]), jobs.get(b["job"]["id"])
                if ja["state"] == jb["state"] == "done":
                    return {"jobs": [ja["state"], jb["state"]]}
                time.sleep(0.03)
        return {"error": "并行 Agent 任务未按时完成"}

    check("并行 Agent 执行器", jobs_parallel_check)

    print("\n[1] 数据层")
    from zhizhi.dataio import loader
    check("load_raw", lambda: {"n": len(loader.load_raw())})
    check("build_matrix", lambda: {"shape": str(loader.build_matrix()["X"].shape)})
    check("data_health", loader.data_health)

    print("\n[2] 模型层")
    from zhizhi.tools import ml_tools as ML
    check("ml_data_qc", ML.ml_data_qc)
    check("ml_list_features", ML.ml_list_features)
    check("ml_model_report", ML.ml_model_report, slow=True)
    check("ml_residuals", lambda: ML.ml_residuals(top_n_list=3))
    check("ml_extrapolate", lambda: ML.ml_extrapolate(["compound"]))
    check("ml_explain(global)", lambda: ML.ml_explain("global", top_k=5))
    check("ml_explain(subgroup)",
          lambda: ML.ml_explain("subgroup", subgroup_query="membrane_class=='NF'", top_k=5))
    check("ml_counterfactual(matched)",
          lambda: ML.ml_counterfactual(mode="matched", min_tanimoto=0.6, min_gap=25))
    check("ml_counterfactual(synthetic)",
          lambda: ML.ml_counterfactual(mode="synthetic", row_id=0, target_removal=90))
    check("ml_mixed_effects",
          lambda: ML.ml_mixed_effects(["compound size (nm)", "Compound log K ow"],
                                      group="membrane"))
    check("ml_ablate", ML.ml_ablate, slow=True)
    check("ml_predict", lambda: ML.ml_predict([{"compound size (nm)": 0.5, "pH": 7}]))

    print("\n[3] 3D 构象原语")
    from zhizhi.desc import primitives as prim
    smi = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    check("ensemble", lambda: {"n_conf": prim.ensemble(smi)["n_conf"]})
    check("min_cross_section", lambda: prim.min_cross_section(smi))
    check("shape_descriptors", lambda: prim.shape_descriptors(smi))
    check("intramolecular_hbonds", lambda: prim.intramolecular_hbonds("OCC(O)C(O)C(O)C(O)CO"))
    check("chameleonicity", lambda: prim.chameleonicity("OCC(O)C(O)C(O)C(O)CO"))
    check("dipole", lambda: prim.dipole(smi))
    check("flexibility", lambda: prim.flexibility(smi))

    print("\n[4] 沙箱")
    from zhizhi.sandbox import runner
    check("sandbox 正常代码",
          lambda: runner.run_descriptor(
              "def compute(s):\n    import math\n    return float(len(s))",
              ["CCO", "c1ccccc1"], timeout=120))
    r = runner.run_descriptor("import os\ndef compute(s):\n    return 1.0", ["CCO"])
    if r.get("ok"):
        FAIL.append(("sandbox 拦截禁用 import", "没拦住 os"))
        print("  FAIL sandbox 拦截禁用 import: 没拦住 os")
    else:
        PASS.append("sandbox 拦截禁用 import")
        print("  OK   sandbox 拦截禁用 import")

    print("\n[5] 发现层")
    from zhizhi.tools import disc_tools as D
    check("disc_list_primitives", D.disc_list_primitives)
    check("disc_residual_clusters(含文献取证)",
          lambda: D.disc_residual_clusters(n_clusters=4, with_literature=True))
    check("disc_coverage_map(含文献核对)",
          lambda: D.disc_coverage_map(top_n=5, with_literature=True,
                                      check_top_k=3), slow=True)
    check("disc_list_cards", D.disc_list_cards)

    print("\n[6] 文献层（本地部分）")
    from zhizhi.tools import lit_tools as L
    check("lit_status", L.lit_status)
    check("lit_list_papers", lambda: L.lit_list_papers(limit=5))
    check("lit_kg_stats", L.lit_kg_stats)
    n_chunks = db.q1("SELECT COUNT(*) c FROM chunks")["c"]
    if n_chunks > 0:
        check("lit_search", lambda: L.lit_search("nanofiltration rejection log Kow", 3))
        check("lit_claims", lambda: L.lit_claims("size"))
    else:
        print("  --   lit_search (语料为空，先跑 python -m zhizhi.cli ingest)")

    print("\n[7] 验证层")
    from zhizhi.tools import val_tools as V
    check("val_power_analysis", lambda: V.val_power_analysis(10, 3))
    check("val_negative_results", V.val_negative_results)
    check("val_schedule", V.val_schedule)

    print("\n[7b] 生产模型层")
    from zhizhi.tools import prod_tools as PT
    check("ml_production_report", lambda: PT.ml_production_report(mode="base"), slow=True)
    check("ml_predict_smiles",
          lambda: PT.ml_predict_smiles([{"SMILES": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
                                         "pH": 7, "Pressure (kPa)": 800}]))
    check("ml_feature_importance(weight)",
          lambda: PT.ml_feature_importance(metric="weight", make_plot=True))
    check("ml_feature_importance(shap)",
          lambda: PT.ml_feature_importance(metric="shap", make_plot=False))
    check("ml_stratified_performance",
          lambda: PT.ml_stratified_performance(make_plot=False))
    check("ml_error_plots",
          lambda: PT.ml_error_plots(mode="oof",
                                    residual_vs_feature="compound size (nm)"))
    check("ml_export_predictions", PT.ml_export_predictions)
    check("ml_production_report(enhanced)",
          lambda: PT.ml_production_report(mode="enhanced"), slow=True)

    print("\n[7c] 图谱可视化")
    from zhizhi.lit import kgviz
    check("plotly_graph", lambda: {"n": kgviz.plotly_graph(min_degree=3)[1]["n_nodes"]})
    check("neighborhood_figure",
          lambda: {"n": kgviz.neighborhood_figure("NF270")[1].get("n_nodes")})
    check("contradiction_heatmap", lambda: kgviz.contradiction_heatmap()[1])
    check("export_static", lambda: kgviz.export_static(min_degree=4, max_nodes=80))
    check("graph_facts", lambda: {"n": kgviz.graph_facts()["stats"]["n_nodes"]})

    print("\n[7e] 三层联证与证据校验")
    check("disc_evidence_check(拦截无证据)",
          lambda: {"blocked": not D.disc_evidence_check({"x": 1}, "discovery")["ok"]})
    check("disc_evidence_check(放行完整证据)",
          lambda: {"passed": D.disc_evidence_check(
              {"delta_r2": 0.01, "quote": "Rejection increased with pH."},
              "discovery")["ok"]})

    print("\n[7d] 文献上传")
    from zhizhi.tools import lit_tools as LT
    check("lit_needs_fulltext", lambda: {"n": LT.lit_needs_fulltext()["n"]})

    print("\n[8] Agent 装配")
    from zhizhi.agents.registry import all_agents
    from zhizhi.core.tools import REGISTRY

    def wiring():
        A = all_agents()
        missing = sorted({t for a in A.values() for t in a.tool_names
                          if t not in REGISTRY.tools})
        if missing:
            return {"error": f"缺失工具 {missing}"}
        return {"n_tools": len(REGISTRY.names()),
                "agents": {k: len(a.tool_names) for k, a in A.items()}}
    check("工具注册与 Agent 绑定", wiring)

    print("\n" + "=" * 60)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    for n, e in FAIL:
        print(f"  ✗ {n}: {e}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
