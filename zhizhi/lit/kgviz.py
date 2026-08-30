"""知识图谱可视化：plotly 交互图 + 邻域子图 + 矛盾热力图 + 静态导出。

只用 plotly / networkx / matplotlib，不引新依赖、不依赖 CDN。
"""
from __future__ import annotations

import json
import time

import networkx as nx
import numpy as np
import pandas as pd

from ..core import db, remote
from ..core.config import CFG
from . import kg

TYPE_COLOR = {
    "Compound": "#2563EB",     # 蓝 化合物
    "Membrane": "#0F766E",     # 青 膜
    "Descriptor": "#B45309",   # 橙 描述符
    "Mechanism": "#7C3AED",    # 紫 机理
    "Observation": "#DC2626",  # 红 观测
    "Condition": "#65A30D",    # 绿 条件
    "Concept": "#DB2777",      # 品红 概念
    "Paper": "#94A3B8",        # 灰 文献
}


def _subgraph(node_types: list[str] | None = None, min_degree: int = 2,
              max_nodes: int = 300, drop_papers: bool = True) -> nx.Graph:
    """取一个能画得清楚的子图。1101 个节点全画会糊成一团。"""
    g = kg.to_networkx()
    u = nx.Graph()
    for a, b, data in g.edges(data=True):
        if u.has_edge(a, b):
            u[a][b]["w"] += 1
        else:
            u.add_edge(a, b, w=1, relation=data.get("relation", ""),
                       quote=(data.get("quote") or "")[:200])
    for n, d in g.nodes(data=True):
        if n in u:
            u.nodes[n].update(d)

    if drop_papers:
        u.remove_nodes_from([n for n, d in u.nodes(data=True)
                             if d.get("type") == "Paper"])
    if node_types:
        u.remove_nodes_from([n for n, d in u.nodes(data=True)
                             if d.get("type") not in node_types])
    u.remove_nodes_from([n for n, deg in dict(u.degree()).items() if deg < min_degree])
    if u.number_of_nodes() > max_nodes:
        keep = [n for n, _ in sorted(u.degree(), key=lambda kv: -kv[1])[:max_nodes]]
        u = u.subgraph(keep).copy()
    return u


def plotly_graph(node_types: list[str] | None = None, min_degree: int = 2,
                 max_nodes: int = 250, layout: str = "spring"):
    """全局交互图。返回 plotly Figure。"""
    if remote.enabled():
        return remote.call("module", "kgviz.plotly_graph", node_types, min_degree,
                           max_nodes, layout)
    import plotly.graph_objects as go
    u = _subgraph(node_types, min_degree, max_nodes)
    if u.number_of_nodes() == 0:
        return None, {"n_nodes": 0, "n_edges": 0,
                      "hint": "过滤太严，放宽 min_degree 或勾选更多节点类型"}

    if layout == "kamada" and u.number_of_nodes() <= 400:
        pos = nx.kamada_kawai_layout(u)
    else:
        pos = nx.spring_layout(u, seed=7, k=1.6 / np.sqrt(max(u.number_of_nodes(), 1)),
                               iterations=90)

    ex, ey = [], []
    for a, b in u.edges():
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    edge_trace = go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="none",
                            line=dict(width=0.6, color="rgba(120,132,148,.45)"))

    traces = [edge_trace]
    for t, color in TYPE_COLOR.items():
        ns = [n for n, d in u.nodes(data=True) if d.get("type") == t]
        if not ns:
            continue
        deg = dict(u.degree())
        traces.append(go.Scatter(
            x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns],
            mode="markers", name=t,
            marker=dict(size=[6 + 2.4 * np.sqrt(deg[n]) for n in ns],
                        color=color, line=dict(width=0.5, color="white")),
            text=[f"{u.nodes[n].get('name', n)}<br>类型 {t}<br>连接数 {deg[n]}" for n in ns],
            hoverinfo="text"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        showlegend=True, hovermode="closest", height=620,
        margin=dict(l=8, r=8, t=32, b=8),
        title=f"知识图谱  {u.number_of_nodes()} 节点 / {u.number_of_edges()} 边"
              f"（已过滤：度≥{min_degree}，隐藏文献节点）",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.03))
    return fig, {"n_nodes": u.number_of_nodes(), "n_edges": u.number_of_edges()}


def neighborhood_figure(name: str, hops: int = 1, max_nodes: int = 60):
    """某个实体的 1-2 跳邻域图，边上标关系类型。比全局图有用得多。"""
    if remote.enabled():
        return remote.call("module", "kgviz.neighborhood_figure", name, hops, max_nodes)
    import plotly.graph_objects as go
    g = kg.to_networkx()
    target = None
    key = kg.norm(name)
    for n, d in g.nodes(data=True):
        if key in n or key in kg.norm(d.get("name", "")):
            target = n
            break
    if target is None:
        return None, {"error": f"图谱里找不到 '{name}'"}

    und = g.to_undirected()
    nodes = {target}
    frontier = {target}
    for _ in range(max(1, hops)):
        nxt = set()
        for n in frontier:
            nxt |= set(und.neighbors(n))
        nodes |= nxt
        frontier = nxt
        if len(nodes) > max_nodes:
            break
    if len(nodes) > max_nodes:
        deg = dict(und.degree())
        nodes = {target} | set(sorted(nodes - {target},
                                      key=lambda n: -deg.get(n, 0))[:max_nodes - 1])
    sub = und.subgraph(nodes).copy()
    pos = nx.spring_layout(sub, seed=3, k=1.1, iterations=120)

    ex, ey, labels, lx, ly = [], [], [], [], []
    for a, b, data in sub.edges(data=True):
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
        rel = data.get("relation", "")
        if rel:
            lx.append((pos[a][0] + pos[b][0]) / 2)
            ly.append((pos[a][1] + pos[b][1]) / 2)
            labels.append(rel[:26])

    traces = [go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="none",
                         line=dict(width=1.0, color="rgba(120,132,148,.55)")),
              go.Scatter(x=lx, y=ly, mode="text", text=labels, hoverinfo="none",
                         textfont=dict(size=8, color="rgba(90,100,115,.85)"),
                         showlegend=False)]
    for t, color in TYPE_COLOR.items():
        ns = [n for n in sub.nodes if sub.nodes[n].get("type") == t]
        if not ns:
            continue
        traces.append(go.Scatter(
            x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns],
            mode="markers+text", name=t,
            marker=dict(size=[22 if n == target else 12 for n in ns], color=color,
                        line=dict(width=2 if target in ns else 0.6, color="white")),
            text=[str(sub.nodes[n].get("name", n))[:22] for n in ns],
            textposition="top center", textfont=dict(size=9),
            hovertext=[f"{sub.nodes[n].get('name', n)} ({t})" for n in ns],
            hoverinfo="text"))

    fig = go.Figure(data=traces)
    fig.update_layout(showlegend=True, height=560, margin=dict(l=8, r=8, t=34, b=8),
                      title=f"「{g.nodes[target].get('name', target)}」的 {hops} 跳邻域"
                            f"（{sub.number_of_nodes()} 节点）",
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.03))
    # 附带该实体的原文引语
    quotes = []
    for a, b, data in g.edges(data=True):
        if target in (a, b) and data.get("quote"):
            quotes.append({"relation": data.get("relation"),
                           "other": b if a == target else a,
                           "quote": data["quote"][:300], "paper": data.get("paper")})
    return fig, {"n_nodes": sub.number_of_nodes(), "quotes": quotes[:12]}


def contradiction_heatmap(min_claims: int = 2, top_desc: int = 22):
    """描述符 × 膜 的方向冲突热力图。颜色 = up 票数 − down 票数。

    颜色在同一行里出现正负交替 = 同一个因素在不同膜上效应反号，
    这正是引擎3 要吃的矛盾。
    """
    if remote.enabled():
        return remote.call("module", "kgviz.contradiction_heatmap", min_claims, top_desc)
    import plotly.graph_objects as go
    rows = db.q("SELECT descriptor, membrane, direction FROM claims "
                "WHERE direction IN ('up','down')")
    if not rows:
        return None, {"error": "还没有方向明确的机理主张，先跑文献摄取"}
    df = pd.DataFrame([dict(r) for r in rows])
    df["descriptor"] = df["descriptor"].str.strip().str.lower()
    df["membrane"] = (df["membrane"].fillna("(未指明)").str.strip()
                      .replace({"": "(未指明)", "all": "(通用)", "All": "(通用)"}))
    df["v"] = np.where(df["direction"] == "up", 1, -1)

    keep_d = df["descriptor"].value_counts()
    keep_d = keep_d[keep_d >= min_claims].head(top_desc).index
    keep_m = df["membrane"].value_counts().head(16).index
    d2 = df[df["descriptor"].isin(keep_d) & df["membrane"].isin(keep_m)]
    if d2.empty:
        return None, {"error": "过滤后没有数据，降低 min_claims"}

    piv = d2.pivot_table(index="descriptor", columns="membrane", values="v",
                         aggfunc="sum", fill_value=0)
    cnt = d2.pivot_table(index="descriptor", columns="membrane", values="v",
                         aggfunc="count", fill_value=0)
    conflict = [d for d in piv.index
                if ((piv.loc[d] > 0).any() and (piv.loc[d] < 0).any())]

    fig = go.Figure(go.Heatmap(
        z=piv.values, x=list(piv.columns), y=list(piv.index),
        colorscale=[[0, "#DC2626"], [0.5, "#F8FAFC"], [1, "#2563EB"]],
        zmid=0, colorbar=dict(title="up − down<br>票数"),
        text=cnt.values, texttemplate="%{text}", textfont=dict(size=9),
        hovertemplate="%{y} @ %{x}<br>净票数 %{z}<br>主张数 %{text}<extra></extra>"))
    fig.update_layout(height=max(360, 26 * len(piv)),
                      margin=dict(l=8, r=8, t=40, b=8),
                      title="描述符 × 膜 的效应方向（蓝=升高截留，红=降低截留，格内数字=主张数）",
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig, {"n_descriptors": len(piv), "n_membranes": len(piv.columns),
                 "conflicting_descriptors": conflict,
                 "read": "同一行里同时出现蓝和红 = 该因素在不同膜上效应反号，是引擎3 的直接素材。"}


def export_static(node_types: list[str] | None = None, min_degree: int = 3,
                  max_nodes: int = 160) -> dict:
    """matplotlib 静态图，PNG + SVG，可直接进论文。"""
    if remote.enabled():
        return remote.call("module", "kgviz.export_static", node_types, min_degree,
                           max_nodes)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..ml.plots import _save

    u = _subgraph(node_types, min_degree, max_nodes)
    if u.number_of_nodes() == 0:
        return {"error": "过滤后无节点"}
    pos = nx.spring_layout(u, seed=7, k=1.8 / np.sqrt(u.number_of_nodes()),
                           iterations=120)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    nx.draw_networkx_edges(u, pos, ax=ax, alpha=0.22, width=0.6)
    deg = dict(u.degree())
    for t, color in TYPE_COLOR.items():
        ns = [n for n, d in u.nodes(data=True) if d.get("type") == t]
        if ns:
            nx.draw_networkx_nodes(u, pos, nodelist=ns, ax=ax, node_color=color,
                                   node_size=[28 + 14 * deg[n] for n in ns],
                                   alpha=0.9, label=t, linewidths=0)
    hub = sorted(deg, key=lambda n: -deg[n])[:28]
    nx.draw_networkx_labels(u, pos, ax=ax,
                            labels={n: str(u.nodes[n].get("name", n))[:18] for n in hub},
                            font_size=7)
    ax.legend(scatterpoints=1, fontsize=8, loc="upper left")
    ax.set_title(f"知识图谱  {u.number_of_nodes()} 节点 / {u.number_of_edges()} 边"
                 f"（度≥{min_degree}）")
    ax.axis("off")
    out = _save(fig, "knowledge_graph")
    out.update({"n_nodes": u.number_of_nodes(), "n_edges": u.number_of_edges()})
    return out


# ==================== 自然语言导读 ====================
def graph_facts(top_n: int = 12) -> dict:
    """从图谱里抽出可读的事实，供导读使用。全部来自 DB，不是 LLM 编的。"""
    if remote.enabled():
        return remote.call("module", "kgviz.graph_facts", top_n)
    g = kg.to_networkx()
    und = g.to_undirected()
    deg = dict(und.degree())

    hubs = {}
    for t in ("Compound", "Membrane", "Descriptor", "Mechanism", "Concept"):
        ns = [(g.nodes[n].get("name", n), deg.get(n, 0))
              for n in g.nodes if g.nodes[n].get("type") == t]
        ns.sort(key=lambda kv: -kv[1])
        hubs[t] = [{"name": a, "degree": b} for a, b in ns[:top_n] if b > 0]

    # descriptor -> 方向票数
    dirs = db.q("SELECT descriptor, direction, COUNT(*) c FROM claims "
                "WHERE direction IN ('up','down','nonmonotonic') "
                "GROUP BY lower(descriptor), direction")
    agg: dict[str, dict] = {}
    for r in dirs:
        d = (r["descriptor"] or "").strip().lower()
        agg.setdefault(d, {"up": 0, "down": 0, "nonmonotonic": 0})[r["direction"]] += r["c"]
    ranked = sorted(agg.items(), key=lambda kv: -sum(kv[1].values()))[:top_n]
    descriptors = [{"descriptor": k, **v, "total": sum(v.values()),
                    "conflicted": bool(v["up"] > 0 and v["down"] > 0)}
                   for k, v in ranked]

    rels = {r["relation"]: r["c"] for r in db.q(
        "SELECT relation, COUNT(*) c FROM kg_edges GROUP BY relation ORDER BY c DESC LIMIT 12")}
    contras = db.rows_to_dicts(db.q(
        "SELECT descriptor, topic, status FROM contradictions ORDER BY id DESC LIMIT 8"))
    lvl = {r["evidence_level"]: r["c"] for r in db.q(
        "SELECT evidence_level, COUNT(*) c FROM papers WHERE status='done' "
        "GROUP BY evidence_level")}
    return {"stats": kg.stats(), "hubs": hubs, "descriptors": descriptors,
            "top_relations": rels, "contradictions": contras,
            "evidence_levels": lvl,
            "n_papers_done": db.q1("SELECT COUNT(*) c FROM papers WHERE status='done'")["c"]}


OVERVIEW_SYS = """你在给一个膜分离知识图谱写「导读」，读者是不熟悉图论的实验研究者。
他们看到一堆点和线不知道该看什么，你要用大白话告诉他们这张图在说什么。

要求：
1. 开门见山说这张图是从多少篇文献抽出来的、包含哪几类东西、规模多大。
2. 指出**枢纽节点**（连接最多的化合物/膜/描述符），并解释为什么它们是枢纽
   （通常是被研究得最多的对象），提醒读者这也意味着**采样偏倚**。
3. 讲清楚**哪些因素被最多文献讨论**，以及其中哪些存在方向冲突（同一因素有人说升有人说降）。
   方向冲突要单独点出来，这是最有价值的部分。
4. 给出 3-5 条**"你应该去看什么"**的具体建议，指名道姓到实体名。
5. 明确说明这张图的**局限**：只反映已入库文献说了什么，不等于事实；
   摘要级证据比全文级弱；没被研究过的组合在图上是看不见的。

写成 4-6 段中文，不要用编号列表堆砌，要像一个同行在旁边给你讲。
专业术语保留英文。不要吹，不要用"令人兴奋"这类词。不要编造数据里没有的东西。"""


def narrate_overview(agent: str = "bowen") -> dict:
    """整张图谱的自然语言导读。"""
    if remote.enabled():
        return remote.call("module", "kgviz.narrate_overview", agent)
    from ..core.llm import LLM
    facts = graph_facts()
    if facts["stats"]["n_nodes"] == 0:
        return {"error": "图谱为空，先跑文献摄取"}
    text = LLM(agent, model=CFG.llm_model, fallbacks=[CFG.llm_model],
               usage_kind="literature_answer").ask(
        OVERVIEW_SYS,
        "图谱事实（全部来自数据库，请勿添加其它内容）：\n"
        + json.dumps(facts, ensure_ascii=False, indent=1)[:12000],
        temperature=0.3, thinking=False)
    return {"narrative": text, "facts": facts}


ENTITY_SYS = """你在解释一个膜分离知识图谱里某个实体的「邻域」，读者是实验研究者。
给你的是这个实体连出去的所有关系，以及每条关系的原文引语。

要求：
1. 一句话说清这个实体是什么、在这批文献里被研究到什么程度（几篇提到、连了多少个邻居）。
2. 按**关系类型**归类讲：它影响什么、被什么影响、和什么共现。不要逐条念，要归纳。
3. 凡是引用结论，必须**带上原文英文引语**（用引号），这是硬要求。
4. 如果邻居里出现方向矛盾（既有 increases 又有 decreases），必须单独指出来并说明可能的条件差异。
5. 最后一段说：关于这个实体，**图谱里还缺什么**（哪些明显该有的关系没有）。

写成 3-5 段中文，术语保留英文。不要编造引语。"""


def narrate_entity(name: str, agent: str = "bowen") -> dict:
    """某个实体的自然语言导读，带原文引语。"""
    if remote.enabled():
        return remote.call("module", "kgviz.narrate_entity", name, agent)
    from ..core.llm import LLM
    nb = kg.neighbors(name, limit=60)
    if "error" in nb:
        return nb
    m = nb["matches"][0]
    claims = db.rows_to_dicts(db.q(
        "SELECT c.descriptor,c.direction,c.membrane,c.scope,c.statement,c.quote,c.page,"
        "p.title,p.year FROM claims c JOIN papers p ON p.id=c.paper_id "
        "WHERE lower(c.descriptor) LIKE ? OR lower(c.membrane) LIKE ? LIMIT 20",
        (f"%{name.lower()}%", f"%{name.lower()}%")))
    payload = {"node": m["node"], "n_edges": len(m["edges"]),
               "edges": m["edges"][:40],
               "claims": [{k: (str(v)[:300] if k == "quote" else v)
                           for k, v in c.items()} for c in claims]}
    text = LLM(agent, model=CFG.llm_model, fallbacks=[CFG.llm_model],
               usage_kind="literature_answer").ask(
        ENTITY_SYS,
        f"实体：{name}\n\n图谱数据：\n"
        + json.dumps(payload, ensure_ascii=False, indent=1)[:12000],
        temperature=0.3, thinking=False)
    return {"entity": m["node"]["name"], "narrative": text,
            "n_edges": len(m["edges"]), "n_claims": len(claims)}


CONFLICT_SYS = """你在解读一张「描述符 × 膜」的效应方向热力图，读者是实验研究者。
蓝色 = 该因素升高时截留率升高，红色 = 降低，格内数字 = 支撑该结论的文献主张数。

要求：
1. 先用两句话说清这张图怎么看。
2. 逐个讲**同一行里蓝红并存**的因素（这意味着同一个因素在不同膜上效应反号），
   每个都要说：在哪些膜上是升、哪些膜上是降、可能的物理原因是什么。
3. 指出**证据薄弱**的格子（主张数只有 1-2 条），提醒不要过度解读。
4. 最后给 2-3 条"下一步该测什么"的具体建议，指名膜型号和条件。

写成 3-5 段中文，术语保留英文。只讲数据里有的，不要外推。"""


def narrate_conflicts(agent: str = "bowen") -> dict:
    """矛盾热力图的自然语言解读。"""
    if remote.enabled():
        return remote.call("module", "kgviz.narrate_conflicts", agent)
    from ..core.llm import LLM
    rows = db.q("SELECT descriptor, membrane, direction, COUNT(*) c FROM claims "
                "WHERE direction IN ('up','down') GROUP BY lower(descriptor), membrane, direction")
    if not rows:
        return {"error": "还没有方向明确的主张"}
    tbl: dict = {}
    for r in rows:
        d = (r["descriptor"] or "").strip().lower()
        mb = (r["membrane"] or "(未指明)").strip() or "(未指明)"
        tbl.setdefault(d, {}).setdefault(mb, {"up": 0, "down": 0})[r["direction"]] += r["c"]
    conflicted = {d: v for d, v in tbl.items()
                  if any(x["up"] for x in v.values()) and any(x["down"] for x in v.values())}
    contras = db.rows_to_dicts(db.q(
        "SELECT descriptor, side_a, side_b, note FROM contradictions LIMIT 8"))
    payload = {"conflicted_descriptors": conflicted,
               "n_total_descriptors": len(tbl),
               "recorded_contradictions": contras}
    text = LLM(agent, model=CFG.llm_model, fallbacks=[CFG.llm_model],
               usage_kind="literature_answer").ask(
        CONFLICT_SYS,
        "热力图数据：\n" + json.dumps(payload, ensure_ascii=False, indent=1)[:12000],
        temperature=0.3, thinking=False)
    return {"narrative": text, "n_conflicted": len(conflicted),
            "conflicted_descriptors": sorted(conflicted)}
