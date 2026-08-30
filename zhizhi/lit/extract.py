"""结构化抽取：一篇文献 -> PaperCard + 机理主张 + 知识图谱三元组。

每条主张必须带原文引语和页码，否则丢弃 —— 图谱里不允许出现无出处的断言。
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from ..core import db
from ..core.llm import LLM
from . import dedup, kg

EXTRACT_API_VERSION = 2
PREPROCESS_SYS = """你负责文献摄取的低风险预处理。给你带 chunk 编号和页码的论文原文，
只做基础元数据/实验条件整理，并选出最值得交给高级模型判断机理的原文块。

不要提出机理主张，不要生成知识图谱三元组，不要判断创新性。
evidence_chunk_ids 只能填写输入中真实存在的 chunk 编号，优先覆盖摘要、结果、讨论、结论，
以及明确讨论截留机理、异常现象和条件依赖的段落，最多 14 个。

输出 JSON：
{"title": str, "year": int|null, "journal": str, "doi": str,
 "membranes": [{"name": str, "type": "NF|RO|other", "mwco_da": num|null,
                 "material": str, "note": str}],
 "compounds": [{"name": str, "class": str}],
 "conditions": {"pressure_kpa": str, "pH": str, "ionic_strength": str,
                 "temperature": str, "feed_conc": str, "mode": str, "fouling": str},
 "evidence_chunk_ids": [int]}
"""


SEMANTIC_SYS = """你是环境膜分离领域的高级文献语义审校专家。基础元数据已经由预处理模型
整理好；你的任务只负责高风险科研判断：机理主张、异常/局限、关键发现和知识图谱三元组。
输入中的证据段落全部来自论文原文。

铁律：
1. 每一条 mechanism_claim 和每一个 kg_triple 都必须带 quote（原文英文原句，逐字复制，
   不得改写、不得翻译），以及该句所在的 page（用文本里的 [p.N] 标记）。
   拿不出原句的主张一律不要写。
2. direction 只能是 up / down / nonmonotonic / none，含义是"该 descriptor 数值升高时，
   截留率如何变化"。
3. 不要输出你的领域常识，只输出这篇论文说了什么。
4. anomalies 里记录作者自己承认的反常、无法解释的现象、与文献不一致之处 —— 这是最有价值的部分。

输出 JSON schema：
{
 "key_findings": [str],
 "mechanism_claims": [{"descriptor": str, "direction": str, "membrane": str,
                       "scope": str, "statement": str, "quote": str, "page": int,
                       "confidence": 0-1}],
 "limitations": [str],
 "anomalies": [str],
 "kg_triples": [{"src_type": str, "src": str, "relation": str,
                 "dst_type": str, "dst": str, "quote": str}]
}
kg_triples 的 src_type/dst_type 取值限于：Compound, Membrane, Mechanism, Descriptor,
Condition, Observation, Concept。relation 用简短英文动词短语，如 rejected_by,
increases_rejection_of, explained_by, measured_under, contradicts, correlates_with。"""


def safe_page(value: Any) -> int:
    """Accept model page variants such as 18, ``p.18`` or ``page 18``."""
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(0, int(value))
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else 0


def safe_confidence(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        match = re.search(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", str(value or ""))
        result = float(match.group(0)) if match else 0.5
    return max(0.0, min(1.0, result))


def _ranked_blocks(paper_id: str, max_chars: int | None = None) -> list[dict]:
    """挑选最有信息量的原文块，保留真实 chunk 编号供预处理模型选择。"""
    from ..core.config import CFG
    max_chars = int(max_chars or CFG.get("literature.extract_max_chars", 22000))
    rows = db.q("SELECT idx,section,page,text FROM chunks WHERE paper_id=? ORDER BY idx",
                (paper_id,))
    if not rows:
        return []
    priority = {"abstract": 0, "front": 1, "conclusions": 2, "conclusion": 2,
                "results and discussion": 3, "results": 3, "discussion": 3,
                "introduction": 5, "materials and methods": 6, "methods": 6,
                "experimental": 6, "theory": 6}
    ranked = sorted(rows, key=lambda r: (priority.get((r["section"] or "").lower(), 7),
                                         r["idx"]))
    picked: list[dict] = []
    total = 0
    for r in ranked:
        t = f"[chunk.{r['idx']}][p.{r['page']}][{r['section']}] {r['text']}"
        if total + len(t) > max_chars:
            continue
        picked.append({"idx": int(r["idx"]), "page": safe_page(r["page"]),
                       "section": r["section"] or "", "text": r["text"],
                       "formatted": t})
        total += len(t)
    picked.sort(key=lambda x: x["idx"])
    return picked


def build_context(paper_id: str, max_chars: int | None = None) -> str:
    """兼容入口：返回带 chunk 编号和页码的候选上下文。"""
    return "\n\n".join(x["formatted"] for x in _ranked_blocks(paper_id, max_chars))


def preprocess_paper(paper_id: str, blocks: list[dict], agent: str = "bowen") -> dict:
    """V3.2：只做元数据/条件整理和证据块筛选，不产出科研主张。"""
    from ..core.config import CFG
    model = CFG.literature_preprocess_model
    llm = LLM(agent, model=model, fallbacks=[model], usage_kind="literature_preprocess")
    row = db.q1("SELECT title,abstract FROM papers WHERE id=?", (paper_id,))
    hint = f"已知题名（可能不准，以正文为准）：{row['title'] if row else ''}\n\n"
    out = llm.ask_json(
        PREPROCESS_SYS,
        hint + "论文候选原文块：\n\n" + "\n\n".join(x["formatted"] for x in blocks),
        temperature=0.05, thinking=False,
    )
    if not isinstance(out, dict):
        raise ValueError("预处理结果不是 JSON 对象")
    allowed = {x["idx"] for x in blocks}
    selected = []
    for raw in out.get("evidence_chunk_ids") or []:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        if idx in allowed and idx not in selected:
            selected.append(idx)
    out["evidence_chunk_ids"] = selected[:int(CFG.get("literature.preprocess_max_chunks", 14))]
    return out


def _semantic_context(blocks: list[dict], selected_ids: list[int]) -> str:
    """从原始块精确拼接 V4-Pro 上下文；绝不使用预处理模型改写后的引语。"""
    from ..core.config import CFG
    budget = int(CFG.get("literature.semantic_max_chars", 13000))
    selected = set(selected_ids)
    # 摘要/结果/讨论/结论作为保险块，避免低价预筛漏掉关键证据。
    must = {x["idx"] for x in blocks if (x["section"] or "").lower() in {
        "abstract", "results", "results and discussion", "discussion",
        "conclusion", "conclusions"}}
    if not selected:
        ordered = blocks
    else:
        wanted = selected | must
        ordered = [x for x in blocks if x["idx"] in wanted]
    parts, total = [], 0
    for x in ordered:
        t = x["formatted"]
        if total + len(t) > budget:
            continue
        parts.append(t)
        total += len(t)
    return "\n\n".join(parts)


def extract_paper(paper_id: str, agent: str = "bowen",
                  persist_result: bool = True,
                  progress: Callable[[str, float], None] | None = None) -> dict:
    blocks = _ranked_blocks(paper_id)
    if not blocks:
        return {"error": "无正文块"}
    row = db.q1("SELECT title, abstract FROM papers WHERE id=?", (paper_id,))
    from ..core.config import CFG
    if progress:
        progress("V3.2 筛选证据块与基础元数据", 0.62)
    try:
        prep = preprocess_paper(paper_id, blocks, agent)
        prep_error = ""
    except Exception as e:  # noqa: BLE001
        # 预处理失败不自动升级该步骤到 V4-Pro；V4 仍按保险块完成关键语义判断。
        prep = {"title": row["title"] if row else "", "year": None, "journal": "",
                "doi": "", "membranes": [], "compounds": [], "conditions": {},
                "evidence_chunk_ids": []}
        prep_error = f"{type(e).__name__}: {e}"[:300]

    ctx = _semantic_context(blocks, prep.get("evidence_chunk_ids") or [])
    if progress:
        progress("V4-Pro 审校机理主张与知识图谱三元组", 0.76)
    llm = LLM(agent, model=CFG.llm_model, fallbacks=[CFG.llm_model],
              usage_kind="literature_semantic")  # 固定主模型：V4-Pro
    semantic = llm.ask_json(
        SEMANTIC_SYS,
        "【V3.2 预处理的基础信息（只作索引，科研判断以原文为准）】\n"
        + json.dumps({k: prep.get(k) for k in
                      ("title", "year", "journal", "doi", "membranes", "compounds",
                       "conditions")}, ensure_ascii=False)
        + "\n\n【供 V4-Pro 审校的论文原文证据块】\n" + ctx,
        temperature=0.1, thinking=False,
    )
    if not isinstance(semantic, dict):
        return {"error": "抽取结果不是 JSON 对象"}
    card = {k: prep.get(k) for k in
            ("title", "year", "journal", "doi", "membranes", "compounds", "conditions")}
    card.update({k: semantic.get(k, []) for k in
                 ("key_findings", "mechanism_claims", "limitations", "anomalies",
                  "kg_triples")})
    card["_model_route"] = {
        "preprocess": CFG.literature_preprocess_model,
        "semantic": CFG.llm_model,
        "preprocess_error": prep_error,
        "semantic_context_chars": len(ctx),
    }
    if persist_result:
        persist(paper_id, card)
    return card


def persist(paper_id: str, card: dict[str, Any]) -> dict:
    """写入 claims / kg / papers.meta。丢弃无 quote 的条目。"""
    kept_claims = 0
    for c in (card.get("mechanism_claims") or []):
        q = (c.get("quote") or "").strip()
        if len(q) < 25:
            continue
        db.ex("INSERT INTO claims(paper_id,descriptor,direction,membrane,scope,"
              "statement,quote,page,confidence) VALUES(?,?,?,?,?,?,?,?,?)",
              (paper_id, str(c.get("descriptor", ""))[:120],
               str(c.get("direction", "none"))[:20], str(c.get("membrane", ""))[:120],
               str(c.get("scope", ""))[:300], str(c.get("statement", ""))[:1200],
               q[:1500], safe_page(c.get("page")), safe_confidence(c.get("confidence"))))
        kept_claims += 1

    pnode = kg.add_node("Paper", paper_id, {"title": card.get("title", "")})
    kept_edges = 0
    for m in (card.get("membranes") or []):
        if m.get("name"):
            n = kg.add_node("Membrane", m["name"], m)
            kg.add_edge(pnode, "studies", n, paper_id)
            kept_edges += 1
    for c in (card.get("compounds") or []):
        if c.get("name"):
            n = kg.add_node("Compound", c["name"], c)
            kg.add_edge(pnode, "studies", n, paper_id)
            kept_edges += 1
    for t in (card.get("kg_triples") or []):
        q = (t.get("quote") or "").strip()
        if len(q) < 25 or not t.get("src") or not t.get("dst"):
            continue
        st = t.get("src_type") if t.get("src_type") in kg.NODE_TYPES else "Concept"
        dt = t.get("dst_type") if t.get("dst_type") in kg.NODE_TYPES else "Concept"
        s = kg.add_node(st, t["src"])
        d = kg.add_node(dt, t["dst"])
        kg.add_edge(s, str(t.get("relation", "related_to"))[:60], d, paper_id, q)
        kept_edges += 1
    for cl in (card.get("mechanism_claims") or []):
        if cl.get("descriptor") and (cl.get("quote") or ""):
            n = kg.add_node("Descriptor", cl["descriptor"])
            kg.add_edge(n, f"affects_rejection_{cl.get('direction','none')}",
                        kg.add_node("Observation", "rejection"), paper_id, cl["quote"])

    extracted_title = (card.get("title") or "").strip()
    db.ex("UPDATE papers SET meta=?, title=CASE WHEN ?!='' THEN ? ELSE title END, "
          "year=COALESCE(year,?), doi=COALESCE(NULLIF(doi,''),?) WHERE id=?",
          (json.dumps(card, ensure_ascii=False), extracted_title, extracted_title,
           card.get("year"), card.get("doi") or "", paper_id))
    p = db.q1("SELECT doi,title,path FROM papers WHERE id=?", (paper_id,))
    if p:
        keys = dedup.identity(p["doi"], p["title"], p["path"] or "")
        db.ex("UPDATE papers SET doi_key=?,title_key=?,content_hash=? WHERE id=?",
              (keys["doi_key"], keys["title_key"], keys["content_hash"], paper_id))
    return {"claims": kept_claims, "edges": kept_edges}


# ---------------- 矛盾探测 ----------------
CONTRA_SYS = """你在为膜分离知识库做矛盾研判。给你同一个 descriptor 下、方向相反的两组文献主张。
判断这是不是真矛盾，还是被某个未记录的条件变量（膜孔径、pH、离子强度、结垢、
化合物子类、浓度区间、错流条件……）调和掉的表观矛盾。

输出 JSON：
{"is_real_contradiction": bool,
 "topic": str,
 "side_a_summary": str, "side_b_summary": str,
 "candidate_reconciling_variable": str,
 "reconciliation_hypothesis": str,
 "falsifiable_prediction": str,
 "confidence": 0-1}
falsifiable_prediction 必须写成"若调和假设成立，则在 ___ 条件下应观察到 ___"的可检验形式。"""


def canonicalize_descriptors(names: list[str], threshold: float = 0.82) -> dict[str, str]:
    """把自由文本的 descriptor 名归并到语义簇。

    "log Kow" / "hydrophobicity" / "octanol-water partition coefficient" 说的是同一件事，
    按字符串精确分组会漏掉大量真实矛盾。用 bge-m3 向量做单遍贪心聚类，
    簇名取该簇里出现频次最高的原始写法。向量服务不可用时退化为字符串归一。
    """
    uniq = sorted({n for n in names if n and n.strip()})
    if len(uniq) < 2:
        return {n: kg.norm(n) for n in uniq}
    try:
        import numpy as np
        vecs = LLM("bowen").embed(uniq)
    except Exception:  # noqa: BLE001
        return {n: kg.norm(n) for n in uniq}

    from collections import Counter
    freq = Counter(names)
    order = sorted(range(len(uniq)), key=lambda i: -freq[uniq[i]])
    centers: list[int] = []
    assign: dict[int, int] = {}
    for i in order:
        best, best_sim = -1, 0.0
        for c in centers:
            sim = float(vecs[i] @ vecs[c])
            if sim > best_sim:
                best, best_sim = c, sim
        if best >= 0 and best_sim >= threshold:
            assign[i] = best
        else:
            centers.append(i)
            assign[i] = i
    return {uniq[i]: kg.norm(uniq[c]) for i, c in assign.items()}


def detect_contradictions(min_papers: int = 2, agent: str = "bowen") -> list[dict]:
    """按 descriptor 语义簇归并主张，找方向相反的对，交给 LLM 研判。"""
    rows = db.q("SELECT id,paper_id,descriptor,direction,membrane,scope,statement,quote "
                "FROM claims WHERE direction IN ('up','down')")
    canon = canonicalize_descriptors([r["descriptor"] for r in rows])
    groups: dict[str, dict[str, list]] = {}
    for r in rows:
        key = canon.get(r["descriptor"]) or kg.norm(r["descriptor"])
        if not key:
            continue
        groups.setdefault(key, {"up": [], "down": []})[r["direction"]].append(dict(r))

    llm = LLM(agent, model=CFG.llm_model, fallbacks=[CFG.llm_model],
              usage_kind="literature_contradiction")
    made = []
    for desc, sides in groups.items():
        if len(sides["up"]) < min_papers or len(sides["down"]) < min_papers:
            continue
        exists = db.q1("SELECT id FROM contradictions WHERE descriptor=?", (desc,))
        if exists:
            continue
        def fmt(lst):
            return "\n".join(
                f"- [{c['membrane'] or '?'}|{c['scope'] or '?'}] {c['statement']}\n"
                f"  原句: \"{c['quote'][:280]}\"" for c in lst[:6])
        prompt = (f"descriptor: {desc}\n\n【A 组：认为升高该量会提高截留】\n{fmt(sides['up'])}\n\n"
                  f"【B 组：认为升高该量会降低截留】\n{fmt(sides['down'])}")
        try:
            j = llm.ask_json(CONTRA_SYS, prompt, temperature=0.2)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(j, dict):
            continue
        cid = db.ex(
            "INSERT INTO contradictions(topic,descriptor,side_a,side_b,claim_ids,status,"
            "note,created_at) VALUES(?,?,?,?,?,?,?,strftime('%s','now'))",
            (j.get("topic", desc), desc, j.get("side_a_summary", ""),
             j.get("side_b_summary", ""),
             json.dumps([c["id"] for c in sides["up"][:6] + sides["down"][:6]]),
             "open" if j.get("is_real_contradiction") else "explained",
             json.dumps(j, ensure_ascii=False))).lastrowid
        made.append({"id": cid, "descriptor": desc, **j})
    return made
