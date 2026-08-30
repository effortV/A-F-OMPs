"""知识图谱：节点/边写入、查询、导出。每条边都挂原文引语，可回溯到 PDF。"""
from __future__ import annotations

import json
import re
import unicodedata

from ..core import db, remote

NODE_TYPES = ("Compound", "Membrane", "Mechanism", "Descriptor",
              "Condition", "Observation", "Paper", "Concept")


_MEMBRANE_CODE = re.compile(r"^[A-Za-z]{1,12}[\s_-]*\d{1,4}$")


def canonical_name(ntype: str, name: str) -> str:
    """实体显示名归一化，重点合并 NF270 / NF 270 / NF-270 一类别名。"""
    raw = unicodedata.normalize("NFKC", str(name or "")).strip()
    raw = re.sub(r"\s+", " ", raw)
    if ntype == "Membrane" and _MEMBRANE_CODE.fullmatch(raw):
        return re.sub(r"[\s_-]+", "", raw).upper()
    return raw


def norm(name: str, ntype: str = "") -> str:
    raw = canonical_name(ntype, name)
    raw = raw.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", raw).strip().lower()


def node_id(ntype: str, name: str) -> str:
    return f"{ntype}:{norm(name, ntype)}"


def add_node(ntype: str, name: str, meta: dict | None = None) -> str:
    nid = node_id(ntype, name)
    display = canonical_name(ntype, name)
    db.ex("INSERT INTO kg_nodes(id,type,name,meta) VALUES(?,?,?,?) "
          "ON CONFLICT(id) DO NOTHING",
          (nid, ntype, display, json.dumps(meta or {}, ensure_ascii=False)))
    return nid


def add_edge(src: str, rel: str, dst: str, paper_id: str = "",
             quote: str = "", weight: float = 1.0) -> None:
    duplicate = db.q1(
        "SELECT id FROM kg_edges WHERE src=? AND dst=? AND relation=? "
        "AND COALESCE(paper_id,'')=? AND COALESCE(quote,'')=? LIMIT 1",
        (src, dst, rel, paper_id or "", (quote or "")[:1200]),
    )
    if duplicate:
        return
    db.ex("INSERT INTO kg_edges(src,dst,relation,paper_id,quote,weight) VALUES(?,?,?,?,?,?)",
          (src, dst, rel, paper_id, (quote or "")[:1200], weight))


def canonicalize_existing_nodes() -> dict:
    """一次性合并历史实体别名，并去掉完全重复的边。"""
    changed = 0
    c = db.conn()
    with c:
        rows = c.execute("SELECT id,type,name,meta FROM kg_nodes").fetchall()
        for r in rows:
            new_name = canonical_name(r["type"], r["name"])
            new_id = node_id(r["type"], new_name)
            if new_id == r["id"]:
                continue
            c.execute(
                "INSERT INTO kg_nodes(id,type,name,meta) VALUES(?,?,?,?) "
                "ON CONFLICT(id) DO NOTHING",
                (new_id, r["type"], new_name, r["meta"]),
            )
            c.execute("UPDATE kg_edges SET src=? WHERE src=?", (new_id, r["id"]))
            c.execute("UPDATE kg_edges SET dst=? WHERE dst=?", (new_id, r["id"]))
            c.execute("DELETE FROM kg_nodes WHERE id=?", (r["id"],))
            changed += 1
        before = c.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
        c.execute(
            "DELETE FROM kg_edges WHERE id NOT IN ("
            "SELECT MIN(id) FROM kg_edges GROUP BY src,dst,relation,"
            "COALESCE(paper_id,''),COALESCE(quote,''))"
        )
        after = c.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
    return {"entities_merged": changed, "duplicate_edges_removed": before - after}


def entity_choices(ntype: str = "", limit: int = 1000) -> list[dict]:
    """供 UI 搜索下拉框使用，按关系数排序。"""
    if remote.enabled():
        return remote.call("module", "kg.entity_choices", ntype, limit)
    args: tuple = ()
    where = ""
    if ntype:
        where = "WHERE n.type=?"
        args = (ntype,)
    rows = db.q(
        "SELECT n.type,n.name,COUNT(e.id) degree FROM kg_nodes n "
        "LEFT JOIN kg_edges e ON e.src=n.id OR e.dst=n.id "
        f"{where} GROUP BY n.id ORDER BY degree DESC,n.name LIMIT ?",
        (*args, int(limit)),
    )
    return [{"type": r["type"], "name": r["name"], "degree": int(r["degree"] or 0)}
            for r in rows]


def stats() -> dict:
    if remote.enabled():
        return remote.call("module", "kg.stats")
    n = db.q1("SELECT COUNT(*) c FROM kg_nodes")["c"]
    e = db.q1("SELECT COUNT(*) c FROM kg_edges")["c"]
    by_type = {r["type"]: r["c"] for r in
               db.q("SELECT type, COUNT(*) c FROM kg_nodes GROUP BY type ORDER BY c DESC")}
    by_rel = {r["relation"]: r["c"] for r in
              db.q("SELECT relation, COUNT(*) c FROM kg_edges GROUP BY relation "
                   "ORDER BY c DESC LIMIT 20")}
    return {"n_nodes": n, "n_edges": e, "nodes_by_type": by_type, "top_relations": by_rel}


def neighbors(name: str, ntype: str = "", limit: int = 40) -> dict:
    exact = node_id(ntype, name) if ntype else ""
    like = f"%{norm(name, ntype)}%"
    if ntype:
        rows = db.q(
            "SELECT id,type,name FROM kg_nodes WHERE type=? AND (id=? OR id LIKE ?) "
            "ORDER BY (id=?) DESC LIMIT 8", (ntype, exact, like, exact))
    else:
        membrane_like = f"%{norm(name, 'Membrane')}%"
        rows = db.q("SELECT id,type,name FROM kg_nodes WHERE id LIKE ? OR id LIKE ? LIMIT 8",
                    (like, membrane_like))
    if not rows:
        return {"error": f"图谱中找不到节点 '{name}'"}
    out = []
    for r in rows:
        edges = db.q(
            "SELECT e.relation, e.src, e.dst, e.quote, e.paper_id, p.title, p.year "
            "FROM kg_edges e LEFT JOIN papers p ON p.id=e.paper_id "
            "WHERE e.src=? OR e.dst=? LIMIT ?", (r["id"], r["id"], limit))
        out.append({
            "node": {"id": r["id"], "type": r["type"], "name": r["name"]},
            "edges": [{"relation": x["relation"],
                       "other": x["dst"] if x["src"] == r["id"] else x["src"],
                       "direction": "out" if x["src"] == r["id"] else "in",
                       "quote": (x["quote"] or "")[:300],
                       "source": f"{x['title']} ({x['year']})" if x["title"] else x["paper_id"]}
                      for x in edges]})
    return {"matches": out}


def to_networkx():
    import networkx as nx
    g = nx.MultiDiGraph()
    for r in db.q("SELECT id,type,name FROM kg_nodes"):
        g.add_node(r["id"], type=r["type"], name=r["name"])
    for r in db.q("SELECT src,dst,relation,paper_id,quote FROM kg_edges"):
        g.add_edge(r["src"], r["dst"], relation=r["relation"],
                   paper=r["paper_id"], quote=r["quote"])
    return g


def export_graphml(path: str) -> str:
    import networkx as nx
    g = to_networkx()
    nx.write_graphml(g, path)
    return path
