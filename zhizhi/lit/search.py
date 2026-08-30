"""文献外部检索：OpenAlex / Crossref / Europe PMC / Semantic Scholar + 全文获取。

两条触发路径：
  A. 用户给关键词 -> LLM 扩展成中英检索式
  B. 发现层索要证据 -> 必须同时检正例与反例（stance='both_sides'），防确认偏误
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from ..core import db
from ..core.config import CFG
from ..core.llm import LLM
from . import dedup

SEARCH_API_VERSION = 2
UA = {"User-Agent": f"ZHIZHI/1.0 (mailto:{CFG.env('OPENALEX_EMAIL', 'anon@example.com')})"}
TIMEOUT = 30


def _get(url: str, params: dict | None = None, headers: dict | None = None,
         timeout: float | None = None, attempts: int = 3) -> Any:
    h = dict(UA)
    h.update(headers or {})
    attempts = max(1, int(attempts))
    timeout = float(timeout or TIMEOUT)
    for attempt in range(attempts):
        try:
            r = requests.get(url, params=params, headers=h, timeout=timeout)
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001
            if attempt == attempts - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())[:90]


def work_id(doi: str, title: str) -> str:
    base = (doi or "").lower().strip() or _norm_title(title)
    return "w" + hashlib.md5(base.encode()).hexdigest()[:14]


# ---------------- 检索式扩展 ----------------
EXPAND_SYS = """你在为膜分离（NF/RO 去除有机微污染物）领域做文献检索式设计。
根据研究问题生成检索式。要求：
- 英文检索式 6-10 条，覆盖同义词、术语变体、膜型号别名、机理术语；
- 中文检索式 3-5 条（用于中文库或双语核对）；
- 若 stance = both_sides，必须专门设计能捞到"反例/否定结论/不一致报道"的检索式
  （如加 "no correlation" / "contrary" / "does not" / "overestimat*" 等）；
- 若问题涉及跨学科概念，额外给 2-3 条**外领域**检索式（如药物化学、色谱、沸石催化）。

输出 JSON：
{"en": [str], "zh": [str], "cross_domain": [str],
 "negative_evidence": [str], "year_min": int|null, "rationale": str}"""


def expand_queries(question: str, must_cover: list[str] | None = None,
                   stance: str = "both_sides", agent: str = "bowen",
                   model: str | None = None,
                   request_timeout: float | None = None,
                   attempts: int | None = None) -> dict:
    # Query construction is mechanical literature preprocessing.  Critical
    # relevance decisions below remain fixed to V4-Pro.
    chosen_model = model or CFG.literature_preprocess_model
    llm = LLM(agent, model=chosen_model, fallbacks=[chosen_model],
              usage_kind="literature_query_design")
    u = f"研究问题：{question}\nstance：{stance}\n"
    if must_cover:
        u += "必须覆盖的要点：" + "；".join(must_cover)
    try:
        return llm.ask_json(EXPAND_SYS, u, temperature=0.4,
                            thinking=False, request_timeout=request_timeout,
                            attempts=attempts)
    except Exception:  # noqa: BLE001
        return {"en": [question], "zh": [], "cross_domain": [],
                "negative_evidence": [], "year_min": None, "rationale": "LLM 扩展失败，用原问题"}


# ---------------- 各数据源 ----------------
def s_openalex(query: str, n: int = 25, year_min: int | None = None,
               timeout: float | None = None, attempts: int = 3) -> list[dict]:
    params = {"search": query, "per-page": min(n, 50),
              "mailto": CFG.env("OPENALEX_EMAIL", "")}
    key = CFG.env("OPENALEX_API_KEY")
    if key:
        params["api_key"] = key
    if year_min:
        params["filter"] = f"from_publication_date:{year_min}-01-01"
    j = _get("https://api.openalex.org/works", params,
             timeout=timeout, attempts=attempts)
    out = []
    for w in ((j or {}).get("results") or []):
        inv = w.get("abstract_inverted_index")
        abstract = ""
        if inv:
            pos: dict[int, str] = {}
            for word, idxs in inv.items():
                for i in idxs:
                    pos[i] = word
            abstract = " ".join(pos[i] for i in sorted(pos))[:4000]
        loc = w.get("best_oa_location") or w.get("primary_location") or {}
        out.append({
            "source": "openalex",
            "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
            "title": w.get("title") or "",
            "year": w.get("publication_year"),
            "journal": ((w.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
            "abstract": abstract,
            "authors": ", ".join(a["author"]["display_name"]
                                 for a in (w.get("authorships") or [])[:6]),
            "cited_by": w.get("cited_by_count", 0),
            "oa_pdf": loc.get("pdf_url") or "",
        })
    return out


def s_crossref(query: str, n: int = 25, year_min: int | None = None) -> list[dict]:
    params = {"query.bibliographic": query, "rows": min(n, 50),
              "mailto": CFG.env("OPENALEX_EMAIL", "")}
    if year_min:
        params["filter"] = f"from-pub-date:{year_min}-01-01"
    j = _get("https://api.crossref.org/works", params)
    out = []
    for w in (((j or {}).get("message") or {}).get("items") or []):
        title = (w.get("title") or [""])[0]
        yr = None
        for k in ("published-print", "published-online", "issued"):
            parts = (w.get(k) or {}).get("date-parts") or []
            if parts and parts[0] and parts[0][0]:
                yr = parts[0][0]
                break
        out.append({"source": "crossref", "doi": w.get("DOI", ""), "title": title,
                    "year": yr, "journal": (w.get("container-title") or [""])[0],
                    "abstract": re.sub(r"<[^>]+>", "", w.get("abstract", ""))[:4000],
                    "authors": ", ".join(
                        f"{a.get('given','')} {a.get('family','')}".strip()
                        for a in (w.get("author") or [])[:6]),
                    "cited_by": w.get("is-referenced-by-count", 0), "oa_pdf": ""})
    return out


def s_europepmc(query: str, n: int = 25, year_min: int | None = None) -> list[dict]:
    qq = query + (f" AND PUB_YEAR:[{year_min} TO 3000]" if year_min else "")
    j = _get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
             {"query": qq, "format": "json", "pageSize": min(n, 50), "resultType": "core"})
    out = []
    for w in (((j or {}).get("resultList") or {}).get("result") or []):
        out.append({"source": "europepmc", "doi": w.get("doi", ""),
                    "title": w.get("title", ""),
                    "year": int(w["pubYear"]) if str(w.get("pubYear", "")).isdigit() else None,
                    "journal": w.get("journalTitle", ""),
                    "abstract": (w.get("abstractText") or "")[:4000],
                    "authors": w.get("authorString", ""),
                    "cited_by": w.get("citedByCount", 0),
                    "oa_pdf": ("https://europepmc.org/articles/%s?pdf=render" % w["pmcid"])
                    if w.get("pmcid") and w.get("isOpenAccess") == "Y" else ""})
    return out


def s_semanticscholar(query: str, n: int = 20, year_min: int | None = None) -> list[dict]:
    h = {}
    if CFG.env("SEMANTIC_SCHOLAR_API_KEY"):
        h["x-api-key"] = CFG.env("SEMANTIC_SCHOLAR_API_KEY")
    params = {"query": query, "limit": min(n, 40),
              "fields": "title,abstract,year,venue,externalIds,openAccessPdf,citationCount"}
    if year_min:
        params["year"] = f"{year_min}-"
    j = _get("https://api.semanticscholar.org/graph/v1/paper/search", params, h)
    out = []
    for w in ((j or {}).get("data") or []):
        out.append({"source": "s2",
                    "doi": ((w.get("externalIds") or {}).get("DOI") or ""),
                    "title": w.get("title", ""), "year": w.get("year"),
                    "journal": w.get("venue", ""),
                    "abstract": (w.get("abstract") or "")[:4000], "authors": "",
                    "cited_by": w.get("citationCount", 0),
                    "oa_pdf": (w.get("openAccessPdf") or {}).get("url", "")})
    return out


SOURCES = {"openalex": s_openalex, "crossref": s_crossref,
           "europepmc": s_europepmc, "semanticscholar": s_semanticscholar}


def search_many(queries: list[str], per_source: int = 20,
                sources: list[str] | None = None,
                year_min: int | None = None) -> list[dict]:
    """多源多式检索 + 去重（DOI 优先，其次归一化标题）。"""
    sources = sources or list(CFG.get("literature.sources", list(SOURCES)))
    seen: dict[str, dict] = {}
    for qi, qtext in enumerate(queries):
        for sname in sources:
            fn = SOURCES.get(sname)
            if not fn:
                continue
            try:
                res = fn(qtext, per_source, year_min)
            except Exception:  # noqa: BLE001
                res = []
            for w in res:
                if not w.get("title"):
                    continue
                k = (w.get("doi") or "").lower() or _norm_title(w["title"])
                if k in seen:
                    if not seen[k].get("abstract") and w.get("abstract"):
                        seen[k]["abstract"] = w["abstract"]
                    if not seen[k].get("oa_pdf") and w.get("oa_pdf"):
                        seen[k]["oa_pdf"] = w["oa_pdf"]
                    seen[k]["found_by"] = sorted(set(seen[k].get("found_by", []) + [sname]))
                    continue
                w["found_by"] = [sname]
                w["matched_query"] = qtext
                seen[k] = w
            time.sleep(0.25)
    return list(seen.values())


# ---------------- 相关性打分 ----------------
SCORE_SYS = """你在为一个 NF/RO 膜去除有机微污染物的知识库筛选文献。
对每篇候选文献按题名+摘要打分。

输出 JSON：{"scored": [{"i": 序号, "relevance": 0-10, "reason": 一句话,
 "evidence_type": "supports|contradicts|neutral|method|out_of_scope"}]}

评分口径：
- 9-10：直接给出 NF/RO 对特定有机微污染物截留的机理证据或定量数据
- 6-8：相关但间接（膜表征、其它污染物、模型方法）
- 3-5：外领域但概念可迁移（药物化学构象、色谱保留、沸石择形…）—— 这类要保留，注明 reason
- 0-2：无关
标为 contradicts 的（与主流结论相反、报告无相关性、指出模型高估）优先级要**提高**，
因为矛盾证据的信息量最大。"""


def score_relevance(works: list[dict], question: str, agent: str = "bowen") -> list[dict]:
    if not works:
        return []
    # 候选相关性初筛属于关键科研判断，按用户口径固定使用 V4-Pro。
    llm = LLM(agent, model=CFG.llm_model, fallbacks=[CFG.llm_model],
              usage_kind="literature_relevance")
    out: list[dict] = []
    B = 20
    for s in range(0, len(works), B):
        batch = works[s:s + B]
        listing = "\n\n".join(
            f"[{s+i}] {w['title']} ({w.get('year')}, {w.get('journal','')})\n"
            f"{(w.get('abstract') or '')[:700]}" for i, w in enumerate(batch))
        try:
            j = llm.ask_json(SCORE_SYS, f"研究问题：{question}\n\n候选：\n{listing}",
                             temperature=0.1, thinking=False)
            for sc in (j.get("scored") or []):
                i = int(sc.get("i", -1))
                if 0 <= i < len(works):
                    works[i]["relevance"] = float(sc.get("relevance", 0))
                    works[i]["score_reason"] = sc.get("reason", "")
                    works[i]["evidence_type"] = sc.get("evidence_type", "neutral")
        except Exception as e:  # noqa: BLE001
            for w in batch:
                # 关键初筛不允许静默改用便宜模型，也不把失败伪装成中等相关。
                w.setdefault("relevance", 0.0)
                w.setdefault("evidence_type", "unscored")
                w["score_error"] = f"V4-Pro 相关性评分失败: {type(e).__name__}: {e}"[:300]
    for w in works:
        w.setdefault("relevance", 0.0)
        w.setdefault("evidence_type", "neutral")
        # 矛盾证据加权
        if w["evidence_type"] == "contradicts":
            w["relevance"] = min(10.0, w["relevance"] + 1.5)
        out.append(w)
    out.sort(key=lambda x: -x["relevance"])
    return out


# ---------------- 全文获取 ----------------
def try_fetch_pdf(work: dict) -> Path | None:
    """依次尝试：OA 直链 -> Unpaywall -> Elsevier。失败返回 None（降级为摘要级证据）。"""
    dest_dir = CFG.new_pdf_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    wid = work_id(work.get("doi", ""), work.get("title", ""))
    dest = dest_dir / f"{wid}.pdf"
    if dest.exists() and dest.stat().st_size > 20000:
        return dest

    urls: list[tuple[str, dict]] = []
    if work.get("oa_pdf"):
        urls.append((work["oa_pdf"], {}))
    doi = (work.get("doi") or "").strip()
    if doi:
        j = _get(f"https://api.unpaywall.org/v2/{doi}",
                 {"email": CFG.env("UNPAYWALL_EMAIL", CFG.env("OPENALEX_EMAIL", ""))})
        loc = (j or {}).get("best_oa_location") or {}
        if loc.get("url_for_pdf"):
            urls.append((loc["url_for_pdf"], {}))
        if CFG.env("ELSEVIER_API_KEY"):
            h = {"X-ELS-APIKey": CFG.env("ELSEVIER_API_KEY"), "Accept": "application/pdf"}
            if CFG.env("ELSEVIER_INSTTOKEN"):
                h["X-ELS-Insttoken"] = CFG.env("ELSEVIER_INSTTOKEN")
            urls.append((f"https://api.elsevier.com/content/article/doi/{doi}", h))

    for url, hh in urls:
        try:
            h = dict(UA)
            h.update(hh)
            r = requests.get(url, headers=h, timeout=60, allow_redirects=True)
            if r.status_code == 200 and r.content[:5] == b"%PDF-" and len(r.content) > 20000:
                dest.write_bytes(r.content)
                return dest
        except Exception:  # noqa: BLE001
            continue
    return None


def enqueue(works: list[dict], fetch_fulltext: bool = True,
            request_id: int | None = None) -> dict:
    """把选中的文献写入 papers 表并排队。返回统计。"""
    dedup.refresh_identities()
    added, skipped, fulltext = 0, 0, 0
    added_ids: list[str] = []
    duplicate_ids: list[str] = []
    fulltext_ids: list[str] = []
    abstract_ids: list[str] = []
    for w in works:
        wid = work_id(w.get("doi", ""), w.get("title", ""))
        duplicate = dedup.find_duplicate(w.get("doi", ""), w.get("title", ""),
                                         exclude_id=wid)
        existing = db.q1("SELECT id FROM papers WHERE id=?", (wid,))
        if existing or duplicate:
            skipped += 1
            duplicate_ids.append(str(existing["id"] if existing else duplicate["id"]))
            continue
        path = try_fetch_pdf(w) if fetch_fulltext else None
        level = "fulltext" if path else "abstract"
        if path:
            fulltext += 1
            fulltext_ids.append(wid)
        else:
            abstract_ids.append(wid)
        keys = dedup.identity(w.get("doi", ""), w.get("title", ""), path or "")
        db.ex("INSERT INTO papers(id,source,doi,title,authors,year,journal,abstract,path,"
              "evidence_level,pinned,status,meta,added_at,doi_key,title_key,content_hash) "
              "VALUES(?,?,?,?,?,?,?,?,?,?,0,'queued',?,strftime('%s','now'),?,?,?)",
              (wid, w.get("source", "search"), w.get("doi", ""), w.get("title", ""),
               w.get("authors", ""), w.get("year"), w.get("journal", ""),
               w.get("abstract", ""), str(path) if path else "",
               level, json.dumps({k: w.get(k) for k in
                                  ("relevance", "score_reason", "evidence_type",
                                   "found_by", "matched_query", "cited_by",
                                   "request_id" if request_id else "source")},
                                 ensure_ascii=False), keys["doi_key"], keys["title_key"],
               keys["content_hash"]))
        db.task_add("ingest_pdf", wid, (w.get("title") or wid)[:120])
        added += 1
        added_ids.append(wid)
    return {"added": added, "skipped_duplicate": skipped, "fulltext_obtained": fulltext,
            "abstract_only": added - fulltext, "added_ids": added_ids,
            "duplicate_ids": list(dict.fromkeys(duplicate_ids)),
            "fulltext_ids": fulltext_ids, "abstract_ids": abstract_ids}
