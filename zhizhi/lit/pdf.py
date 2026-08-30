"""PDF 摄取：文本抽取 -> 章节识别 -> 分块。"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

SECTION_PAT = re.compile(
    r"^\s*(?:\d+\.?\d*\s+)?("
    r"abstract|introduction|background|materials?\s+and\s+methods?|methods?|"
    r"experimental(?:\s+section)?|theory|model(?:ling|ing)?|"
    r"results?(?:\s+and\s+discussion)?|discussion|conclusions?|"
    r"acknowledge?ments?|references?|supporting\s+information|appendix"
    r")\s*$", re.I)

# 参考文献段落对知识抽取无用，且会污染检索
STOP_SECTIONS = {"references", "reference", "acknowledgment", "acknowledgement",
                 "acknowledgments", "acknowledgements"}


def extract_pages(path: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            txt = page.get_text("text") or ""
            out.append((i + 1, txt))
    return out


def _clean(t: str) -> str:
    t = t.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("\xad", "")
    t = re.sub(r"-\n(?=[a-z])", "", t)          # 断字连接
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def split_sections(pages: list[tuple[int, str]]) -> list[dict]:
    """返回 [{section, page, text}]，遇 References 即停止收录正文。"""
    blocks: list[dict] = []
    current = "front"
    stopped = False
    for pno, raw in pages:
        text = _clean(raw)
        if not text:
            continue
        buf: list[str] = []
        for line in text.split("\n"):
            m = SECTION_PAT.match(line.strip())
            if m:
                if buf:
                    blocks.append({"section": current, "page": pno,
                                   "text": "\n".join(buf).strip()})
                    buf = []
                current = m.group(1).lower().strip()
                stopped = current.split()[0] in STOP_SECTIONS
                continue
            if not stopped:
                buf.append(line)
        if buf:
            blocks.append({"section": current, "page": pno,
                           "text": "\n".join(buf).strip()})
    return [b for b in blocks if len(b["text"]) > 40]


def chunk(blocks: list[dict], target_chars: int = 3200, overlap: int = 480) -> list[dict]:
    """章节内滑窗分块；块不跨章节，保留页码。"""
    out: list[dict] = []
    for b in blocks:
        t = b["text"]
        if len(t) <= target_chars:
            out.append({**b, "idx": len(out)})
            continue
        start = 0
        while start < len(t):
            end = min(start + target_chars, len(t))
            if end < len(t):
                cut = t.rfind(". ", start + target_chars // 2, end)
                if cut > 0:
                    end = cut + 1
            out.append({"section": b["section"], "page": b["page"],
                        "text": t[start:end].strip(), "idx": len(out)})
            if end >= len(t):
                break
            start = max(end - overlap, start + 1)
    return [c for c in out if len(c["text"]) > 60]


def ingest_file(path: Path, target_chars: int = 3200) -> dict:
    """返回 {chunks, n_chars, needs_ocr, title_guess}"""
    pages = extract_pages(path)
    total = sum(len(p[1]) for p in pages)
    needs_ocr = total < 500 or (len(pages) > 0 and total / max(len(pages), 1) < 120)
    blocks = split_sections(pages)
    chunks = chunk(blocks, target_chars=target_chars)
    title = ""
    if pages:
        head = [l.strip() for l in _clean(pages[0][1]).split("\n") if len(l.strip()) > 12]
        for line in head[:6]:
            if not re.search(r"(journal|elsevier|doi|www\.|received|©|http)", line, re.I):
                title = line
                break
    return {"chunks": chunks, "n_chars": total, "n_pages": len(pages),
            "needs_ocr": needs_ocr, "title_guess": title[:300],
            "head_text": _clean(pages[0][1])[:4000] if pages else ""}
