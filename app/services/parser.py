from __future__ import annotations

import json
import re
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import httpx
from bs4 import BeautifulSoup

from app.config import Settings, get_settings


@dataclass(slots=True)
class ParsedBlock:
    kind: str
    text: str
    section: str | None = None
    page: int | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    title: str | None
    blocks: list[ParsedBlock]
    parser: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DocumentParser:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def parse(self, path: Path, content_type: str | None = None) -> ParsedDocument:
        if path.suffix.lower() in {".html", ".htm"} or content_type == "text/html":
            return self._parse_html(path)
        if path.suffix.lower() == ".xml" or content_type == "application/xml":
            return self._parse_xml(path)
        if self.settings.mineru_command:
            try:
                return self._parse_mineru(path)
            except Exception as exc:  # parser fallback is intentional
                warning = f"MinerU 失败，已回退: {exc}"
            else:
                warning = ""
        else:
            warning = "未配置 MinerU，使用 GROBID/PyMuPDF 回退解析"
        if self.settings.grobid_url:
            try:
                parsed = self._parse_grobid(path)
                parsed.warnings.append(warning)
                return parsed
            except Exception as exc:
                warning += f"；GROBID 失败: {exc}"
        parsed = self._parse_pymupdf(path)
        parsed.warnings.append(warning)
        return parsed

    def _parse_grobid(self, path: Path) -> ParsedDocument:
        with path.open("rb") as handle:
            response = httpx.post(
                f"{self.settings.grobid_url.rstrip('/')}/api/processFulltextDocument",
                files={"input": (path.name, handle, "application/pdf")},
                data={"consolidateHeader": "1", "includeRawCitations": "1"},
                timeout=self.settings.parser_timeout_seconds,
            )
        response.raise_for_status()
        return self._tei_to_document(response.content, "grobid")

    def _parse_mineru(self, path: Path) -> ParsedDocument:
        with tempfile.TemporaryDirectory(prefix="nf-mineru-") as temp_dir:
            output = Path(temp_dir) / "output"
            output.mkdir()
            args = [part.format(input=str(path), output=str(output)) for part in shlex.split(self.settings.mineru_command or "")]
            completed = subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.settings.parser_timeout_seconds,
            )
            json_files = list(output.rglob("*.json"))
            markdown_files = list(output.rglob("*.md"))
            if json_files:
                data = json.loads(json_files[0].read_text(encoding="utf-8"))
                blocks = []
                items = data if isinstance(data, list) else data.get("pdf_info", [])
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        blocks.append(ParsedBlock(kind=item.get("type", "paragraph"), text=text, page=item.get("page_idx")))
                if blocks:
                    return ParsedDocument(title=None, blocks=blocks, parser="mineru")
            if markdown_files:
                return self._markdown_to_document(markdown_files[0].read_text(encoding="utf-8"), "mineru")
            raise RuntimeError(f"MinerU 未生成可识别 JSON/Markdown: {completed.stderr[-500:]}")

    def _parse_pymupdf(self, path: Path) -> ParsedDocument:
        import fitz

        pdf = fitz.open(path)
        blocks: list[ParsedBlock] = []
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:
                blocks.append(ParsedBlock(kind="page", text=text, page=page_index + 1, section=f"Page {page_index + 1}"))
        if not blocks:
            raise RuntimeError("PDF 无可提取文本；请配置 MinerU 处理扫描版或复杂排版 PDF")
        title = (pdf.metadata or {}).get("title") or None
        return ParsedDocument(title=title, blocks=blocks, parser="pymupdf")

    def _parse_xml(self, path: Path) -> ParsedDocument:
        data = path.read_bytes()
        if b"<TEI" in data[:5000] or b"tei-c.org" in data[:5000]:
            return self._tei_to_document(data, "tei-xml")
        root = ET.fromstring(data)
        title = self._first_text(root, ("title", "article-title"))
        blocks: list[ParsedBlock] = []
        for element in root.iter():
            tag = element.tag.rsplit("}", 1)[-1].lower()
            if tag in {"section", "sec"}:
                section_title = self._first_text(element, ("section-title", "title"))
                paragraphs = [self._all_text(child) for child in element.iter() if child.tag.rsplit("}", 1)[-1].lower() in {"para", "p"}]
                text = "\n".join(item for item in paragraphs if item)
                if text:
                    blocks.append(ParsedBlock(kind="section", text=text, section=section_title))
            elif tag in {"table", "table-wrap"}:
                text = self._all_text(element)
                if text:
                    blocks.append(ParsedBlock(kind="table", text=text, label=element.attrib.get("id")))
            elif tag in {"figure", "fig"}:
                text = self._all_text(element)
                if text:
                    blocks.append(ParsedBlock(kind="figure_caption", text=text, label=element.attrib.get("id")))
        if not blocks:
            text = self._all_text(root)
            blocks = [ParsedBlock(kind="document", text=text)] if text else []
        return ParsedDocument(title=title, blocks=blocks, parser="sciencedirect-xml")

    def _parse_html(self, path: Path) -> ParsedDocument:
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        title = None
        for meta in soup.find_all("meta"):
            key = str(meta.get("name") or meta.get("property") or "").casefold()
            if key in {"citation_title", "dc.title", "og:title"} and meta.get("content"):
                title = self._clean_html_text(str(meta["content"]))
                break
        if not title:
            node = soup.find("h1") or soup.find("title")
            title = self._clean_html_text(node.get_text(" ", strip=True)) if node else None

        for tag in soup.find_all(["script", "style", "noscript", "nav", "header", "footer", "form", "button", "svg"]):
            tag.decompose()
        roots = []
        for selector in (
            "article",
            "main",
            "[role='main']",
            ".article-content",
            ".article-body",
            "#article-body",
            ".entry-content",
            ".post-content",
            ".fulltext",
            "#fulltext",
        ):
            roots.extend(soup.select(selector))
        roots.append(soup.body or soup)
        root = max(roots, key=lambda item: len(self._clean_html_text(item.get_text(" ", strip=True))))

        blocks: list[ParsedBlock] = []
        section: str | None = None
        seen_text: set[tuple[str, str | None, str]] = set()
        for node in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "figcaption"]):
            if node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                heading = self._clean_html_text(node.get_text(" ", strip=True))
                if heading and heading != title:
                    section = heading
                continue
            if node.find_parent("table") and node.name != "table":
                continue
            if node.name == "table":
                rows = []
                for row in node.find_all("tr"):
                    cells = [self._clean_html_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
                    if any(cells):
                        rows.append("\t".join(cells))
                text = "\n".join(rows) or self._clean_html_text(node.get_text(" ", strip=True))
                kind = "table"
                label = str(node.get("id") or "") or None
            elif node.name == "figcaption":
                text = self._clean_html_text(node.get_text(" ", strip=True))
                kind = "figure_caption"
                figure = node.find_parent("figure")
                label = str(figure.get("id") or "") or None if figure else None
            else:
                text = self._clean_html_text(node.get_text(" ", strip=True))
                kind = "paragraph"
                label = None
            if len(text) < 2:
                continue
            key = (kind, section, text)
            if key in seen_text:
                continue
            seen_text.add(key)
            blocks.append(ParsedBlock(kind=kind, text=text, section=section, label=label))
        if not blocks:
            text = self._clean_html_text(root.get_text(" ", strip=True))
            if text:
                blocks.append(ParsedBlock(kind="document", text=text, section="Full text"))
        if not blocks:
            raise RuntimeError("HTML 页面没有可提取正文")
        return ParsedDocument(title=title, blocks=blocks, parser="public-html")

    def _tei_to_document(self, data: bytes, parser_name: str) -> ParsedDocument:
        root = ET.fromstring(data)
        title = self._first_text(root, ("title",))
        blocks: list[ParsedBlock] = []
        for div in root.iter():
            if div.tag.rsplit("}", 1)[-1] != "div":
                continue
            head = next((self._all_text(child) for child in div if child.tag.rsplit("}", 1)[-1] == "head"), None)
            paragraphs = [self._all_text(child) for child in div if child.tag.rsplit("}", 1)[-1] == "p"]
            if paragraphs:
                blocks.append(ParsedBlock(kind="section", text="\n".join(paragraphs), section=head))
        for figure in (item for item in root.iter() if item.tag.rsplit("}", 1)[-1] == "figure"):
            kind = "table" if figure.attrib.get("type") == "table" else "figure_caption"
            blocks.append(ParsedBlock(kind=kind, text=self._all_text(figure), label=figure.attrib.get("{http://www.w3.org/XML/1998/namespace}id")))
        return ParsedDocument(title=title, blocks=[block for block in blocks if block.text], parser=parser_name)

    @staticmethod
    def _markdown_to_document(text: str, parser_name: str) -> ParsedDocument:
        blocks = []
        section = None
        buffer: list[str] = []
        for line in text.splitlines():
            if line.startswith("#"):
                if buffer:
                    blocks.append(ParsedBlock(kind="section", text="\n".join(buffer), section=section))
                    buffer = []
                section = line.lstrip("# ").strip()
            else:
                buffer.append(line)
        if buffer:
            blocks.append(ParsedBlock(kind="section", text="\n".join(buffer), section=section))
        return ParsedDocument(title=None, blocks=blocks, parser=parser_name)

    @staticmethod
    def _all_text(element: ET.Element) -> str:
        return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()

    @staticmethod
    def _clean_html_text(value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip()

    @classmethod
    def _first_text(cls, root: ET.Element, tags: tuple[str, ...]) -> str | None:
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1].lower() in tags:
                value = cls._all_text(element)
                if value:
                    return value
        return None


def chunk_document(parsed: ParsedDocument, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for block in parsed.blocks:
        text = re.sub(r"\s+", " ", block.text).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            if end < len(text):
                boundary = max(text.rfind("。", start, end), text.rfind(". ", start, end), text.rfind("; ", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary + 1
            chunks.append(
                {
                    "text": text[start:end],
                    "section": block.section or block.kind,
                    "page_start": block.page,
                    "page_end": block.page,
                    "metadata": {"kind": block.kind, "label": block.label, **block.metadata},
                }
            )
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
