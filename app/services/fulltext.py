from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import threading
import time
from dataclasses import dataclass
from urllib.parse import quote, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree as ET

import httpx
from bs4 import BeautifulSoup

from app.config import Settings, get_settings
from app.services.dedupe import extract_doi, normalize_doi, sha256_bytes


class FullTextUnavailable(RuntimeError):
    pass


class ElsevierAuthenticationError(ValueError):
    """The API key is missing, invalid, disabled, or not enabled for this API."""


class ElsevierEntitlementError(ValueError):
    """The API key is valid but the caller is not entitled to the requested FULL view."""


@dataclass(slots=True)
class FullTextResult:
    content: bytes
    content_type: str
    extension: str
    source: str
    url: str
    license: str | None
    sha256: str
    title: str | None = None
    doi: str | None = None


@dataclass(slots=True)
class FetchedResource:
    content: bytes
    content_type: str
    status_code: int
    url: str
    headers: dict[str, str]


@dataclass(slots=True)
class HtmlPageInfo:
    title: str | None
    doi: str | None
    license: str | None
    pdf_urls: list[str]
    article_chars: int
    looks_like_fulltext: bool


_DOMAIN_SCHEDULE: dict[str, float] = {}
_DOMAIN_SCHEDULE_LOCK = threading.Lock()
_ROBOTS_CACHE: dict[str, tuple[float, RobotFileParser | None]] = {}


class FullTextResolver:
    """Resolve legally reachable PDF/XML/HTML without login or paywall bypasses."""

    _redirect_codes = {301, 302, 303, 307, 308}
    _article_signals = (
        "introduction",
        "materials and methods",
        "materials & methods",
        "experimental",
        "methodology",
        "results",
        "discussion",
        "conclusion",
        "引言",
        "材料与方法",
        "实验",
        "结果",
        "讨论",
        "结论",
    )

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        contact = self.settings.unpaywall_email or self.settings.openalex_email or "noreply@example.org"
        self.user_agent = f"NF-Atlas/0.2 (+mailto:{contact}; public scholarly full-text resolver)"
        self.client = httpx.AsyncClient(
            timeout=90,
            follow_redirects=False,
            headers={"User-Agent": self.user_agent},
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def resolve_and_download(
        self,
        *,
        doi: str | None,
        hinted_url: str | None = None,
        landing_url: str | None = None,
        hinted_open_access: bool = False,
    ) -> FullTextResult:
        del hinted_open_access  # Content checks are safer than a possibly stale metadata flag.
        errors: list[str] = []
        normalized_doi = normalize_doi(doi)

        if normalized_doi and self.settings.direct_web_fetch:
            try:
                result = await self._from_europe_pmc(normalized_doi)
                if result:
                    return result
            except (httpx.HTTPError, ValueError) as exc:
                errors.append(self._safe_error("Europe PMC 开放全文", exc))

        if self.settings.direct_web_fetch:
            direct_urls: list[tuple[str, str]] = []
            for label, url in (("候选全文网址", hinted_url), ("论文落地页", landing_url)):
                if url and url not in {item[1] for item in direct_urls}:
                    direct_urls.append((label, url))
            for label, url in direct_urls:
                try:
                    return await self._from_public_page(url, "direct-public")
                except (httpx.HTTPError, ValueError) as exc:
                    errors.append(self._safe_error(label, exc))

        if normalized_doi and self.settings.unpaywall_email:
            try:
                result = await self._from_unpaywall(normalized_doi)
                if result:
                    return result
            except (httpx.HTTPError, ValueError) as exc:
                errors.append(self._safe_error("Unpaywall", exc))

        elsevier_pii = self.extract_elsevier_pii(hinted_url, landing_url)
        if (normalized_doi or elsevier_pii) and self.settings.elsevier_api_key:
            try:
                if normalized_doi:
                    result = await self._from_elsevier(normalized_doi, identifier_type="doi")
                else:
                    result = await self._from_elsevier(elsevier_pii or "", identifier_type="pii")
                if result:
                    return result
            except (httpx.HTTPError, ValueError) as exc:
                errors.append(self._safe_error("ScienceDirect TDM", exc))

        if normalized_doi and self.settings.direct_web_fetch:
            try:
                return await self._from_public_page(
                    f"https://doi.org/{quote(normalized_doi, safe='/()')}",
                    "doi-public-page",
                )
            except (httpx.HTTPError, ValueError) as exc:
                errors.append(self._safe_error("DOI 官网落地页", exc))

        detail = "；".join(errors) if errors else "没有发现公开可读全文或已授权 TDM 全文"
        raise FullTextUnavailable(detail)

    async def resolve_public_source(self, source: str) -> FullTextResult:
        """Resolve a user-confirmed public DOI/article/PDF URL without literature APIs."""
        value = source.strip()
        doi = extract_doi(value)
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"}:
            url = value
        elif doi:
            url = f"https://doi.org/{quote(doi, safe='/()')}"
        else:
            raise ValueError("请输入完整的 http(s) 网址或有效 DOI")
        if not self.settings.direct_web_fetch:
            raise ValueError("DIRECT_WEB_FETCH 已关闭")
        if doi:
            try:
                result = await self._from_europe_pmc(doi)
                if result:
                    return result
            except (httpx.HTTPError, ValueError):
                pass
        return await self._from_public_page(url, "direct-public")

    async def _from_europe_pmc(self, doi: str) -> FullTextResult | None:
        """Use Europe PMC's official OA API when publisher robots block direct crawling."""
        normalized = normalize_doi(doi)
        if not normalized:
            return None
        search = await self.client.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query": f'DOI:"{normalized}"',
                "format": "json",
                "resultType": "core",
                "pageSize": 5,
            },
        )
        search.raise_for_status()
        rows = (search.json().get("resultList") or {}).get("result") or []
        match = next(
            (
                item
                for item in rows
                if normalize_doi(item.get("doi")) == normalized and item.get("pmcid")
            ),
            None,
        )
        if not match:
            return None
        pmcid = str(match["pmcid"]).upper()
        response = await self.client.get(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{quote(pmcid, safe='')}/fullTextXML"
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        content = response.content
        if not content.lstrip().startswith(b"<") or b"<article" not in content[:5000]:
            raise ValueError("Europe PMC 返回内容不是 JATS 全文 XML")
        return FullTextResult(
            content=content,
            content_type="application/xml",
            extension="xml",
            source="europe-pmc-xml",
            url=self._safe_url(str(response.url)),
            license=match.get("license") or "open-access",
            sha256=sha256_bytes(content),
            title=match.get("title"),
            doi=normalized,
        )

    async def _from_unpaywall(self, doi: str) -> FullTextResult | None:
        response = await self.client.get(
            f"https://api.unpaywall.org/v2/{quote(doi, safe='')}",
            params={"email": self.settings.unpaywall_email},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        locations = [data.get("best_oa_location"), *(data.get("oa_locations") or [])]
        seen: set[str] = set()
        for location in locations:
            if not location:
                continue
            url = location.get("url_for_pdf") or location.get("url_for_landing_page")
            if not url or url in seen:
                continue
            seen.add(url)
            try:
                return await self._from_public_page(url, "unpaywall", location.get("license"))
            except (httpx.HTTPError, ValueError):
                continue
        return None

    async def _from_elsevier(
        self,
        identifier: str,
        *,
        identifier_type: str = "doi",
    ) -> FullTextResult | None:
        api_key = self.settings.elsevier_api_key
        if not api_key:
            return None
        if identifier_type not in {"doi", "pii"}:
            raise ValueError("Elsevier 文献标识类型必须是 DOI 或 PII")
        value = identifier.strip()
        if not value:
            return None
        url = f"https://api.elsevier.com/content/article/{identifier_type}/{quote(value, safe='')}"
        headers = {
            "X-ELS-APIKey": api_key.get_secret_value(),
            "X-ELS-ResourceVersion": "new",
            "Accept": "text/xml",
        }
        if self.settings.elsevier_insttoken:
            headers["X-ELS-Insttoken"] = self.settings.elsevier_insttoken.get_secret_value()
        # FULL is deliberate. Without it Elsevier may return META_ABS XML for an
        # unentitled caller, which must never be saved and labelled as full text.
        response = await self.client.get(url, params={"view": "FULL"}, headers=headers)
        if response.status_code == 404:
            return None
        if response.status_code == 401:
            service_code = self._elsevier_service_code(response)
            if service_code == "AUTHORIZATION_ERROR":
                raise ElsevierEntitlementError(
                    "HTTP 401（AUTHORIZATION_ERROR）：当前 Key/服务器出口没有 Article Retrieval FULL 权限"
                )
            raise ElsevierAuthenticationError(
                f"HTTP 401{f'（{service_code}）' if service_code else ''}："
                "Elsevier API Key 无效、已停用，或未启用 Article Retrieval API"
            )
        if response.status_code == 403:
            service_code = self._elsevier_service_code(response)
            raise ElsevierEntitlementError(
                f"HTTP 403{f'（{service_code}）' if service_code else ''}："
                "当前 Key/服务器 IP/Insttoken 配置不足，无法取得该文献的 ScienceDirect FULL 正文"
            )
        if response.status_code == 429:
            raise ValueError("HTTP 429：Elsevier API 配额或速率限制，请稍后重试")
        response.raise_for_status()
        content = response.content
        if not content.lstrip().startswith(b"<"):
            raise ValueError("ScienceDirect 返回内容不是 XML")
        if not self.is_elsevier_fulltext_xml(content):
            raise ElsevierEntitlementError(
                "Elsevier 只返回了题录/摘要 XML（META_ABS），未返回 FULL 正文；"
                "请从学校订阅 IP 发起请求或配置官方 Insttoken"
            )
        return FullTextResult(
            content=content,
            content_type="text/xml",
            extension="xml",
            source="sciencedirect-tdm",
            url=self._safe_url(str(response.url)),
            license=None,
            sha256=sha256_bytes(content),
            doi=value if identifier_type == "doi" else None,
        )

    @staticmethod
    def _elsevier_service_code(response: httpx.Response) -> str | None:
        try:
            status = (response.json().get("service-error") or {}).get("status") or {}
            return str(status.get("statusCode") or "").strip() or None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def extract_elsevier_pii(*urls_or_identifiers: str | None) -> str | None:
        """Extract a conservative Elsevier PII from a ScienceDirect/API URL."""
        for value in urls_or_identifiers:
            if not value:
                continue
            match = re.search(r"(?i)(?:/pii/|\b(?:pii|scidir):)([a-z0-9]{10,40})", value)
            if match:
                return match.group(1).upper()
        return None

    @staticmethod
    def is_elsevier_fulltext_xml(content: bytes) -> bool:
        """Reject META/META_ABS payloads that contain no article body."""
        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            raise ValueError("ScienceDirect 返回的 XML 无法解析") from exc

        def local_name(tag: object) -> str:
            return str(tag).rsplit("}", 1)[-1].casefold()

        strong_body_tags = {"originaltext", "body", "sections"}
        article_tags = {"article", "doc"}
        section_tags = {"section", "sec"}
        article_nodes = []
        has_section = False
        for element in root.iter():
            name = local_name(element.tag)
            if name in strong_body_tags:
                body_text = " ".join(part.strip() for part in element.itertext() if part.strip())
                if len(body_text) >= 500:
                    return True
            if name in article_tags:
                article_nodes.append(element)
            elif name in section_tags:
                has_section = True
        if has_section:
            for element in article_nodes:
                article_text = " ".join(part.strip() for part in element.itertext() if part.strip())
                if len(article_text) >= 1000:
                    return True
        return False

    async def _from_public_page(
        self,
        url: str,
        source: str,
        license_value: str | None = None,
        *,
        depth: int = 0,
        seen: set[str] | None = None,
    ) -> FullTextResult:
        if depth > 2:
            raise ValueError("公开页面中的全文链接跳转层级过深")
        seen = seen or set()
        canonical = self._loop_key(url)
        if canonical in seen:
            raise ValueError("公开全文链接发生循环跳转")
        seen.add(canonical)
        fetched = await self._fetch_bytes(
            url,
            accept="application/pdf,application/xml,text/xml,text/html;q=0.9",
            check_robots=True,
        )
        content = fetched.content
        content_type = fetched.content_type
        final_url = fetched.url

        if content.startswith(b"%PDF-"):
            return self._make_result(content, "application/pdf", "pdf", source, final_url, license_value)
        if content.lstrip().startswith(b"<") and (
            "xml" in content_type or b"<full-text" in content[:2000] or b"<article" in content[:2000]
        ) and "html" not in content_type:
            return self._make_result(content, "application/xml", "xml", source, final_url, license_value)
        if "html" not in content_type and b"<html" not in content[:3000].lower():
            raise ValueError(f"返回内容不是 PDF/XML/HTML（Content-Type={content_type or 'unknown'}）")

        info = self.inspect_html(content, final_url, self.settings.direct_html_min_chars)
        link_errors: list[str] = []
        for pdf_url in info.pdf_urls[:12]:
            if self._loop_key(pdf_url) in seen:
                continue
            try:
                nested = await self._from_public_page(
                    pdf_url,
                    source,
                    info.license or license_value,
                    depth=depth + 1,
                    seen=seen,
                )
                if nested.extension in {"pdf", "xml"}:
                    nested.title = nested.title or info.title
                    nested.doi = nested.doi or info.doi
                    return nested
            except (httpx.HTTPError, ValueError) as exc:
                link_errors.append(self._safe_error("页面 PDF 链接", exc))

        if info.looks_like_fulltext:
            result = self._make_result(
                content,
                "text/html",
                "html",
                source,
                final_url,
                info.license or license_value,
            )
            result.title = info.title
            result.doi = info.doi
            return result
        suffix = f"；PDF 链接失败：{'；'.join(link_errors[:3])}" if link_errors else ""
        raise ValueError(f"页面是 HTML，但未检测到足够的论文正文（正文约 {info.article_chars} 字符）{suffix}")

    def _make_result(
        self,
        content: bytes,
        content_type: str,
        extension: str,
        source: str,
        url: str,
        license_value: str | None,
    ) -> FullTextResult:
        source_name = source
        if source in {"direct-public", "doi-public-page"}:
            source_name = f"{source}-{extension}"
        elif extension == "html" and not source.endswith("-html"):
            source_name = f"{source}-html"
        return FullTextResult(
            content=content,
            content_type=content_type,
            extension=extension,
            source=source_name,
            url=self._safe_url(url),
            license=license_value,
            sha256=sha256_bytes(content),
        )

    async def _fetch_bytes(self, url: str, *, accept: str, check_robots: bool) -> FetchedResource:
        current = url
        if check_robots and self.settings.direct_web_respect_robots and not await self._robots_allowed(current):
            raise ValueError("robots.txt 不允许自动读取该网址")

        for _ in range(self.settings.direct_web_max_redirects + 1):
            await self._validate_public_url(current)
            await self._throttle(current)
            async with self.client.stream("GET", current, headers={"Accept": accept}) as response:
                if response.status_code in self._redirect_codes:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError(f"HTTP {response.status_code} 缺少 Location")
                    current = urljoin(str(response.url), location)
                    if check_robots and self.settings.direct_web_respect_robots and not await self._robots_allowed(current):
                        raise ValueError("跳转目标被 robots.txt 禁止自动读取")
                    continue
                response.raise_for_status()
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > self.settings.direct_web_max_bytes:
                    raise ValueError("远程文件超过允许大小")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self.settings.direct_web_max_bytes:
                        raise ValueError("远程文件超过允许大小")
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                return FetchedResource(
                    content=bytes(body),
                    content_type=content_type,
                    status_code=response.status_code,
                    url=str(response.url),
                    headers={key.lower(): value for key, value in response.headers.items()},
                )
        raise ValueError("公开全文网址重定向次数过多")

    async def _robots_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        cached = _ROBOTS_CACHE.get(origin)
        if cached and time.monotonic() - cached[0] < 3600:
            parser = cached[1]
            return True if parser is None else parser.can_fetch(self.user_agent, url)
        robots_url = origin + "/robots.txt"
        try:
            fetched = await self._fetch_bytes(robots_url, accept="text/plain", check_robots=False)
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(fetched.content.decode("utf-8", errors="replace").splitlines())
            _ROBOTS_CACHE[origin] = (time.monotonic(), parser)
            return parser.can_fetch(self.user_agent, url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {401, 403}:
                return False
            _ROBOTS_CACHE[origin] = (time.monotonic(), None)
            return True
        except (httpx.HTTPError, ValueError):
            _ROBOTS_CACHE[origin] = (time.monotonic(), None)
            return True

    async def _validate_public_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("只允许完整的 http(s) 网址")
        if parsed.username or parsed.password:
            raise ValueError("网址不能包含账号或密码")
        host = parsed.hostname.rstrip(".").casefold()
        if host == "localhost" or host.endswith(".localhost"):
            raise ValueError("不允许访问本机或内网网址")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            records = await asyncio.to_thread(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError("网址域名无法解析") from exc
        addresses = {record[4][0].split("%", 1)[0] for record in records}
        if not addresses:
            raise ValueError("网址域名没有可用地址")
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address)
            except ValueError as exc:
                raise ValueError("网址解析到了无效地址") from exc
            if not ip.is_global:
                raise ValueError("不允许访问本机、内网、链路本地或保留地址")

    async def _throttle(self, url: str) -> None:
        host = urlparse(url).hostname or ""
        interval = max(0.0, float(self.settings.direct_web_min_interval_seconds))
        if not host or interval <= 0:
            return
        now = time.monotonic()
        with _DOMAIN_SCHEDULE_LOCK:
            available = _DOMAIN_SCHEDULE.get(host, now)
            scheduled = max(now, available)
            _DOMAIN_SCHEDULE[host] = scheduled + interval
        if scheduled > now:
            await asyncio.sleep(scheduled - now)

    @classmethod
    def inspect_html(cls, content: bytes, base_url: str, min_chars: int = 3000) -> HtmlPageInfo:
        soup = BeautifulSoup(content, "html.parser")
        title = cls._meta_content(soup, {"citation_title", "dc.title", "og:title"})
        if not title:
            node = soup.find("h1") or soup.find("title")
            title = cls._clean_text(node.get_text(" ", strip=True)) if node else None

        doi_value = cls._meta_content(soup, {"citation_doi", "dc.identifier", "prism.doi", "dc.identifier.doi"})
        doi = extract_doi(doi_value)
        if not doi:
            canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
            doi = extract_doi(canonical.get("href")) if canonical else None

        license_value = cls._meta_content(soup, {"citation_license", "dc.rights", "dcterms.license"})
        if not license_value:
            license_link = soup.find("a", rel=lambda value: value and "license" in value)
            if license_link:
                license_value = license_link.get("href") or cls._clean_text(license_link.get_text(" ", strip=True))

        pdf_urls: list[str] = []
        for meta in soup.find_all("meta"):
            key = str(meta.get("name") or meta.get("property") or "").casefold()
            value = str(meta.get("content") or "").strip()
            if value and key in {"citation_pdf_url", "pdf_url", "wkhealth_pdf_url", "eprints.document_url"}:
                pdf_urls.append(urljoin(base_url, value))
        for node in soup.find_all(["a", "link", "iframe", "embed", "object"]):
            value = node.get("href") or node.get("src") or node.get("data")
            if not value or str(value).startswith(("javascript:", "data:")):
                continue
            text = cls._clean_text(node.get_text(" ", strip=True)).casefold()
            attrs = " ".join(str(node.get(key) or "") for key in ("type", "title", "class", "id")).casefold()
            lowered = str(value).casefold()
            if (
                ".pdf" in lowered
                or "view-pdf" in lowered
                or "download-pdf" in lowered
                or "application/pdf" in attrs
                or text in {"pdf", "download pdf", "full text pdf", "下载pdf", "下载 pdf"}
            ):
                pdf_urls.append(urljoin(base_url, str(value)))
        pdf_urls = list(dict.fromkeys(pdf_urls))

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
        root = max(roots, key=lambda item: len(cls._clean_text(item.get_text(" ", strip=True))))
        article_text = cls._clean_text(root.get_text(" ", strip=True))
        headings = [cls._clean_text(item.get_text(" ", strip=True)).casefold() for item in root.find_all(re.compile("^h[1-6]$"))]
        signals = {signal for signal in cls._article_signals if any(signal in heading for heading in headings)}
        paragraph_count = sum(
            1 for item in root.find_all("p") if len(cls._clean_text(item.get_text(" ", strip=True))) >= 40
        )
        table_count = len(root.find_all("table"))
        structural_signal = len(signals) >= 2 or (len(headings) >= 4 and paragraph_count >= 10) or paragraph_count >= 20
        looks_like_fulltext = len(article_text) >= min_chars and structural_signal and (paragraph_count >= 5 or table_count > 0)
        return HtmlPageInfo(
            title=title,
            doi=doi,
            license=license_value,
            pdf_urls=pdf_urls,
            article_chars=len(article_text),
            looks_like_fulltext=looks_like_fulltext,
        )

    @staticmethod
    def _meta_content(soup: BeautifulSoup, names: set[str]) -> str | None:
        for meta in soup.find_all("meta"):
            key = str(meta.get("name") or meta.get("property") or "").casefold()
            if key in names and meta.get("content"):
                return str(meta["content"]).strip() or None
        return None

    @staticmethod
    def _clean_text(value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip()

    @classmethod
    def _safe_url(cls, value: str) -> str:
        parsed = urlparse(value)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))

    @staticmethod
    def _loop_key(value: str) -> str:
        parsed = urlparse(value)
        return urlunparse((parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path, parsed.params, parsed.query, ""))

    @classmethod
    def _safe_error(cls, label: str, exc: Exception) -> str:
        message = re.sub(r"https?://\S+", lambda match: cls._safe_url(match.group(0)), str(exc))
        message = re.sub(r"\s+", " ", message).strip()[:500]
        return f"{label}: {type(exc).__name__}: {message}"
