from __future__ import annotations

import asyncio
import hashlib
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable
from urllib.parse import quote, urlparse

import httpx

from app.config import Settings, get_settings
from app.services.dedupe import normalize_doi, normalize_openalex_id, title_author_fingerprint


def usable_fulltext_hint(value: str | None) -> str | None:
    """Reject landing-page URLs that upstream APIs sometimes mislabel as OA PDFs."""
    if not value:
        return None
    parsed = urlparse(value)
    host = (parsed.hostname or "").casefold()
    if host in {"doi.org", "dx.doi.org"} or host.endswith("semanticscholar.org"):
        return None
    return value


@dataclass(slots=True)
class LiteratureCandidate:
    candidate_id: str
    source: str
    title: str
    authors: list[dict[str, Any]] = field(default_factory=list)
    doi: str | None = None
    openalex_id: str | None = None
    semantic_scholar_id: str | None = None
    publication_year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    landing_url: str | None = None
    fulltext_url: str | None = None
    license: str | None = None
    is_open_access: bool = False
    relevance_score: float = 0.0
    relevance_reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, *, source: str, title: str, authors: list[dict[str, Any]], **kwargs: Any) -> LiteratureCandidate:
        doi = normalize_doi(kwargs.get("doi"))
        openalex_id = normalize_openalex_id(kwargs.get("openalex_id"))
        fingerprint = title_author_fingerprint(title, authors)
        stable = doi or openalex_id or fingerprint
        candidate_id = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:32]
        return cls(
            candidate_id=candidate_id,
            source=source,
            title=title.strip(),
            authors=authors,
            doi=doi,
            openalex_id=openalex_id,
            semantic_scholar_id=kwargs.get("semantic_scholar_id"),
            publication_year=kwargs.get("publication_year"),
            venue=kwargs.get("venue"),
            abstract=kwargs.get("abstract"),
            landing_url=kwargs.get("landing_url"),
            fulltext_url=usable_fulltext_hint(kwargs.get("fulltext_url")),
            license=kwargs.get("license"),
            is_open_access=bool(kwargs.get("is_open_access")),
            raw=kwargs.get("raw") or {},
        )

    @property
    def fingerprint(self) -> str:
        return title_author_fingerprint(self.title, self.authors)

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("raw", None)
        return result


def _openalex_abstract(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        words.extend((position, word) for position in positions)
    return " ".join(word for _, word in sorted(words))


def _clean_abstract(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"<[^>]+>", " ", value).replace("\n", " ").strip()


class LiteratureDiscovery:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        headers = {"User-Agent": f"NF-Atlas/0.1 (mailto:{self.settings.openalex_email or 'noreply@example.org'})"}
        self.client = httpx.AsyncClient(timeout=35, follow_redirects=True, headers=headers)
        self.last_diagnostics: dict[str, dict[str, Any]] = {}
        self._source_limits = {
            "openalex": (asyncio.Semaphore(2), 0.20),
            "crossref": (asyncio.Semaphore(1), 0.55),
            "semantic_scholar": (asyncio.Semaphore(1), 1.10),
            "elsevier": (asyncio.Semaphore(1), 1.00),
            "unpaywall": (asyncio.Semaphore(3), 0.20),
        }
        self._last_request = {source: 0.0 for source in self._source_limits}
        self._blocked_sources: dict[str, int] = {}

    async def close(self) -> None:
        await self.client.aclose()

    async def _request(
        self,
        source: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        allow_statuses: set[int] | None = None,
        max_attempts: int = 4,
    ) -> httpx.Response:
        """Rate-limit each upstream independently and retry only transient failures."""
        semaphore, minimum_interval = self._source_limits[source]
        allowed = allow_statuses or set()
        async with semaphore:
            if source in self._blocked_sources:
                status = self._blocked_sources[source]
                request = httpx.Request("GET", url, params=params, headers=headers)
                response = httpx.Response(status, request=request)
                raise httpx.HTTPStatusError(f"{source} circuit open", request=request, response=response)
            delay = minimum_interval - (time.monotonic() - self._last_request[source])
            if delay > 0:
                await asyncio.sleep(delay)
            for attempt in range(max_attempts):
                response = await self.client.get(url, params=params, headers=headers)
                self._last_request[source] = time.monotonic()
                if response.status_code in allowed or response.status_code < 400:
                    return response
                if response.status_code in {401, 403}:
                    self._blocked_sources[source] = response.status_code
                    response.raise_for_status()
                if response.status_code not in {429, 500, 502, 503, 504} or attempt + 1 >= max_attempts:
                    response.raise_for_status()
                retry_after = response.headers.get("retry-after")
                try:
                    wait_seconds = float(retry_after) if retry_after else 0.0
                except ValueError:
                    wait_seconds = 0.0
                if wait_seconds <= 0:
                    wait_seconds = min(12.0, 1.5 * (2**attempt)) + random.uniform(0.1, 0.6)
                await asyncio.sleep(min(wait_seconds, 30.0))
            raise RuntimeError(f"{source} request failed after retries")

    def _add_openalex_auth(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.settings.openalex_api_key:
            params["api_key"] = self.settings.openalex_api_key.get_secret_value()
        if self.settings.openalex_email:
            params["mailto"] = self.settings.openalex_email
        return params

    def _elsevier_headers(self, accept: str) -> dict[str, str]:
        if not self.settings.elsevier_api_key:
            return {"Accept": accept}
        headers = {
            "X-ELS-APIKey": self.settings.elsevier_api_key.get_secret_value(),
            "X-ELS-ResourceVersion": "new",
            "Accept": accept,
        }
        if self.settings.elsevier_insttoken:
            headers["X-ELS-Insttoken"] = self.settings.elsevier_insttoken.get_secret_value()
        return headers

    @staticmethod
    def _safe_error(exc: BaseException) -> dict[str, Any]:
        if isinstance(exc, httpx.HTTPStatusError):
            request_url = exc.request.url
            error_type = "circuit_open" if "circuit open" in str(exc).casefold() else "http_status"
            result = {
                "type": error_type,
                "status_code": exc.response.status_code,
                "endpoint": f"{request_url.scheme}://{request_url.host}{request_url.path}",
            }
            if request_url.host == "api.elsevier.com":
                try:
                    status = (exc.response.json().get("service-error") or {}).get("status") or {}
                    service_code = str(status.get("statusCode") or "").strip()
                    if service_code:
                        result["service_code"] = service_code
                except (TypeError, ValueError):
                    pass
            return result
        if isinstance(exc, httpx.TimeoutException):
            return {"type": "timeout"}
        if isinstance(exc, httpx.RequestError):
            request_url = exc.request.url if exc.request else None
            return {
                "type": "network_error",
                "endpoint": f"{request_url.scheme}://{request_url.host}{request_url.path}" if request_url else None,
            }
        return {"type": type(exc).__name__}

    async def search(
        self,
        queries: list[str],
        expanded_terms: dict[str, list[str]],
        *,
        limit: int = 100,
        year_from: int | None = None,
        year_to: int | None = None,
        include_citation_expansion: bool = True,
    ) -> list[LiteratureCandidate]:
        per_source = max(10, min(100, math.ceil(limit / max(1, len(queries)))))
        calls: list[tuple[str, Any]] = []
        for query_index, query in enumerate(queries):
            calls.append(("crossref", self._search_crossref(query, per_source, year_from, year_to)))
            # Semantic Scholar's unauthenticated quota is intentionally small.
            # Two diverse queries complement OpenAlex without turning every
            # discovery round into a long chain of 429 retries.
            if self.settings.semantic_scholar_api_key or query_index < 2:
                calls.append(("semantic_scholar", self._search_semantic_scholar(query, min(per_source, 100), year_from, year_to)))
            if self.settings.openalex_api_key:
                calls.append(("openalex", self._search_openalex(query, per_source, year_from, year_to)))
            if self.settings.elsevier_api_key:
                calls.append(("elsevier", self._search_elsevier(query, min(per_source, 50), year_from, year_to)))
        batches = await asyncio.gather(*(call for _, call in calls), return_exceptions=True)
        diagnostics: dict[str, dict[str, Any]] = {
            "openalex": {"status": "not_configured" if not self.settings.openalex_api_key else "pending", "requests": 0, "successes": 0, "result_count": 0, "errors": []},
            "crossref": {"status": "pending", "requests": 0, "successes": 0, "result_count": 0, "errors": []},
            "semantic_scholar": {"status": "pending", "requests": 0, "successes": 0, "result_count": 0, "errors": [], "authenticated": bool(self.settings.semantic_scholar_api_key)},
            "elsevier": {"status": "not_configured" if not self.settings.elsevier_api_key else "pending", "requests": 0, "successes": 0, "result_count": 0, "errors": []},
            "unpaywall": {"status": "not_configured" if not self.settings.unpaywall_email else "pending", "requests": 0, "successes": 0, "result_count": 0, "errors": []},
        }
        valid_batches: list[list[LiteratureCandidate]] = []
        for (source, _), batch in zip(calls, batches, strict=False):
            status = diagnostics[source]
            if isinstance(batch, BaseException):
                error = self._safe_error(batch)
                status["failures"] = status.get("failures", 0) + 1
                if error["type"] == "circuit_open":
                    status["skipped_after_circuit"] = status.get("skipped_after_circuit", 0) + 1
                else:
                    status["requests"] += 1
                    if error not in status["errors"]:
                        status["errors"].append(error)
            else:
                status["requests"] += 1
                status["successes"] += 1
                status["result_count"] += len(batch)
                valid_batches.append(batch)
        for status in diagnostics.values():
            if status["status"] == "not_configured":
                continue
            if status["successes"] == 0 and status["errors"]:
                status["status"] = "error"
            elif status["errors"]:
                status["status"] = "degraded"
            elif status["requests"]:
                status["status"] = "ok"
        self.last_diagnostics = diagnostics
        merged = self._merge(valid_batches)
        self._score(merged, expanded_terms)
        merged.sort(key=lambda item: item.relevance_score, reverse=True)

        if include_citation_expansion and merged and self.settings.openalex_api_key:
            seeds = [item for item in merged[: min(5, len(merged))] if item.openalex_id]
            expanded = await self._expand_openalex_network(seeds, per_seed=8)
            merged = self._merge([merged, expanded])
            self._score(merged, expanded_terms)
            merged.sort(key=lambda item: item.relevance_score, reverse=True)
        await self._enrich_fulltext_availability(merged[: min(limit, 200)])
        unpaywall = self.last_diagnostics["unpaywall"]
        if self.settings.unpaywall_email:
            availability = [item.raw.get("fulltext_availability") for item in merged[: min(limit, 200)]]
            unpaywall["requests"] = sum(value not in {None, "open_fulltext_hint"} for value in availability)
            unpaywall["successes"] = sum(value == "unpaywall_open" for value in availability)
            unpaywall["result_count"] = unpaywall["successes"]
            failed = [value for value in availability if isinstance(value, str) and value.startswith("check_failed:")]
            if failed:
                unpaywall["status"] = "degraded"
                unpaywall["errors"] = [{"type": value.split(":", 1)[-1]} for value in failed[:5]]
            else:
                unpaywall["status"] = "ok"
        self._score(merged, expanded_terms)
        merged.sort(key=lambda item: item.relevance_score, reverse=True)
        return merged[:limit]

    async def _enrich_fulltext_availability(self, candidates: list[LiteratureCandidate]) -> None:
        semaphore = asyncio.Semaphore(8)

        async def check(candidate: LiteratureCandidate) -> None:
            if candidate.fulltext_url:
                candidate.raw["fulltext_availability"] = "open_fulltext_hint"
                return
            if "sciencedirect" in candidate.source and self.settings.elsevier_api_key:
                candidate.raw["tdm_availability"] = "authorization_checked_when_selected"
            if not candidate.doi or not self.settings.unpaywall_email:
                candidate.raw.setdefault("fulltext_availability", "not_checked_no_unpaywall_email")
                return
            async with semaphore:
                response = await self._request(
                    "unpaywall",
                    f"https://api.unpaywall.org/v2/{quote(candidate.doi, safe='')}",
                    params={"email": self.settings.unpaywall_email},
                    allow_statuses={404},
                )
            if response.status_code != 200:
                candidate.raw["fulltext_availability"] = "not_found"
                return
            data = response.json()
            location = data.get("best_oa_location") or {}
            url = location.get("url_for_pdf") or location.get("url_for_landing_page")
            if url:
                candidate.fulltext_url = url
                candidate.is_open_access = True
                candidate.license = candidate.license or location.get("license")
                candidate.raw["fulltext_availability"] = "unpaywall_open"
            else:
                candidate.raw["fulltext_availability"] = "no_open_location"

        results = await asyncio.gather(*(check(candidate) for candidate in candidates), return_exceptions=True)
        for candidate, result in zip(candidates, results, strict=False):
            if isinstance(result, Exception):
                candidate.raw["fulltext_availability"] = f"check_failed:{type(result).__name__}"

    async def _search_openalex(
        self, query: str, limit: int, year_from: int | None, year_to: int | None
    ) -> list[LiteratureCandidate]:
        filters = ["type:article|review"]
        if year_from:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if year_to:
            filters.append(f"to_publication_date:{year_to}-12-31")
        params: dict[str, Any] = {
            "search": query,
            "per-page": min(limit, 100),
            "filter": ",".join(filters),
            "select": "id,doi,title,publication_year,authorships,primary_location,open_access,abstract_inverted_index,referenced_works,cited_by_count,topics",
        }
        self._add_openalex_auth(params)
        response = await self._request("openalex", "https://api.openalex.org/works", params=params)
        return [self._from_openalex(item) for item in response.json().get("results", []) if item.get("title")]

    def _from_openalex(self, item: dict[str, Any]) -> LiteratureCandidate:
        location = item.get("primary_location") or {}
        source = location.get("source") or {}
        oa = item.get("open_access") or {}
        authors = [
            {
                "name": (authorship.get("author") or {}).get("display_name", ""),
                "orcid": (authorship.get("author") or {}).get("orcid"),
            }
            for authorship in item.get("authorships") or []
        ]
        return LiteratureCandidate.create(
            source="openalex",
            title=item["title"],
            authors=authors,
            doi=item.get("doi"),
            openalex_id=item.get("id"),
            publication_year=item.get("publication_year"),
            venue=source.get("display_name"),
            abstract=_openalex_abstract(item.get("abstract_inverted_index")),
            landing_url=location.get("landing_page_url"),
            fulltext_url=location.get("pdf_url") if location.get("is_oa") else None,
            license=location.get("license"),
            is_open_access=bool(oa.get("is_oa")),
            raw={
                "referenced_works": item.get("referenced_works") or [],
                "cited_by_count": item.get("cited_by_count", 0),
                "topics": item.get("topics") or [],
            },
        )

    async def _search_crossref(
        self, query: str, limit: int, year_from: int | None, year_to: int | None
    ) -> list[LiteratureCandidate]:
        filters = ["type:journal-article"]
        if year_from:
            filters.append(f"from-pub-date:{year_from}-01-01")
        if year_to:
            filters.append(f"until-pub-date:{year_to}-12-31")
        params: dict[str, Any] = {
            "query.bibliographic": query,
            "rows": min(limit, 1000),
            "filter": ",".join(filters),
            "select": "DOI,title,author,published,container-title,abstract,URL,license,references-count,is-referenced-by-count",
        }
        if self.settings.openalex_email:
            params["mailto"] = self.settings.openalex_email
        response = await self._request("crossref", "https://api.crossref.org/works", params=params)
        results = []
        for item in response.json().get("message", {}).get("items", []):
            title = " ".join(item.get("title") or []).strip()
            if not title:
                continue
            authors = [
                {"name": " ".join(part for part in (author.get("given"), author.get("family")) if part), "orcid": author.get("ORCID")}
                for author in item.get("author") or []
            ]
            date_parts = (item.get("published") or {}).get("date-parts") or [[]]
            year = date_parts[0][0] if date_parts and date_parts[0] else None
            licenses = item.get("license") or []
            results.append(
                LiteratureCandidate.create(
                    source="crossref",
                    title=title,
                    authors=authors,
                    doi=item.get("DOI"),
                    publication_year=year,
                    venue="; ".join(item.get("container-title") or []),
                    abstract=_clean_abstract(item.get("abstract")),
                    landing_url=item.get("URL"),
                    license=licenses[0].get("URL") if licenses else None,
                    raw={"reference_count": item.get("references-count"), "cited_by_count": item.get("is-referenced-by-count")},
                )
            )
        return results

    async def _search_semantic_scholar(
        self, query: str, limit: int, year_from: int | None, year_to: int | None
    ) -> list[LiteratureCandidate]:
        headers = {}
        if self.settings.semantic_scholar_api_key:
            headers["x-api-key"] = self.settings.semantic_scholar_api_key.get_secret_value()
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "paperId,title,abstract,year,authors,venue,externalIds,url,openAccessPdf,citationCount,referenceCount",
        }
        if year_from or year_to:
            params["year"] = f"{year_from or ''}-{year_to or ''}"
        response = await self._request(
            "semantic_scholar",
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers=headers,
        )
        results = []
        for item in response.json().get("data", []):
            external = item.get("externalIds") or {}
            oa_pdf = item.get("openAccessPdf") or {}
            oa_url = oa_pdf.get("url")
            results.append(
                LiteratureCandidate.create(
                    source="semantic_scholar",
                    title=item.get("title") or "",
                    authors=[{"name": author.get("name", ""), "semantic_scholar_id": author.get("authorId")} for author in item.get("authors") or []],
                    doi=external.get("DOI"),
                    openalex_id=external.get("OpenAlex"),
                    semantic_scholar_id=item.get("paperId"),
                    publication_year=item.get("year"),
                    venue=item.get("venue"),
                    abstract=item.get("abstract"),
                    landing_url=item.get("url"),
                    fulltext_url=oa_url,
                    is_open_access=bool(oa_url),
                    raw={
                        "citation_count": item.get("citationCount"),
                        "reference_count": item.get("referenceCount"),
                        "semantic_scholar_oa_url": oa_url,
                    },
                )
            )
        return [item for item in results if item.title]

    async def _search_elsevier(
        self, query: str, limit: int, year_from: int | None, year_to: int | None
    ) -> list[LiteratureCandidate]:
        api_key = self.settings.elsevier_api_key
        if not api_key:
            return []
        clauses = [f"TITLE-ABSTR-KEY({query})"]
        if year_from:
            clauses.append(f"PUBYEAR > {year_from - 1}")
        if year_to:
            clauses.append(f"PUBYEAR < {year_to + 1}")
        response = await self._request(
            "elsevier",
            "https://api.elsevier.com/content/search/sciencedirect",
            params={"query": " AND ".join(clauses), "count": min(limit, 100)},
            headers=self._elsevier_headers("application/json"),
        )
        results = []
        for item in response.json().get("search-results", {}).get("entry", []):
            title = item.get("dc:title") or ""
            creator = item.get("dc:creator") or ""
            identifier = str(item.get("dc:identifier") or "")
            pii = str(item.get("pii") or "") or (
                identifier.split(":", 1)[1]
                if identifier.casefold().startswith(("scidir:", "pii:")) and ":" in identifier
                else ""
            )
            landing_url = (
                f"https://www.sciencedirect.com/science/article/pii/{pii}"
                if pii
                else item.get("prism:url")
            )
            raw = dict(item)
            if pii:
                raw["elsevier_pii"] = pii
            results.append(
                LiteratureCandidate.create(
                    source="sciencedirect",
                    title=title,
                    authors=[{"name": creator}] if creator else [],
                    doi=item.get("prism:doi"),
                    publication_year=int(item["prism:coverDate"][:4]) if item.get("prism:coverDate") else None,
                    venue=item.get("prism:publicationName"),
                    landing_url=landing_url,
                    raw=raw,
                )
            )
        return [item for item in results if item.title]

    async def _expand_openalex_network(
        self, seeds: list[LiteratureCandidate], per_seed: int
    ) -> list[LiteratureCandidate]:
        calls = []
        for seed in seeds:
            refs = (seed.raw.get("referenced_works") or [])[:per_seed]
            if refs:
                ids = "|".join(normalize_openalex_id(ref) or "" for ref in refs)
                calls.append(self._openalex_filter(f"openalex_id:{ids}", per_seed))
            calls.append(self._openalex_filter(f"cites:{seed.openalex_id}", per_seed))
            if seed.authors:
                author_name = seed.authors[0].get("name")
                if author_name:
                    calls.append(self._search_openalex(f"{author_name} nanofiltration", min(4, per_seed), None, None))
        batches = await asyncio.gather(*calls, return_exceptions=True)
        return [item for batch in batches if isinstance(batch, list) for item in batch]

    async def _openalex_filter(self, filter_value: str, limit: int) -> list[LiteratureCandidate]:
        params = {
            "filter": filter_value,
            "per-page": min(limit, 100),
            "select": "id,doi,title,publication_year,authorships,primary_location,open_access,abstract_inverted_index,referenced_works,cited_by_count,topics",
        }
        self._add_openalex_auth(params)
        response = await self._request("openalex", "https://api.openalex.org/works", params=params)
        return [self._from_openalex(item) for item in response.json().get("results", []) if item.get("title")]

    @staticmethod
    def _merge(batches: Iterable[list[LiteratureCandidate]]) -> list[LiteratureCandidate]:
        merged: dict[str, LiteratureCandidate] = {}
        for batch in batches:
            for item in batch:
                keys = [value for value in (item.doi, item.openalex_id, item.fingerprint) if value]
                existing = next((merged[key] for key in keys if key in merged), None)
                if existing is None:
                    existing = item
                else:
                    existing.openalex_id = existing.openalex_id or item.openalex_id
                    existing.semantic_scholar_id = existing.semantic_scholar_id or item.semantic_scholar_id
                    existing.abstract = existing.abstract or item.abstract
                    existing.fulltext_url = existing.fulltext_url or item.fulltext_url
                    existing.landing_url = existing.landing_url or item.landing_url
                    existing.license = existing.license or item.license
                    existing.is_open_access = existing.is_open_access or item.is_open_access
                    existing.raw.update({f"{item.source}_{key}": value for key, value in item.raw.items()})
                    if item.source not in existing.source.split("+"):
                        existing.source += "+" + item.source
                for key in keys:
                    merged[key] = existing
        return list({id(item): item for item in merged.values()}.values())

    @staticmethod
    def _score(candidates: list[LiteratureCandidate], expanded: dict[str, list[str]]) -> None:
        topic_terms = expanded.get("zh", []) + expanded.get("en", []) + expanded.get("abbreviations", [])
        detail_terms = (
            expanded.get("materials", [])
            + expanded.get("methods", [])
            + expanded.get("systems", [])
            + expanded.get("metrics", [])
        )
        for item in candidates:
            title = item.title.casefold()
            abstract = (item.abstract or "").casefold()
            score = 0.0
            reasons = []
            if "nanofiltration" in title or "纳滤" in title:
                score += 3.0
                reasons.append("题名命中纳滤领域")
            elif "nanofiltration" in abstract or "纳滤" in abstract:
                score += 1.5
                reasons.append("摘要命中纳滤领域")
            title_hits = [term for term in topic_terms if len(term) > 1 and term.casefold() in title]
            abstract_hits = [term for term in topic_terms if len(term) > 1 and term.casefold() in abstract]
            detail_hits = [term for term in detail_terms if len(term) > 1 and term.casefold() in title + " " + abstract]
            score += min(5, len(title_hits) * 1.2) + min(3, len(abstract_hits) * 0.35) + min(2, len(detail_hits) * 0.25)
            if title_hits:
                reasons.append("题名主题词: " + ", ".join(title_hits[:4]))
            if detail_hits:
                reasons.append("材料/方法/体系命中: " + ", ".join(detail_hits[:4]))
            if item.is_open_access:
                score += 0.35
                reasons.append("存在开放全文线索")
            item.relevance_score = round(score, 4)
            item.relevance_reasons = reasons
