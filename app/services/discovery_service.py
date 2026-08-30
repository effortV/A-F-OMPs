from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import DiscoveryCandidate, ImportJob, JobStatus
from app.services.dedupe import find_duplicate, title_author_fingerprint
from app.services.literature import LiteratureDiscovery
from app.services.vocab import VocabularyExpander


class DiscoveryService:
    async def discover(
        self,
        db: Session,
        job: ImportJob,
        *,
        limit: int,
        year_from: int | None,
        year_to: int | None,
        include_citation_expansion: bool,
    ) -> list[DiscoveryCandidate]:
        job.status = JobStatus.discovering
        job.stage = "关键词扩展"
        job.progress = 0.05
        db.commit()

        expander = VocabularyExpander()
        expanded = await expander.expand(job.query)
        job.expanded_terms = expanded
        job.stage = "多源文献检索"
        job.progress = 0.15
        db.commit()

        discovery = LiteratureDiscovery()
        try:
            candidates = await discovery.search(
                expander.search_queries(expanded),
                expanded,
                limit=limit,
                year_from=year_from,
                year_to=year_to,
                include_citation_expansion=include_citation_expansion,
            )
            connector_status = discovery.last_diagnostics
        finally:
            await discovery.close()

        db.execute(delete(DiscoveryCandidate).where(DiscoveryCandidate.job_id == job.id))
        rows: list[DiscoveryCandidate] = []
        existing_count = 0
        for rank, candidate in enumerate(candidates, start=1):
            fingerprint = title_author_fingerprint(candidate.title, candidate.authors)
            _, duplicate_reason = find_duplicate(
                db,
                job.knowledge_base_id,
                doi=candidate.doi,
                openalex_id=candidate.openalex_id,
                fingerprint=fingerprint,
            )
            already_exists = duplicate_reason is not None
            existing_count += int(already_exists)
            row = DiscoveryCandidate(
                job_id=job.id,
                candidate_id=candidate.candidate_id,
                rank=rank,
                source=candidate.source,
                doi=candidate.doi,
                openalex_id=candidate.openalex_id,
                semantic_scholar_id=candidate.semantic_scholar_id,
                title=candidate.title,
                title_author_fingerprint=fingerprint,
                authors=candidate.authors,
                publication_year=candidate.publication_year,
                venue=candidate.venue,
                abstract=candidate.abstract,
                landing_url=candidate.landing_url,
                fulltext_url=candidate.fulltext_url,
                license=candidate.license,
                is_open_access=candidate.is_open_access,
                relevance_score=candidate.relevance_score,
                relevance_reasons=candidate.relevance_reasons,
                already_exists=already_exists,
                duplicate_reason=duplicate_reason,
                raw_json=candidate.raw,
            )
            db.add(row)
            rows.append(row)
        job.status = JobStatus.awaiting_selection
        job.stage = "等待用户选择"
        job.progress = 0.25
        job.counts = {
            "total_found": len(rows),
            "existing": existing_count,
            "new": len(rows) - existing_count,
            "selected": 0,
            "downloaded": 0,
            "parsed": 0,
            "indexed": 0,
            "failed": 0,
            "connectors": connector_status,
        }
        connector_logs = [
            {
                "stage": "connector_warning",
                "message": f"{name} 渠道状态：{status['status']}" + (
                    f"（HTTP {status['errors'][0].get('status_code')}）" if status.get("errors") and status["errors"][0].get("status_code") else ""
                ),
            }
            for name, status in connector_status.items()
            if status.get("status") in {"error", "degraded", "not_configured"}
        ]
        job.log = [
            *job.log,
            *connector_logs,
            {"stage": "discovery", "message": f"发现 {len(rows)} 篇，其中去重后新文献 {len(rows) - existing_count} 篇"},
        ]
        db.commit()
        return rows


def candidates_for_job(db: Session, job_id: str, *, new_only: bool = False) -> list[DiscoveryCandidate]:
    statement = select(DiscoveryCandidate).where(DiscoveryCandidate.job_id == job_id)
    if new_only:
        statement = statement.where(DiscoveryCandidate.already_exists.is_(False))
    return list(db.scalars(statement.order_by(DiscoveryCandidate.rank)))
