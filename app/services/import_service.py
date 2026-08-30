from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import DiscoveryCandidate, Document, DocumentStatus, ImportJob, JobStatus
from app.services.dedupe import find_duplicate, normalize_doi
from app.services.pipeline import append_job_log


class ImportSelectionError(RuntimeError):
    pass


def create_documents_from_candidates(
    db: Session,
    job: ImportJob,
    candidate_ids: list[str] | None = None,
    *,
    top_n: int | None = None,
) -> list[str]:
    """Create selected documents after rechecking uniqueness inside the transaction."""
    if candidate_ids is not None:
        candidate_ids = list(dict.fromkeys(candidate_ids))
    statement = select(DiscoveryCandidate).where(
        DiscoveryCandidate.job_id == job.id,
        DiscoveryCandidate.already_exists.is_(False),
    )
    if candidate_ids is not None:
        statement = statement.where(DiscoveryCandidate.candidate_id.in_(candidate_ids))
    rows = list(db.scalars(statement.order_by(DiscoveryCandidate.rank)))
    by_id = {item.candidate_id: item for item in rows}
    if candidate_ids is not None:
        missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in by_id]
        if missing:
            raise ImportSelectionError(f"候选 ID 无效、已存在或不属于该任务：{missing[:5]}")
        rows = [by_id[candidate_id] for candidate_id in candidate_ids]
    elif top_n is not None:
        rows = rows[:top_n]

    document_ids: list[str] = []
    for item in rows:
        duplicate, reason = find_duplicate(
            db,
            job.knowledge_base_id,
            doi=item.doi,
            openalex_id=item.openalex_id,
            fingerprint=item.title_author_fingerprint,
        )
        if duplicate:
            item.already_exists = True
            item.duplicate_reason = reason
            continue
        document = Document(
            knowledge_base_id=job.knowledge_base_id,
            doi=item.doi,
            doi_normalized=normalize_doi(item.doi),
            openalex_id=item.openalex_id,
            semantic_scholar_id=item.semantic_scholar_id,
            title=item.title,
            title_author_fingerprint=item.title_author_fingerprint,
            authors=item.authors,
            abstract=item.abstract,
            publication_year=item.publication_year,
            venue=item.venue,
            landing_url=item.landing_url,
            fulltext_url=item.fulltext_url,
            license=item.license,
            is_open_access=item.is_open_access,
            relevance_score=item.relevance_score,
            relevance_reasons=item.relevance_reasons,
            status=DocumentStatus.selected,
            metadata_json={"import_job_id": job.id, "candidate_id": item.candidate_id, "sources": item.source},
        )
        db.add(document)
        try:
            db.flush()
        except IntegrityError as exc:
            db.rollback()
            raise ImportSelectionError("并发导入触发唯一约束；请刷新后重新检索") from exc
        document_ids.append(document.id)

    job.requested_count = len(document_ids)
    job.selected_document_ids = document_ids
    counts = dict(job.counts or {})
    counts["selected"] = len(document_ids)
    counts["execution_state"] = "queued"
    counts["worker_queue"] = get_settings().queue_name
    job.counts = counts
    if document_ids:
        job.status = JobStatus.queued
        job.stage = "已提交后台处理"
        append_job_log(job, "queued", f"已选择 {len(document_ids)} 篇去重后的新文献")
    return document_ids
