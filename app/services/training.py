from __future__ import annotations

import json
from typing import Any, Iterable

from app.models import TrainingTrace


def training_quality(trace: TrainingTrace) -> tuple[float, str]:
    """Estimate curation readiness without treating approval as model quality."""
    evidence = trace.retrieval_evidence or []
    tool_trace = trace.tool_trace or []
    document_ids = {
        str(item.get("document_id"))
        for item in evidence
        if isinstance(item, dict) and item.get("document_id")
    }
    fulltext_count = sum(
        1
        for item in evidence
        if isinstance(item, dict) and item.get("evidence_mode") != "metadata-only"
    )
    cited_count = sum(
        1
        for item in evidence
        if isinstance(item, dict)
        and (item.get("doi") or item.get("title"))
        and (item.get("quote") or item.get("source_sentence"))
    )

    score = 0.0
    score += min(0.25, len(evidence) * 0.04)
    score += min(0.15, len(document_ids) * 0.05)
    score += min(0.15, fulltext_count * 0.04)
    score += min(0.10, cited_count * 0.02)
    score += 0.10 if tool_trace else 0.0
    score += ((trace.rating or 0) / 5) * 0.15
    score += 0.10 if (trace.human_revision or "").strip() else 0.0
    score = round(min(1.0, score), 3)
    label = "高质量" if score >= 0.75 else "可用" if score >= 0.5 else "待审核"
    return score, label


def trace_to_read(trace: TrainingTrace) -> dict[str, Any]:
    score, label = training_quality(trace)
    return {
        "id": trace.id,
        "conversation_id": trace.conversation_id,
        "message_id": trace.message_id,
        "instruction": trace.instruction,
        "input_text": trace.input_text,
        "output_text": trace.output_text,
        "retrieval_evidence": trace.retrieval_evidence or [],
        "tool_trace": trace.tool_trace or [],
        "human_revision": trace.human_revision,
        "rating": trace.rating,
        "approved_for_training": trace.approved_for_training,
        "quality_score": score,
        "quality_label": label,
        "created_at": trace.created_at,
    }


def trace_to_dataset_record(trace: TrainingTrace) -> dict[str, Any]:
    score, label = training_quality(trace)
    return {
        "instruction": trace.instruction,
        "input": trace.input_text,
        "output": trace.human_revision or trace.output_text,
        "metadata": {
            "trace_id": trace.id,
            "conversation_id": trace.conversation_id,
            "message_id": trace.message_id,
            "rating": trace.rating,
            "approved_for_training": trace.approved_for_training,
            "quality_score": score,
            "quality_label": label,
            "retrieval_evidence": trace.retrieval_evidence or [],
            "tool_trace": trace.tool_trace or [],
        },
    }


def traces_to_jsonl(traces: Iterable[TrainingTrace]) -> bytes:
    lines = [json.dumps(trace_to_dataset_record(trace), ensure_ascii=False) for trace in traces]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
