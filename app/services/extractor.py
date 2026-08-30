from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import Chunk, Document, ExtractedFact
from app.services.llm import DeepSeekClient


ENTITY_ALIASES = {
    "piperazine": "PIP",
    "哌嗪": "PIP",
    "trimesoyl chloride": "TMC",
    "均苯三甲酰氯": "TMC",
    "polyamide": "PA",
    "聚酰胺": "PA",
    "magnesium chloride": "MgCl2",
    "氯化镁": "MgCl2",
    "lithium chloride": "LiCl",
    "氯化锂": "LiCl",
}


def align_entity(value: str) -> str:
    clean = re.sub(r"\s+", " ", value.strip())
    return ENTITY_ALIASES.get(clean.casefold(), clean)


def normalize_measure(value: float | None, unit: str | None, predicate: str) -> tuple[float | None, str | None]:
    if value is None or not unit:
        return value, unit
    compact = unit.replace(" ", "").replace("−", "-").replace("·", "")
    lower = compact.casefold()
    if lower in {"bar"}:
        return value * 0.1, "MPa"
    if lower in {"kpa"}:
        return value / 1000, "MPa"
    if lower in {"psi"}:
        return value * 0.00689476, "MPa"
    if lower in {"pa"} and "pressure" in predicate.casefold():
        return value / 1_000_000, "MPa"
    if lower in {"k", "kelvin"}:
        return value - 273.15, "°C"
    if lower in {"fraction", "ratio"} and 0 <= value <= 1 and any(term in predicate.casefold() for term in ("rejection", "截留")):
        return value * 100, "%"
    if lower in {"lmh", "l/m2/h", "lm-2h-1", "l/m²/h"}:
        return value, "L·m⁻²·h⁻¹"
    if lower in {"lmhbar-1", "l/m2/h/bar", "lm-2h-1bar-1"}:
        return value, "L·m⁻²·h⁻¹·bar⁻¹"
    return value, unit


class FactExtractor:
    def __init__(self, llm: DeepSeekClient | None = None):
        self.llm = llm or DeepSeekClient()

    async def extract_document(self, db: Session, document: Document, batch_size: int = 6) -> list[ExtractedFact]:
        db.execute(delete(ExtractedFact).where(ExtractedFact.document_id == document.id))
        if not self.llm.configured:
            return []
        chunks = list(db.scalars(select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.chunk_index)))
        extracted: list[dict[str, Any]] = []
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            payload = [
                {
                    "chunk_id": chunk.id,
                    "section": chunk.section,
                    "block_kind": chunk.block_kind,
                    "table_id": chunk.source_label if chunk.block_kind == "table" else None,
                    "page": chunk.page_start,
                    "text": chunk.text,
                }
                for chunk in batch
            ]
            response = await self.llm.json_chat(
                system=(
                    "你是纳滤论文数据抽取专家。仅抽取原文明确陈述的信息，不推测，不把综述引用的他人数据当成本论文实验数据。"
                    "把膜材料、膜批次/样品名、制备工艺、实验条件、溶质体系、通量/渗透率、截留率、选择性、"
                    "结构性质和机理结论输出为 facts 数组。每项必须含 fact_type, subject, predicate, object_text, "
                    "value, unit, conditions, source_sentence, page, table_id, confidence；未知字段用 null。"
                    "conditions 中尽量保留 pressure、temperature、feed_concentration、pH、solute、solvent、membrane_batch。"
                    "source_sentence 必须是输入中的短原文证据，confidence 为 0~1。"
                ),
                user=json.dumps({"doi": document.doi, "title": document.title, "chunks": payload}, ensure_ascii=False),
                max_tokens=6000,
            )
            facts = response.get("facts", []) if isinstance(response, dict) else []
            if isinstance(facts, list):
                extracted.extend(item for item in facts if isinstance(item, dict))

        rows: list[ExtractedFact] = []
        for raw in extracted:
            sentence = str(raw.get("source_sentence") or "").strip()
            subject = str(raw.get("subject") or "").strip()
            predicate = str(raw.get("predicate") or "").strip()
            if not sentence or not subject or not predicate:
                continue
            try:
                confidence = max(0.0, min(1.0, float(raw.get("confidence") or 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            value = raw.get("value")
            try:
                value = float(value) if value is not None else None
            except (TypeError, ValueError):
                value = None
            unit = str(raw.get("unit")) if raw.get("unit") else None
            normalized_value, normalized_unit = normalize_measure(value, unit, predicate)
            conditions = raw.get("conditions") if isinstance(raw.get("conditions"), dict) else {}
            conditions = {key: align_entity(str(val)) if isinstance(val, str) else val for key, val in conditions.items()}
            row = ExtractedFact(
                document_id=document.id,
                fact_type=str(raw.get("fact_type") or "relation")[:80],
                subject=align_entity(subject)[:500],
                predicate=predicate[:200],
                object_text=align_entity(str(raw["object_text"])) if raw.get("object_text") else None,
                value=value,
                unit=unit,
                normalized_value=normalized_value,
                normalized_unit=normalized_unit,
                conditions=conditions,
                source_sentence=sentence,
                page=int(raw["page"]) if str(raw.get("page") or "").isdigit() else None,
                table_id=str(raw["table_id"])[:80] if raw.get("table_id") else None,
                confidence=confidence,
                review_status="auto_accepted" if confidence >= 0.85 else "pending",
                raw_json=raw,
            )
            db.add(row)
            rows.append(row)
        db.flush()
        return rows
