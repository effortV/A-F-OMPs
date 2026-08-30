from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Conversation, KnowledgeInsight
from app.services.graph_store import GraphStore
from app.services.llm import DeepSeekClient

ALLOWED_INSIGHT_TYPES = {"pattern", "contradiction", "hypothesis"}


class KnowledgeDiscoveryEngine:
    """Create traceable cross-paper syntheses without mixing them with extracted facts."""

    def __init__(
        self,
        db: Session,
        *,
        settings: Settings | None = None,
        llm: DeepSeekClient | None = None,
        graph_store: GraphStore | None = None,
    ):
        self.db = db
        self.settings = settings or get_settings()
        self.llm = llm or DeepSeekClient(self.settings)
        self.graph_store = graph_store or GraphStore(self.settings)

    async def discover(
        self,
        conversation: Conversation,
        question: str,
        evidence: list[dict[str, Any]],
    ) -> list[KnowledgeInsight]:
        if not self.llm.configured or len(evidence) < 3:
            return []
        selected_evidence = self._diverse_evidence(evidence, 36)
        indexed = [self._evidence_payload(index, item) for index, item in enumerate(selected_evidence, 1)]
        try:
            result = await self.llm.json_chat(
                system=(
                    "你是纳滤跨文献知识发现引擎。你的任务不是复述单篇论文，而是比较多篇证据中的材料—工艺—结构—"
                    "条件—性能关系，寻找重复规律、条件依赖、表面矛盾及可检验的新机理假设。"
                    "严格区分三类：pattern=跨文献归纳，contradiction=在条件差异下出现的冲突，hypothesis=尚未验证的新假设。"
                    "只能引用输入 evidence_id，不得编造论文、DOI、数值或证据。每条至少引用2条独立证据；"
                    "contradiction 必须同时给 supporting_evidence_ids 和 contradicting_evidence_ids。"
                    "假设必须给 assumptions、boundary_conditions，以及 validation_plan，其中包含 experiment、variables、"
                    "predicted_outcome、falsification_criterion。输出 insights 数组，最多6条。"
                    "每项字段：type,title,claim,rationale,supporting_evidence_ids,contradicting_evidence_ids,assumptions,"
                    "boundary_conditions,validation_plan,confidence,novelty_score。confidence 和 novelty_score 为0~1。"
                    "这些内容只能作为AI综合或AI假设，不能称为已证实的新事实。"
                ),
                user=json.dumps({"research_question": question, "evidence": indexed}, ensure_ascii=False),
                max_tokens=6500,
                enable_thinking=self.llm.enable_thinking,
            )
        except Exception:
            return []
        raw_insights = result.get("insights", []) if isinstance(result, dict) else []
        if not isinstance(raw_insights, list):
            return []

        rows: list[KnowledgeInsight] = []
        for raw in raw_insights[:6]:
            if not isinstance(raw, dict):
                continue
            normalized = self._normalize(raw, selected_evidence)
            if not normalized:
                continue
            existing = self.db.scalar(
                select(KnowledgeInsight).where(
                    KnowledgeInsight.knowledge_base_id == conversation.knowledge_base_id,
                    KnowledgeInsight.claim == normalized["claim"],
                    KnowledgeInsight.status != "rejected",
                )
            )
            if existing:
                rows.append(existing)
                continue
            row = KnowledgeInsight(
                knowledge_base_id=conversation.knowledge_base_id,
                conversation_id=conversation.id,
                question=question,
                model_name=self.llm.model_name,
                **normalized,
            )
            self.db.add(row)
            self.db.flush()
            rows.append(row)
        if rows:
            self.db.commit()
            for row in rows:
                try:
                    self.graph_store.upsert_insight(row)
                except Exception:
                    pass
        return rows

    def _normalize(self, raw: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
        insight_type = str(raw.get("type") or "").strip().casefold()
        if insight_type not in ALLOWED_INSIGHT_TYPES:
            return None
        title = str(raw.get("title") or "").strip()[:500]
        claim = str(raw.get("claim") or "").strip()[:8000]
        rationale = str(raw.get("rationale") or "").strip()[:12_000]
        if not title or not claim or not rationale:
            return None

        supporting = self._valid_ids(raw.get("supporting_evidence_ids"), len(evidence))
        contradicting = self._valid_ids(raw.get("contradicting_evidence_ids"), len(evidence))
        if insight_type == "contradiction":
            if not supporting or not contradicting:
                return None
        elif len(set(supporting)) < 2:
            return None
        supporting_sources = {
            evidence[item - 1].get("document_id")
            or evidence[item - 1].get("doi")
            or evidence[item - 1].get("title")
            for item in supporting
        }
        supporting_sources.discard(None)
        if insight_type != "contradiction" and len(supporting_sources) < 2:
            return None
        if insight_type == "contradiction":
            contradicting_sources = {
                evidence[item - 1].get("document_id")
                or evidence[item - 1].get("doi")
                or evidence[item - 1].get("title")
                for item in contradicting
            }
            contradicting_sources.discard(None)
            # A contradiction must have at least one exclusive source on each side. Reusing the
            # same paper/chunk as both support and opposition would manufacture a conflict.
            if not (supporting_sources - contradicting_sources) or not (contradicting_sources - supporting_sources):
                return None
        all_ids = list(dict.fromkeys([*supporting, *contradicting]))
        refs = []
        for evidence_id in all_ids:
            item = evidence[evidence_id - 1]
            refs.append(
                {
                    "evidence_id": evidence_id,
                    "document_id": item.get("document_id"),
                    "title": item.get("title"),
                    "doi": item.get("doi"),
                    "page": item.get("page") if item.get("page") is not None else item.get("page_start"),
                    "quote": (item.get("quote") or item.get("source_sentence") or "")[:1500],
                    "stance": "contradicts" if evidence_id in contradicting else "supports",
                    "evidence_mode": item.get("evidence_mode") or "unknown",
                }
            )
        document_ids = list(dict.fromkeys(str(item["document_id"]) for item in refs if item.get("document_id")))

        confidence = self._score(raw.get("confidence"))
        modes = {str(item.get("evidence_mode") or "unknown") for item in refs}
        if modes == {"metadata-only"}:
            confidence = min(confidence, 0.45)
        elif "metadata-only" in modes or "unknown" in modes:
            confidence = min(confidence, 0.65)
        else:
            confidence = min(confidence, 0.82)
        novelty = min(self._score(raw.get("novelty_score")), 0.95)
        validation = raw.get("validation_plan") if isinstance(raw.get("validation_plan"), dict) else {}
        assumptions = self._string_list(raw.get("assumptions"), 20)
        boundaries = self._string_list(raw.get("boundary_conditions"), 20)
        if insight_type == "hypothesis":
            required_validation_fields = {
                "experiment",
                "variables",
                "predicted_outcome",
                "falsification_criterion",
            }
            if not required_validation_fields.issubset(validation) or any(
                not validation.get(field) for field in required_validation_fields
            ):
                return None
        return {
            "insight_type": insight_type,
            "title": title,
            "claim": claim,
            "rationale": rationale,
            "evidence_document_ids": document_ids,
            "evidence_refs": refs,
            "assumptions": assumptions,
            "boundary_conditions": boundaries,
            "validation_plan": validation,
            "confidence": confidence,
            "novelty_score": novelty,
            "status": "ai_hypothesis" if insight_type == "hypothesis" else "ai_synthesis",
        }

    @staticmethod
    def _evidence_payload(index: int, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "evidence_id": index,
            "document_id": item.get("document_id"),
            "title": item.get("title"),
            "doi": item.get("doi"),
            "page": item.get("page") if item.get("page") is not None else item.get("page_start"),
            "section": item.get("section"),
            "quote": (item.get("quote") or item.get("source_sentence") or "")[:1800],
            "fact": {
                key: item.get(key)
                for key in ("subject", "predicate", "object_text", "value", "unit", "conditions")
                if item.get(key) is not None
            },
            "evidence_mode": item.get("evidence_mode") or "unknown",
        }

    @staticmethod
    def _diverse_evidence(evidence: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """Prefer broad cross-paper coverage before adding second passages from a paper."""
        selected: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        documents: set[str] = set()
        for item in evidence:
            key = str(item.get("document_id") or item.get("doi") or item.get("title") or "")
            if key and key not in documents:
                documents.add(key)
                selected.append(item)
            else:
                deferred.append(item)
            if len(selected) >= limit:
                return selected
        selected.extend(deferred[: max(0, limit - len(selected))])
        return selected

    @staticmethod
    def _valid_ids(value: Any, maximum: int) -> list[int]:
        if not isinstance(value, list):
            return []
        output = []
        for item in value:
            try:
                number = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= number <= maximum and number not in output:
                output.append(number)
        return output

    @staticmethod
    def _string_list(value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip()[:1000] for item in value[:limit] if str(item).strip()]

    @staticmethod
    def _score(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
