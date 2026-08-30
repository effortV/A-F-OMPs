from __future__ import annotations

import json
from typing import Any

from app.config import Settings, get_settings
from app.models import Document, ExtractedFact, KnowledgeInsight


class GraphStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.driver = None
        if self.settings.neo4j_uri and self.settings.neo4j_password:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password.get_secret_value()),
            )

    @property
    def configured(self) -> bool:
        return self.driver is not None

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def ensure_schema(self) -> None:
        if not self.driver:
            return
        statements = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (n:Entity) REQUIRE n.key IS UNIQUE",
            "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (n:Fact) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT insight_id IF NOT EXISTS FOR (n:AIInsight) REQUIRE n.id IS UNIQUE",
        ]
        with self.driver.session(database=self.settings.neo4j_database) as session:
            for statement in statements:
                session.run(statement).consume()

    def upsert_document(self, document: Document, facts: list[ExtractedFact]) -> None:
        if not self.driver:
            return
        self.ensure_schema()
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.run(
                """
                MERGE (d:Document {id: $id})
                SET d.title=$title, d.doi=$doi, d.year=$year, d.knowledge_base_id=$kb,
                    d.evidence_mode=$evidence_mode
                """,
                id=document.id,
                title=document.title,
                doi=document.doi_normalized,
                year=document.publication_year,
                kb=document.knowledge_base_id,
                evidence_mode="metadata-only" if (document.metadata_json or {}).get("metadata_only") else "fulltext",
            ).consume()
            for fact in facts:
                entity_key = f"{document.knowledge_base_id}:{fact.subject.casefold()}"
                entity_label = self._entity_label(fact.fact_type)
                session.run(
                    """
                    MATCH (d:Document {id: $document_id})
                    MERGE (e:Entity {key: $entity_key})
                    SET e.name=$subject, e.knowledge_base_id=$kb
                    MERGE (f:Fact {id: $fact_id})
                    SET f.type=$fact_type, f.predicate=$predicate, f.object_text=$object_text,
                        f.value=$value, f.unit=$unit, f.conditions_json=$conditions_json,
                        f.source_sentence=$source_sentence, f.page=$page, f.table_id=$table_id,
                        f.confidence=$confidence, f.doi=$doi
                    MERGE (d)-[:REPORTS]->(f)
                    MERGE (f)-[:ABOUT]->(e)
                    """,
                    document_id=document.id,
                    entity_key=entity_key,
                    subject=fact.subject,
                    kb=document.knowledge_base_id,
                    fact_id=fact.id,
                    fact_type=fact.fact_type,
                    predicate=fact.predicate,
                    object_text=fact.object_text,
                    value=fact.normalized_value if fact.normalized_value is not None else fact.value,
                    unit=fact.normalized_unit or fact.unit,
                    conditions_json=json.dumps(fact.conditions, ensure_ascii=False),
                    source_sentence=fact.source_sentence,
                    page=fact.page,
                    table_id=fact.table_id,
                    confidence=fact.confidence,
                    doi=document.doi_normalized,
                ).consume()
                session.run(
                    f"MATCH (e:Entity {{key: $entity_key}}) SET e:{entity_label}",
                    entity_key=entity_key,
                ).consume()

    def replace_document(self, document: Document, facts: list[ExtractedFact]) -> None:
        if not self.driver:
            return
        self.ensure_schema()
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.run(
                "MATCH (d:Document {id: $id})-[:REPORTS]->(f:Fact) DETACH DELETE f",
                id=document.id,
            ).consume()
        self.upsert_document(document, facts)
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.run(
                "MATCH (e:Entity {knowledge_base_id: $kb}) WHERE NOT (e)<-[:ABOUT]-(:Fact) DELETE e",
                kb=document.knowledge_base_id,
            ).consume()

    def upsert_insight(self, insight: KnowledgeInsight) -> None:
        if not self.driver:
            return
        self.ensure_schema()
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.run(
                """
                MERGE (h:AIInsight {id: $id})
                SET h.knowledge_base_id=$kb, h.type=$type, h.title=$title, h.claim=$claim,
                    h.rationale=$rationale, h.confidence=$confidence, h.novelty_score=$novelty,
                    h.status=$status, h.model_name=$model_name
                WITH h
                OPTIONAL MATCH (h)-[r:SUPPORTED_BY|CONTRADICTED_BY]->(:Document)
                DELETE r
                """,
                id=insight.id,
                kb=insight.knowledge_base_id,
                type=insight.insight_type,
                title=insight.title,
                claim=insight.claim,
                rationale=insight.rationale,
                confidence=insight.confidence,
                novelty=insight.novelty_score,
                status=insight.status,
                model_name=insight.model_name,
            ).consume()
            for reference in insight.evidence_refs:
                document_id = reference.get("document_id")
                if not document_id:
                    continue
                relationship = "CONTRADICTED_BY" if reference.get("stance") == "contradicts" else "SUPPORTED_BY"
                session.run(
                    f"""
                    MATCH (h:AIInsight {{id: $insight_id}}), (d:Document {{id: $document_id}})
                    MERGE (h)-[:{relationship}]->(d)
                    """,
                    insight_id=insight.id,
                    document_id=document_id,
                ).consume()

    def update_insight_status(self, insight_id: str, status: str, review_note: str | None) -> None:
        if not self.driver:
            return
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.run(
                "MATCH (h:AIInsight {id: $id}) SET h.status=$status, h.review_note=$note",
                id=insight_id,
                status=status,
                note=review_note,
            ).consume()

    @staticmethod
    def _entity_label(fact_type: str) -> str:
        value = fact_type.casefold()
        mapping = [
            (("membrane_batch", "membrane"), "Membrane"),
            (("material", "monomer"), "Material"),
            (("process", "preparation", "制备"), "Process"),
            (("solute", "separation_system", "feed"), "SeparationSystem"),
            (("condition", "pressure", "temperature"), "Condition"),
            (("structure", "property"), "Structure"),
            (("mechanism", "机理"), "Mechanism"),
            (("performance", "flux", "rejection", "selectivity"), "Performance"),
        ]
        return next((label for keywords, label in mapping if any(keyword in value for keyword in keywords)), "DomainEntity")

    def search_facts(self, knowledge_base_id: str, terms: list[str], limit: int = 20) -> list[dict[str, Any]]:
        if not self.driver or not terms:
            return []
        lowered = [term.casefold() for term in terms if term.strip()][:20]
        with self.driver.session(database=self.settings.neo4j_database) as session:
            labels_record = session.run("CALL db.labels() YIELD label RETURN collect(label) AS labels").single()
            types_record = session.run(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS types"
            ).single()
            labels = set((labels_record or {}).get("labels") or [])
            relationship_types = set((types_record or {}).get("types") or [])
            if not {"Document", "Fact", "Entity"}.issubset(labels) or not {"REPORTS", "ABOUT"}.issubset(
                relationship_types
            ):
                return []
            result = session.run(
                """
                MATCH (d:Document)-[:REPORTS]->(f:Fact)-[:ABOUT]->(e:Entity)
                WHERE d.knowledge_base_id=$kb AND any(term IN $terms WHERE
                    toLower(e.name) CONTAINS term OR toLower(f.predicate) CONTAINS term OR
                    toLower(coalesce(f.object_text,'')) CONTAINS term)
                RETURN d.id AS document_id, d.title AS title, d.doi AS doi,
                       d.evidence_mode AS evidence_mode, f.type AS fact_type,
                       e.name AS subject, f.predicate AS predicate, f.object_text AS object_text,
                       f.value AS value, f.unit AS unit, f.conditions_json AS conditions,
                       f.source_sentence AS quote, properties(f)['page'] AS page,
                       properties(f)['table_id'] AS table_id,
                       f.confidence AS confidence
                ORDER BY f.confidence DESC
                LIMIT $limit
                """,
                kb=knowledge_base_id,
                terms=lowered,
                limit=limit,
            )
            return [dict(record) for record in result]
