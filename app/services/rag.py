from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Chunk, Conversation, Document, ExtractedFact, KnowledgeInsight, Message
from app.services.graph_store import GraphStore
from app.services.knowledge_discovery import KnowledgeDiscoveryEngine
from app.services.llm import DeepSeekClient
from app.services.vector_store import VectorStore
from app.services.vocab import VocabularyExpander


class AgentState(TypedDict, total=False):
    original_question: str
    standalone_question: str
    conversation_context: str
    knowledge_base_id: str
    route: Literal["graph", "vector", "hybrid"]
    route_reason: str
    intent: str
    explicit_literature_intent: bool
    graph_evidence: list[dict[str, Any]]
    vector_evidence: list[dict[str, Any]]
    keyword_evidence: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    retrieval_terms: list[str]
    corpus_profile: dict[str, Any]
    existing_insights: list[dict[str, Any]]
    insights: list[dict[str, Any]]
    enable_knowledge_discovery: bool
    research_mode: Literal["deep_research", "evidence_strict", "rapid"]
    conversation_id: str
    needs_literature: bool
    literature_query: str
    literature_reason: str
    answer: str
    tool_calls: list[dict[str, Any]]


@dataclass(slots=True)
class AgentAnswer:
    answer: str
    route: str
    evidence: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    standalone_question: str
    needs_literature: bool
    literature_query: str | None
    literature_reason: str | None
    insights: list[dict[str, Any]]


GRAPH_HINTS = ("材料", "膜", "通量", "渗透率", "截留", "选择性", "压力", "温度", "浓度", "ph", "性能", "关系", "最高", "比较")
VECTOR_HINTS = ("机理", "为什么", "解释", "综述", "进展", "趋势", "挑战", "展望", "总结", "机制")
LITERATURE_HINTS = (
    "找文献",
    "查文献",
    "检索文献",
    "扩充",
    "新增论文",
    "最新论文",
    "相关论文",
    "文献综述",
    "literature",
    "papers",
    "references",
)


def question_terms(question: str) -> list[str]:
    latin = re.findall(r"[A-Za-z][A-Za-z0-9+./-]{1,30}", question)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,8}", question)
    stop = {"哪些", "什么", "如何", "是否", "研究", "文献", "纳滤", "可以", "以及", "之间", "这个", "那些"}
    return list(dict.fromkeys(term for term in [*latin, *chinese] if term.casefold() not in stop))[:20]


class NanofiltrationRAGAgent:
    def __init__(self, db: Session, *, deep_thinking: bool = True):
        self.db = db
        self.settings = get_settings()
        self.deep_thinking = deep_thinking
        self.llm = DeepSeekClient(
            self.settings,
            model_name=self.settings.chat_llm_model,
            enable_thinking=deep_thinking,
        )
        self.graph_store = GraphStore(self.settings)
        self.vector_store = VectorStore(self.settings)

    async def answer(
        self,
        conversation: Conversation,
        question: str,
        *,
        enable_knowledge_discovery: bool = True,
        research_mode: Literal["deep_research", "evidence_strict", "rapid"] = "deep_research",
    ) -> AgentAnswer:
        recent = list(
            reversed(
                list(
                    self.db.scalars(
                        select(Message)
                        .where(Message.conversation_id == conversation.id)
                        .order_by(Message.created_at.desc())
                        .limit(14)
                    )
                )
            )
        )
        recent_context = [{"role": item.role, "content": item.content[:5000]} for item in recent]
        context = json.dumps(
            {"rolling_summary": conversation.rolling_summary, "recent_messages": recent_context},
            ensure_ascii=False,
        )
        workflow = StateGraph(AgentState)
        workflow.add_node("understand", self._understand)
        workflow.add_node("classify", self._classify)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("assess_gap", self._assess_gap)
        workflow.add_node("discover_knowledge", self._discover_knowledge)
        workflow.add_node("synthesize", self._synthesize)
        workflow.add_edge(START, "understand")
        workflow.add_edge("understand", "classify")
        workflow.add_edge("classify", "retrieve")
        workflow.add_edge("retrieve", "assess_gap")
        workflow.add_edge("assess_gap", "discover_knowledge")
        workflow.add_edge("discover_knowledge", "synthesize")
        workflow.add_edge("synthesize", END)
        graph = workflow.compile()
        try:
            state = await graph.ainvoke(
                {
                    "original_question": question,
                    "standalone_question": question,
                    "conversation_context": context,
                    "knowledge_base_id": conversation.knowledge_base_id,
                    "conversation_id": conversation.id,
                    "enable_knowledge_discovery": enable_knowledge_discovery,
                    "research_mode": research_mode,
                    "tool_calls": [
                        {
                            "tool": "conversation_llm",
                            "model": self.llm.model_name,
                            "deep_thinking": self.deep_thinking,
                        }
                    ],
                }
            )
        finally:
            self.graph_store.close()
        return AgentAnswer(
            answer=state["answer"],
            route=state["route"],
            evidence=state.get("evidence", []),
            tool_calls=state.get("tool_calls", []),
            standalone_question=state.get("standalone_question", question),
            needs_literature=bool(state.get("needs_literature")),
            literature_query=state.get("literature_query") or None,
            literature_reason=state.get("literature_reason") or None,
            insights=state.get("insights", []),
        )

    async def maybe_update_rolling_summary(self, conversation: Conversation) -> None:
        if not self.llm.configured:
            return
        count = self.db.scalar(
            select(func.count()).select_from(Message).where(Message.conversation_id == conversation.id)
        ) or 0
        if count < 12 or count % 10 != 0:
            return
        recent = list(
            reversed(
                list(
                    self.db.scalars(
                        select(Message)
                        .where(Message.conversation_id == conversation.id)
                        .order_by(Message.created_at.desc())
                        .limit(12)
                    )
                )
            )
        )
        try:
            summary = await self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": "将纳滤科研对话压缩为滚动摘要，保留研究目标、关键结论、实验条件、争议、DOI 引用和待办；不补充新事实。",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "previous_summary": conversation.rolling_summary,
                                "recent_messages": [{"role": item.role, "content": item.content} for item in recent],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                max_tokens=1200,
                enable_thinking=False,
            )
            conversation.rolling_summary = summary
            self.db.commit()
        except Exception:
            self.db.rollback()

    async def _understand(self, state: AgentState) -> dict[str, Any]:
        question = state["original_question"]
        explicit = any(term in question.casefold() for term in LITERATURE_HINTS)
        fallback = {
            "standalone_question": question,
            "intent": "literature_discovery" if explicit else "knowledge_question",
            "explicit_literature_intent": explicit,
            "tool_calls": [*state.get("tool_calls", []), {"tool": "conversation_context", "mode": "heuristic"}],
        }
        if not self.llm.configured:
            return fallback
        try:
            result = await self.llm.json_chat(
                system=(
                    "你是纳滤领域 Agent 的问题理解器。结合滚动摘要和近期对话，解析代词、省略和追问，"
                    "把本轮问题改写成可独立检索的完整问题。判断用户是否明确要求找/新增/更新文献。"
                    "返回 standalone_question、intent、explicit_literature_intent。不得改变用户研究对象和条件。"
                ),
                user=json.dumps(
                    {"context": json.loads(state["conversation_context"]), "question": question},
                    ensure_ascii=False,
                ),
                max_tokens=800,
                enable_thinking=False,
            )
            if isinstance(result, dict):
                standalone = str(result.get("standalone_question") or question).strip()[:2000]
                explicit = bool(result.get("explicit_literature_intent")) or explicit
                return {
                    "standalone_question": standalone,
                    "intent": str(result.get("intent") or fallback["intent"])[:100],
                    "explicit_literature_intent": explicit,
                    "tool_calls": [*state.get("tool_calls", []), {"tool": "conversation_context", "mode": "deepseek"}],
                }
        except Exception:
            pass
        return fallback

    async def _classify(self, state: AgentState) -> dict[str, Any]:
        question = state["standalone_question"]
        folded = question.casefold()
        graph_hit = any(term in folded for term in GRAPH_HINTS)
        vector_hit = any(term in folded for term in VECTOR_HINTS)
        heuristic: Literal["graph", "vector", "hybrid"] = "hybrid" if graph_hit and vector_hit else "graph" if graph_hit else "vector" if vector_hit else "hybrid"
        if not self.llm.configured:
            return {"route": heuristic, "route_reason": "关键词规则路由"}
        try:
            result = await self.llm.json_chat(
                system=(
                    "判断纳滤问题的检索路由。材料、性能、实验条件和实体关系优先 graph；"
                    "机理解释、综述、趋势优先 vector；跨类型或比较推理用 hybrid。"
                    "返回 route 和 reason。"
                ),
                user=question,
                max_tokens=300,
                enable_thinking=False,
            )
            route = result.get("route") if isinstance(result, dict) else None
            if route in {"graph", "vector", "hybrid"}:
                return {"route": route, "route_reason": str(result.get("reason") or "DeepSeek 路由")}
        except Exception:
            pass
        return {"route": heuristic, "route_reason": "关键词规则路由"}

    async def _retrieve(self, state: AgentState) -> dict[str, Any]:
        route = state["route"]
        kb = state["knowledge_base_id"]
        question = state["standalone_question"]
        mode = state.get("research_mode", "deep_research")
        graph_limit = 120 if mode == "evidence_strict" else 80 if mode == "deep_research" else 20
        vector_limit = max(
            self.settings.retrieval_top_k,
            80 if mode == "evidence_strict" else 60 if mode == "deep_research" else 16,
        )
        keyword_limit = 80 if mode == "evidence_strict" else 60 if mode == "deep_research" else 12
        fused_limit = 56 if mode == "evidence_strict" else 40 if mode == "deep_research" else 20

        expander = VocabularyExpander(llm=self.llm)
        try:
            expanded = await expander.expand(question)
        except Exception:
            expanded = expander.local_expand(question)
        terms = list(
            dict.fromkeys(
                term.strip()
                for term in [
                    *question_terms(question),
                    *(item for values in expanded.values() for item in values),
                ]
                if term and term.strip()
            )
        )[:120]
        corpus_profile = self._scan_corpus(kb, terms)
        graph_evidence = self._graph_search(kb, terms, graph_limit) if route in {"graph", "hybrid"} else []
        if route in {"vector", "hybrid"}:
            try:
                vector_evidence = self.vector_store.search(kb, question, vector_limit)
            except Exception:
                vector_evidence = []
        else:
            vector_evidence = []
        keyword_evidence, keyword_scan = self._keyword_search(kb, terms, keyword_limit)
        corpus_profile.update(keyword_scan)
        fused = self._fuse(graph_evidence, vector_evidence, keyword_evidence, fused_limit * 3)
        evidence = self._diversify_evidence(fused, fused_limit)
        existing_insights = self._insight_search(kb, terms, 10 if mode != "rapid" else 4)
        tool_calls = [
            *state.get("tool_calls", []),
            {
                "tool": "complete_library_inventory_scan",
                **corpus_profile,
                "note": "逐篇扫描整个知识库清单后，再压缩与问题相关的可引用证据",
            },
            {
                "tool": "bilingual_nf_vocabulary_expansion",
                "term_count": len(terms),
                "sample": terms[:40],
            },
            {"tool": "neo4j_or_sql_facts", "count": len(graph_evidence), "route": route},
            {"tool": "chroma_bge_m3", "count": len(vector_evidence), "route": route},
            {
                "tool": "exhaustive_fulltext_keyword_scan",
                "count": len(keyword_evidence),
                "route": route,
                **keyword_scan,
            },
            {"tool": "ai_insight_memory", "count": len(existing_insights), "epistemic_role": "hypothesis_not_fact"},
            {"tool": "research_mode", "mode": mode, "evidence_policy": "knowledge_base_only"},
        ]
        return {
            "graph_evidence": graph_evidence,
            "vector_evidence": vector_evidence,
            "keyword_evidence": keyword_evidence,
            "evidence": evidence,
            "retrieval_terms": terms,
            "corpus_profile": corpus_profile,
            "existing_insights": existing_insights,
            "tool_calls": tool_calls,
        }

    async def _discover_knowledge(self, state: AgentState) -> dict[str, Any]:
        enabled = bool(state.get("enable_knowledge_discovery"))
        evidence = state.get("evidence", [])
        if not enabled or len(evidence) < 3 or not self.llm.configured:
            return {
                "insights": [],
                "tool_calls": [
                    *state.get("tool_calls", []),
                    {
                        "tool": "cross_document_knowledge_discovery",
                        "enabled": enabled,
                        "created": 0,
                        "reason": "证据不足或模型未配置" if enabled else "用户关闭",
                    },
                ],
            }
        conversation = self.db.get(Conversation, state["conversation_id"])
        if not conversation:
            return {"insights": []}
        engine = KnowledgeDiscoveryEngine(
            self.db,
            settings=self.settings,
            llm=self.llm,
            graph_store=self.graph_store,
        )
        rows = await engine.discover(conversation, state["standalone_question"], evidence)
        insights = [self._insight_to_dict(item) for item in rows]
        return {
            "insights": insights,
            "tool_calls": [
                *state.get("tool_calls", []),
                {
                    "tool": "cross_document_knowledge_discovery",
                    "enabled": True,
                    "created": len(insights),
                    "epistemic_role": "ai_synthesis_or_hypothesis",
                },
            ],
        }

    async def _assess_gap(self, state: AgentState) -> dict[str, Any]:
        evidence = state.get("evidence", [])
        explicit = bool(state.get("explicit_literature_intent"))
        mode = state.get("research_mode", "deep_research")
        profile = self._evidence_profile(evidence)
        corpus_profile = state.get("corpus_profile", {})
        if mode == "evidence_strict":
            low_coverage = profile["count"] < 6 or profile["unique_documents"] < 3 or profile["fulltext_documents"] < 2
        elif mode == "rapid":
            low_coverage = profile["count"] < 2 or profile["unique_documents"] < 1
        else:
            low_coverage = profile["count"] < 5 or profile["unique_documents"] < 2 or profile["fulltext_documents"] < 1
        fallback_reason = "用户明确要求发现文献" if explicit else "当前知识库证据不足" if low_coverage else ""
        fallback_query = state["standalone_question"][:1000]
        needs = explicit or low_coverage
        if self.llm.configured:
            try:
                result = await self.llm.json_chat(
                    system=(
                        "你是纳滤知识库覆盖度审核器。根据完整问题和已检索证据摘要，判断是否需要发现新文献。"
                        "用户明确要求文献时必须为 true；证据少、过时、缺实验条件或无法支持关键结论时为 true。"
                        "search_query 必须是简洁、可用于学术检索的纳滤主题，保留材料/工艺/体系/性能条件。"
                        "返回 needs_literature、search_query、reason。"
                    ),
                    user=json.dumps(
                        {
                            "question": state["standalone_question"],
                            "research_mode": mode,
                            "explicit_literature_intent": explicit,
                            "corpus_profile": corpus_profile,
                            "evidence_profile": profile,
                            "evidence": [
                                {"title": item.get("title"), "doi": item.get("doi"), "quote": (item.get("quote") or "")[:500]}
                                for item in evidence[:8]
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    max_tokens=600,
                    enable_thinking=False,
                )
                if isinstance(result, dict):
                    needs = bool(result.get("needs_literature")) or explicit or low_coverage
                    fallback_query = str(result.get("search_query") or fallback_query).strip()[:1000]
                    fallback_reason = str(result.get("reason") or fallback_reason).strip()[:1000]
            except Exception:
                pass
        return {
            "needs_literature": needs,
            "literature_query": fallback_query if needs else "",
            "literature_reason": fallback_reason if needs else "",
            "tool_calls": [
                *state.get("tool_calls", []),
                {
                    "tool": "knowledge_gap_assessment",
                    "needs_literature": needs,
                    "corpus_profile": corpus_profile,
                    "evidence_profile": profile,
                    "research_mode": mode,
                    "search_query": fallback_query if needs else None,
                },
            ],
        }

    async def _synthesize(self, state: AgentState) -> dict[str, Any]:
        evidence = state.get("evidence", [])
        if not evidence:
            corpus = state.get("corpus_profile", {})
            scanned = int(corpus.get("documents_scanned") or 0)
            answer = (
                f"本轮已经逐篇扫描当前知识库的 {scanned} 篇文献，并联查图谱事实、全文切片和题名/摘要，"
                "但没有找到能直接支撑本问题的可引用证据。为避免编造，我不会把模型记忆冒充知识库结论。"
            )
            if state.get("needs_literature"):
                answer += (
                    f"\n\n建议补充主题：{state.get('literature_query')}。新文献经你确认并入库后，"
                    "后续回答仍会重新检索整个知识库，而不是只查看本轮新增文献。"
                )
            return {"answer": answer}
        if not self.llm.configured:
            lines = [f"已检索到以下原文证据；配置 SILICONFLOW_API_KEY 后可由 {self.llm.model_name} 综合回答："]
            for index, item in enumerate(evidence[:8], 1):
                lines.append(f"{index}. {item.get('quote') or item.get('source_sentence', '')} {self._citation(item)}")
            return {"answer": "\n\n".join(lines)}
        evidence_payload = [
            {
                "id": index,
                "document_id": item.get("document_id"),
                "title": item.get("title"),
                "doi": item.get("doi"),
                "page": self._page(item),
                "table_id": item.get("table_id"),
                "section": item.get("section"),
                "quote": item.get("quote") or item.get("source_sentence"),
                "evidence_mode": item.get("evidence_mode") or "unknown",
                "fact": {key: item.get(key) for key in ("subject", "predicate", "object_text", "value", "unit", "conditions") if item.get(key) is not None},
            }
            for index, item in enumerate(evidence, 1)
        ]
        insight_payload = [*state.get("insights", []), *state.get("existing_insights", [])][:10]
        answer = await self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是 NF-Atlas 纳滤知识发现 Agent。知识库证据是唯一事实来源；DeepSeek 只负责问题理解、"
                        "工具编排、跨文献比较、因果分析和形成可验证假设，绝不能用模型参数记忆补充事实。"
                        "系统已逐篇扫描当前知识库的文献清单，并用纳滤双语词表、图谱、向量和关键词联查；"
                        "输入 evidence 是从全库中压缩出的高相关可引用证据，不是只检索本轮新增文献。"
                        "不得声称逐字阅读全文库；应准确表述为全库扫描与相关证据聚合。"
                        "延续对话上下文，但事实结论只依据给定知识库证据。"
                        "区分事实、跨文献比较和合理推断。每个关键结论标注 [DOI, p.页码]；"
                        "无页码写 [DOI, 页码未解析]，无 DOI 写 [题名, p.页码]。数值同时说明实验条件；"
                        "题名/摘要模式的证据必须明确称为摘要证据，不得假装读过全文。不得编造 DOI、页码或实验数据。"
                        "如果提供了 ai_insights，必须另设‘AI跨文献推断（尚未验证）’部分；pattern/contradiction 只能称为归纳，"
                        "hypothesis 必须称为假设，并列出置信度、边界条件和验证/证伪实验。"
                        "AI insight 不是论文证据，不能用它替代 DOI/原文引用；status=validated 也要说明其审核来源。"
                        "回答优先组织为：证据支持的结论、跨文献知识发现、可验证的新假设、证据边界与下一步。"
                        "不要为了填满结构而重复内容；如果证据不能支持某项，明确写证据不足。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "conversation_context": json.loads(state["conversation_context"]),
                            "original_question": state["original_question"],
                            "standalone_question": state["standalone_question"],
                            "route": state["route"],
                            "research_mode": state.get("research_mode", "deep_research"),
                            "corpus_profile": state.get("corpus_profile", {}),
                            "retrieval_terms": state.get("retrieval_terms", [])[:60],
                            "evidence_profile": self._evidence_profile(evidence),
                            "evidence": evidence_payload,
                            "ai_insights": insight_payload,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=4500,
        )
        return {"answer": answer}

    @staticmethod
    def _evidence_profile(evidence: list[dict[str, Any]]) -> dict[str, int]:
        documents = {
            str(item.get("document_id") or item.get("doi") or item.get("title"))
            for item in evidence
            if item.get("document_id") or item.get("doi") or item.get("title")
        }
        fulltext_documents = {
            str(item.get("document_id") or item.get("doi") or item.get("title"))
            for item in evidence
            if item.get("evidence_mode") != "metadata-only"
            and (item.get("document_id") or item.get("doi") or item.get("title"))
        }
        return {
            "count": len(evidence),
            "unique_documents": len(documents),
            "fulltext_documents": len(fulltext_documents),
            "page_located": sum(cls_item.get("page") is not None or cls_item.get("page_start") is not None for cls_item in evidence),
            "metadata_only": sum(cls_item.get("evidence_mode") == "metadata-only" for cls_item in evidence),
        }

    def _scan_corpus(self, kb: str, terms: list[str]) -> dict[str, Any]:
        """Inspect every persisted document before ranking evidence for the model context."""
        rows = list(
            self.db.execute(
                select(
                    Document.id,
                    Document.title,
                    Document.abstract,
                    Document.publication_year,
                    Document.status,
                    Document.object_key,
                    Document.metadata_json,
                ).where(Document.knowledge_base_id == kb)
            )
        )
        folded_terms = [item.casefold() for item in terms if len(item.strip()) >= 2]
        lexical_matches = 0
        for row in rows:
            haystack = f"{row.title or ''}\n{row.abstract or ''}".casefold()
            if any(term in haystack for term in folded_terms):
                lexical_matches += 1
        metadata_only = sum(bool((row.metadata_json or {}).get("metadata_only")) for row in rows)
        fulltext = sum(
            bool(row.object_key) and not bool((row.metadata_json or {}).get("metadata_only"))
            for row in rows
        )
        indexed = sum(getattr(row.status, "value", str(row.status)) == "indexed" for row in rows)
        years = [row.publication_year for row in rows if isinstance(row.publication_year, int)]
        chunk_count = self.db.scalar(
            select(func.count(Chunk.id))
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.knowledge_base_id == kb)
        ) or 0
        fact_count = self.db.scalar(
            select(func.count(ExtractedFact.id))
            .join(Document, Document.id == ExtractedFact.document_id)
            .where(Document.knowledge_base_id == kb)
        ) or 0
        return {
            "documents_scanned": len(rows),
            "indexed_documents": indexed,
            "fulltext_documents": fulltext,
            "metadata_only_documents": metadata_only,
            "lexically_matched_documents": lexical_matches,
            "total_chunks": int(chunk_count),
            "total_facts": int(fact_count),
            "year_min": min(years) if years else None,
            "year_max": max(years) if years else None,
        }

    def _graph_search(self, kb: str, terms: list[str], limit: int) -> list[dict[str, Any]]:
        if self.graph_store.configured:
            try:
                rows = self.graph_store.search_facts(kb, terms, limit)
                for row in rows:
                    row["source"] = "graph"
                if rows:
                    return rows
            except Exception:
                pass
        if not terms:
            return []
        filters = []
        for term in terms[:50]:
            pattern = f"%{term}%"
            filters.extend(
                [
                    ExtractedFact.subject.ilike(pattern),
                    ExtractedFact.predicate.ilike(pattern),
                    ExtractedFact.object_text.ilike(pattern),
                    ExtractedFact.source_sentence.ilike(pattern),
                ]
            )
        rows = self.db.execute(
            select(ExtractedFact, Document)
            .join(Document, Document.id == ExtractedFact.document_id)
            .where(Document.knowledge_base_id == kb, or_(*filters))
            .order_by(ExtractedFact.confidence.desc())
            .limit(limit)
        )
        return [
            {
                "source": "sql_facts",
                "document_id": document.id,
                "title": document.title,
                "doi": document.doi_normalized,
                "evidence_mode": "metadata-only" if (document.metadata_json or {}).get("metadata_only") else "fulltext",
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object_text": fact.object_text,
                "value": fact.normalized_value if fact.normalized_value is not None else fact.value,
                "unit": fact.normalized_unit or fact.unit,
                "conditions": fact.conditions,
                "quote": fact.source_sentence,
                "page": fact.page,
                "table_id": fact.table_id,
                "confidence": fact.confidence,
            }
            for fact, document in rows
        ]

    def _keyword_search(
        self,
        kb: str,
        terms: list[str],
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Read every persisted chunk, then keep a diverse evidence-sized result set.

        The limit controls only how many excerpts enter the LLM context. It does not limit
        how many knowledge-base chunks are inspected.
        """
        generic = {
            "nanofiltration",
            "nanofiltration membrane",
            "nanofiltration membranes",
            "membrane",
            "membranes",
            "纳滤",
            "纳滤膜",
        }
        normalized = list(
            dict.fromkeys(
                term.strip().casefold()
                for term in terms
                if term and len(term.strip()) >= 2
            )
        )
        specific = [term for term in normalized if term not in generic]
        active_terms = (specific or normalized)[:100]
        if not active_terms:
            return [], {"chunks_scanned": 0, "content_matched_chunks": 0, "content_matched_documents": 0}

        rows = self.db.execute(
            select(Chunk, Document)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.knowledge_base_id == kb)
            .execution_options(yield_per=500)
        )
        candidates: list[dict[str, Any]] = []
        scanned = 0
        matched_documents: set[str] = set()
        for chunk, document in rows:
            scanned += 1
            folded_text = chunk.text.casefold()
            score = 0.0
            for term in active_terms:
                occurrences = folded_text.count(term)
                if occurrences:
                    score += min(occurrences, 3) * (1.0 + min(len(term), 20) / 20.0)
            if score <= 0:
                continue
            matched_documents.add(document.id)
            title_folded = document.title.casefold()
            score += sum(0.5 for term in active_terms if term in title_folded)
            candidates.append(
                {
                    "source": "exhaustive_keyword",
                    "document_id": document.id,
                    "title": document.title,
                    "doi": document.doi_normalized,
                    "evidence_mode": (
                        "metadata-only" if (document.metadata_json or {}).get("metadata_only") else "fulltext"
                    ),
                    "quote": chunk.text,
                    "section": chunk.section,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "chunk_id": chunk.id,
                    "score": round(score, 5),
                }
            )
        candidates.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
        selected = self._diversify_evidence(candidates, limit)
        return selected, {
            "chunks_scanned": scanned,
            "content_matched_chunks": len(candidates),
            "content_matched_documents": len(matched_documents),
        }

    def _insight_search(self, kb: str, terms: list[str], limit: int) -> list[dict[str, Any]]:
        if not terms:
            return []
        filters = []
        for term in terms[:10]:
            pattern = f"%{term}%"
            filters.extend(
                [
                    KnowledgeInsight.title.ilike(pattern),
                    KnowledgeInsight.claim.ilike(pattern),
                    KnowledgeInsight.rationale.ilike(pattern),
                ]
            )
        rows = self.db.scalars(
            select(KnowledgeInsight)
            .where(
                KnowledgeInsight.knowledge_base_id == kb,
                KnowledgeInsight.status != "rejected",
                or_(*filters),
            )
            .order_by(KnowledgeInsight.confidence.desc(), KnowledgeInsight.created_at.desc())
            .limit(limit)
        )
        return [self._insight_to_dict(item) for item in rows]

    @staticmethod
    def _insight_to_dict(item: KnowledgeInsight) -> dict[str, Any]:
        return {
            "id": item.id,
            "type": item.insight_type,
            "title": item.title,
            "claim": item.claim,
            "rationale": item.rationale,
            "evidence_refs": item.evidence_refs,
            "assumptions": item.assumptions,
            "boundary_conditions": item.boundary_conditions,
            "validation_plan": item.validation_plan,
            "confidence": item.confidence,
            "novelty_score": item.novelty_score,
            "status": item.status,
        }

    @staticmethod
    def _fuse(*args: Any) -> list[dict[str, Any]]:
        *groups, limit = args
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        positions = [0] * len(groups)
        while len(output) < limit:
            added = False
            for index, group in enumerate(groups):
                if positions[index] >= len(group):
                    continue
                item = group[positions[index]]
                positions[index] += 1
                key = str(item.get("chunk_id") or (item.get("doi"), item.get("quote")))
                if key in seen:
                    continue
                seen.add(key)
                output.append(item)
                added = True
                if len(output) >= limit:
                    break
            if not added:
                break
        return output

    @staticmethod
    def _diversify_evidence(evidence: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """Give independent papers a first pass, then fill remaining slots by relevance."""
        output: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        seen_documents: set[str] = set()
        for item in evidence:
            document_key = str(item.get("document_id") or item.get("doi") or item.get("title") or "")
            if document_key and document_key not in seen_documents:
                seen_documents.add(document_key)
                output.append(item)
            else:
                deferred.append(item)
            if len(output) >= limit:
                return output
        output.extend(deferred[: max(0, limit - len(output))])
        return output

    @staticmethod
    def _page(item: dict[str, Any]) -> int | None:
        value = item.get("page") if item.get("page") is not None else item.get("page_start")
        return value if isinstance(value, int) and value >= 0 else None

    @classmethod
    def _citation(cls, item: dict[str, Any]) -> str:
        source = item.get("doi") or item.get("title") or "来源未知"
        page = cls._page(item)
        return f"[{source}, {'p.' + str(page) if page else '页码未解析'}]"
