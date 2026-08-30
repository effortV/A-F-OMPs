from __future__ import annotations

import json
from pathlib import Path

from app.services.llm import DeepSeekClient, LLMNotConfigured

CATEGORIES = ("zh", "en", "abbreviations", "broader", "narrower", "materials", "methods", "systems", "metrics")


class VocabularyExpander:
    def __init__(self, llm: DeepSeekClient | None = None, vocab_path: Path | None = None):
        self.llm = llm or DeepSeekClient()
        self.vocab_path = vocab_path or Path(__file__).parents[2] / "data" / "nanofiltration_vocab.json"
        self.vocab = json.loads(self.vocab_path.read_text(encoding="utf-8"))

    def local_expand(self, query: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {category: [] for category in CATEGORIES}
        result["zh"].append(query.strip())
        lowered = query.casefold()
        for key, values in self.vocab.items():
            haystack = [key, *[str(item) for group in values.values() for item in group]]
            if key in query or query in key or any(lowered in item.casefold() for item in haystack):
                result["zh"].append(key)
                for category, items in values.items():
                    target = "abbreviations" if category == "abbr" else category
                    if target in result:
                        result[target].extend(str(item) for item in items)
        # All NF searches should retain the domain anchor to control false positives.
        result["en"].extend(["nanofiltration", "nanofiltration membrane"])
        result["zh"].append("纳滤")
        return {key: list(dict.fromkeys(item.strip() for item in values if item.strip())) for key, values in result.items()}

    async def expand(self, query: str) -> dict[str, list[str]]:
        local = self.local_expand(query)
        if not self.llm.configured:
            return local
        prompt = {
            "query": query,
            "local_vocabulary": local,
            "required_keys": list(CATEGORIES),
        }
        try:
            remote = await self.llm.json_chat(
                system=(
                    "你是纳滤膜领域的检索词工程师。扩展中英文同义词、标准缩写、上下位概念、"
                    "膜材料、制备方法、分离体系和性能指标。严格限制在用户主题和纳滤领域，避免泛化。"
                    "返回对象，每个键的值均为字符串数组。"
                ),
                user=json.dumps(prompt, ensure_ascii=False),
                max_tokens=2500,
                enable_thinking=False,
            )
        except (LLMNotConfigured, ValueError, json.JSONDecodeError):
            return local
        if not isinstance(remote, dict):
            return local
        merged: dict[str, list[str]] = {}
        for key in CATEGORIES:
            extra = remote.get(key, [])
            if not isinstance(extra, list):
                extra = []
            merged[key] = list(dict.fromkeys([*local[key], *(str(item).strip() for item in extra if item)]))[:40]
        return merged

    @staticmethod
    def search_queries(expanded: dict[str, list[str]], max_queries: int = 8) -> list[str]:
        anchors = expanded.get("zh", [])[:2] + expanded.get("en", [])[:5]
        qualifiers = (
            expanded.get("materials", [])[:2]
            + expanded.get("methods", [])[:2]
            + expanded.get("systems", [])[:2]
        )
        queries = []
        for term in anchors:
            domain = "nanofiltration" if "nanofiltration" not in term.casefold() and "纳滤" not in term else ""
            queries.append(" ".join(item for item in (term, domain) if item))
        if qualifiers:
            queries.append("nanofiltration " + " OR ".join(qualifiers[:4]))
        return list(dict.fromkeys(queries))[:max_queries]
