from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.services.fulltext import ElsevierAuthenticationError, ElsevierEntitlementError, FullTextResolver
from app.services.literature import LiteratureDiscovery
from app.services.llm import DeepSeekClient


async def main() -> None:
    settings = get_settings()
    report: dict[str, object] = {}

    if settings.siliconflow_api_key:
        try:
            answer = await DeepSeekClient(settings).chat(
                [
                    {"role": "system", "content": "You are a connectivity check."},
                    {"role": "user", "content": "Reply with exactly: OK"},
                ],
                temperature=0,
                max_tokens=8,
            )
            report["siliconflow"] = {"ok": bool(answer.strip()), "model": settings.llm_model}
        except Exception as exc:
            report["siliconflow"] = {
                "ok": False,
                "error": {"type": type(exc).__name__, "status_code": getattr(exc, "status_code", None)},
            }
    else:
        report["siliconflow"] = {"ok": False, "error": "SILICONFLOW_API_KEY is missing"}

    discovery = LiteratureDiscovery(settings)
    try:
        if settings.elsevier_api_key:
            try:
                rows = await discovery._search_elsevier("nanofiltration membrane", 1, None, None)
                report["elsevier"] = {
                    "key_configured": True,
                    "metadata_search": {"ok": True, "result_count": len(rows)},
                    "insttoken_configured": bool(settings.elsevier_insttoken),
                }
            except Exception as exc:
                report["elsevier"] = {
                    "key_configured": True,
                    "metadata_search": {"ok": False, "error": discovery._safe_error(exc)},
                    "insttoken_configured": bool(settings.elsevier_insttoken),
                }
        else:
            report["elsevier"] = {"ok": False, "error": "ELSEVIER_API_KEY is missing"}

        if settings.openalex_api_key:
            try:
                rows = await discovery._search_openalex("nanofiltration membrane", 1, None, None)
                report["openalex"] = {"ok": True, "result_count": len(rows)}
            except Exception as exc:
                report["openalex"] = {"ok": False, "error": discovery._safe_error(exc)}
        else:
            report["openalex"] = {"ok": False, "error": "OPENALEX_API_KEY is missing"}
    finally:
        await discovery.close()

    if settings.elsevier_api_key and isinstance(report.get("elsevier"), dict):
        elsevier_report = report["elsevier"]
        assert isinstance(elsevier_report, dict)
        resolver = FullTextResolver(settings)
        try:
            # DOI used in Elsevier's own TDM guide. This distinguishes a valid
            # metadata key from FULL-view institutional entitlement.
            result = await resolver._from_elsevier("10.1016/j.ibusrev.2010.09.002")
            elsevier_report["fulltext_tdm"] = {
                "ok": result is not None,
                "content_mode": "FULL XML" if result else "not_found",
            }
            elsevier_report["key_recognized_by_article_api"] = result is not None
        except ElsevierAuthenticationError as exc:
            elsevier_report["fulltext_tdm"] = {
                "ok": False,
                "reason": "api_key_authentication",
                "message": str(exc),
            }
            elsevier_report["key_recognized_by_article_api"] = False
        except ElsevierEntitlementError as exc:
            elsevier_report["fulltext_tdm"] = {
                "ok": False,
                "reason": "institutional_entitlement",
                "message": str(exc),
            }
            elsevier_report["key_recognized_by_article_api"] = True
        except Exception as exc:
            elsevier_report["fulltext_tdm"] = {
                "ok": False,
                "reason": type(exc).__name__,
            }
        finally:
            await resolver.close()

    if settings.unpaywall_email:
        async with httpx.AsyncClient(timeout=20) as client:
            try:
                response = await client.get(
                    "https://api.unpaywall.org/v2/10.1038/nature12373",
                    params={"email": settings.unpaywall_email},
                )
                response.raise_for_status()
                report["unpaywall"] = {"ok": True, "is_oa": bool(response.json().get("is_oa"))}
            except Exception as exc:
                report["unpaywall"] = {"ok": False, "error": LiteratureDiscovery._safe_error(exc)}
    else:
        report["unpaywall"] = {"ok": False, "error": "UNPAYWALL_EMAIL is missing"}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
