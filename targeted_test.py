from __future__ import annotations

import time
import sys
import types
from pathlib import Path
from unittest.mock import patch

try:
    import xgboost  # noqa: F401
except ModuleNotFoundError:
    # The lightweight desktop test runtime omits xgboost.  These tests do not
    # train models; the stub only lets discovery tool modules be imported.
    sys.modules["xgboost"] = types.SimpleNamespace(XGBRegressor=object)

from zhizhi.core import db, jobs
from zhizhi.core.config import CFG
from zhizhi.core.llm import LLM
from zhizhi.core.tools import report_tool_progress
from zhizhi.lit import dedup, extract, kg, search, worker
from zhizhi.tools import disc_tools, lit_tools

db.init()

cols = {r[1] for r in db.q("PRAGMA table_info(papers)")}
assert {"doi_key", "title_key", "content_hash"} <= cols
assert CFG.llm_model == "deepseek-ai/DeepSeek-V4-Pro"
assert CFG.literature_preprocess_model == "Pro/deepseek-ai/DeepSeek-V3.2"
assert CFG.get("data.mechanism_features") == ["ΦS", "ΦD", "∆Gs-m (J·m-2)"]
assert CFG.get("llm.max_tool_iters") == 8
assert CFG.get("llm.consult_max_depth") == 1
assert CFG.get("llm.max_agent_context_tokens") == 18000
assert CFG.get("llm.chat_thinking") is True
assert CFG.get("discovery.cross_domain_max_seconds") == 1200
assert CFG.get("discovery.cross_domain_llm_timeout") == 1000
assert CFG.get("literature.novelty_cache_hours") == 24
assert CFG.get("llm.max_tokens") is None
assert search.SEARCH_API_VERSION >= 2

# Normal model calls must not send an output-token ceiling.  A clipped JSON
# response is unusable even though the provider still charges for it.
captured_requests = []
fake_client = types.SimpleNamespace(
    chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(
            create=lambda **kwargs: (
                captured_requests.append(kwargs)
                or types.SimpleNamespace(
                    usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                    choices=[types.SimpleNamespace(
                        message=types.SimpleNamespace(content="ok"))])))))
with patch.object(LLM, "_client", fake_client), patch.object(db, "log_usage"):
    LLM("test", fallbacks=[]).chat(
        [{"role": "user", "content": "test"}], thinking=False, attempts=1)
assert "max_tokens" not in captured_requests[0]

# The standalone "先看候选领域" action keeps its original unrestricted path;
# the nested background proposal may have a time budget but never a token cap.
proposal_calls = []


class FakeProposalLLM:
    def __init__(self, *args, **kwargs):
        pass

    def ask_json(self, *args, **kwargs):
        proposal_calls.append(kwargs)
        if len(proposal_calls) == 1:
            return [{"domain": "array-domain", "promise": 0.9},
                    {"domain": "second-domain", "promise": 0.2}]
        return {"candidates": [{"domain": "test-domain"}],
                "recommended": "test-domain", "reasoning": "test"}


with patch("zhizhi.core.llm.LLM", FakeProposalLLM):
    direct_proposal = disc_tools.disc_propose_domains(context="test")
    bounded_proposal = disc_tools.disc_propose_domains(context="test", max_seconds=12)
assert direct_proposal["recommended"] == "array-domain"
assert len(direct_proposal["candidates"]) == 2
assert bounded_proposal["recommended"] == "test-domain"
assert "max_tokens" not in proposal_calls[0]
assert "request_timeout" not in proposal_calls[0]
assert "max_tokens" not in proposal_calls[1]
assert proposal_calls[1]["request_timeout"] == 12

cheap = LLM("test", model=CFG.literature_preprocess_model,
            fallbacks=[CFG.literature_preprocess_model], usage_kind="literature_preprocess")
critical = LLM("test", model=CFG.llm_model, fallbacks=[CFG.llm_model])
assert cheap.model.endswith("DeepSeek-V3.2") and cheap.fallbacks == [cheap.model]
assert critical.model.endswith("DeepSeek-V4-Pro") and critical.fallbacks == [critical.model]

routes = []


class FakeRouteLLM:
    def __init__(self, agent, model=None, fallbacks=None, usage_kind="chat"):
        routes.append((usage_kind, model, list(fallbacks or [])))
        self.usage_kind = usage_kind

    def ask_json(self, system, user, **kwargs):
        if self.usage_kind == "literature_preprocess":
            return {"title": "T", "year": 2024, "journal": "J", "doi": "10.1/x",
                    "membranes": [], "compounds": [], "conditions": {},
                    "evidence_chunk_ids": [0]}
        if self.usage_kind == "literature_relevance":
            return {"scored": [{"i": 0, "relevance": 8,
                                 "reason": "relevant", "evidence_type": "supports"}]}
        return {"key_findings": [], "mechanism_claims": [], "limitations": [],
                "anomalies": [], "kg_triples": []}


paper = db.q1("SELECT id FROM papers WHERE n_chunks>0 LIMIT 1")
assert paper
with patch.object(extract, "LLM", FakeRouteLLM), patch.object(extract, "persist"):
    extract.extract_paper(paper["id"])
with patch.object(search, "LLM", FakeRouteLLM):
    search.score_relevance([{"title": "T", "abstract": "A"}], "question")
    search.expand_queries("question")
assert any(kind == "literature_preprocess" and model == CFG.literature_preprocess_model
           for kind, model, _ in routes)
assert any(kind == "literature_semantic" and model == CFG.llm_model
           for kind, model, _ in routes)
assert any(kind == "literature_relevance" and model == CFG.llm_model
           for kind, model, _ in routes)
assert any(kind == "literature_query_design" and model == CFG.literature_preprocess_model
           for kind, model, _ in routes)

assert dedup.normalize_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
assert dedup.normalize_title("A title: test") == dedup.normalize_title("A TITLE — TEST")
assert kg.node_id("Membrane", "NF270") == kg.node_id("Membrane", "NF 270")
assert kg.node_id("Membrane", "NF270") == kg.node_id("Membrane", "NF-270")
assert extract.safe_page(18) == 18
assert extract.safe_page("p.18") == 18
assert extract.safe_page("page 8") == 8
assert extract.safe_page("unknown") == 0
assert extract.safe_confidence("confidence=0.82") == 0.82
assert worker.failure_category("ValueError: invalid literal for int() with base 10: 'p.18'") == "page_format"
assert worker.failure_category("无 PDF 也无摘要") == "missing_text"

worker_status = worker.status()
assert worker_status["total_papers"] == db.q1("SELECT COUNT(*) c FROM papers")["c"]
assert "deleted" not in worker_status["queue"]
assert "deleted" not in worker_status["task_queue"]
assert isinstance(worker_status["failure_categories"], list)

report = dedup.deduplicate_library(dry_run=True)
assert "duplicate_groups" in report and report["dry_run"] is True


default_thinking_values = []


class FakeAgent:
    def run(self, session_id, prompt, thinking=False, **kwargs):
        default_thinking_values.append(thinking)
        time.sleep(0.12)
        yield {"type": "delta", "text": prompt}
        yield {"type": "text", "text": prompt}
        yield {"type": "done", "steps": 0}


with patch.object(jobs, "_agent_factory", return_value=FakeAgent()):
    a = jobs.submit("bowen", f"test-a-{time.time_ns()}", "A")
    b = jobs.submit("gewu", f"test-b-{time.time_ns()}", "B")
    same = jobs.submit("bowen", a["job"]["session_id"], "same-session")
    assert same.get("error")
    deadline = time.time() + 5
    while time.time() < deadline:
        ja, jb = jobs.get(a["job"]["id"]), jobs.get(b["job"]["id"])
        if ja["state"] == jb["state"] == "done":
            break
        time.sleep(0.03)
    assert ja["state"] == jb["state"] == "done"
assert default_thinking_values and all(default_thinking_values)
db.ex("DELETE FROM tasks WHERE kind='agent_run' AND ref IN (?,?)",
      (a["job"]["id"], b["job"]["id"]))


class CancellableAgent:
    def run(self, session_id, prompt, thinking=False, should_cancel=None, **kwargs):
        for _ in range(100):
            if should_cancel and should_cancel():
                yield {"type": "cancelled", "text": "test cancelled"}
                return
            time.sleep(0.01)
            yield {"type": "reasoning", "text": "."}


with patch.object(jobs, "_agent_factory", return_value=CancellableAgent()):
    pending = jobs.submit("gewu", f"cancel-{time.time_ns()}", "cancel me")
    jobs.cancel(pending["job"]["id"])
    deadline = time.time() + 3
    while time.time() < deadline:
        stopped = jobs.get(pending["job"]["id"])
        if stopped["state"] == "cancelled":
            break
        time.sleep(0.02)
    assert stopped["state"] == "cancelled"
db.ex("DELETE FROM tasks WHERE kind='agent_run' AND ref=?", (pending["job"]["id"],))


class ProgressAgent:
    def run(self, session_id, prompt, thinking=False, **kwargs):
        yield {"type": "tool_call", "name": "disc_crossdomain_scan"}
        report_tool_progress("OpenAlex 元数据检索 1/2", 0.5)
        yield {"type": "tool_result", "name": "disc_crossdomain_scan"}
        yield {"type": "text", "text": "done"}
        yield {"type": "done", "steps": 1}


progress_updates = []
real_task_set = jobs.db.task_set


def capture_task_set(task_id, **kwargs):
    progress_updates.append(kwargs)
    return real_task_set(task_id, **kwargs)


with patch.object(jobs, "_agent_factory", return_value=ProgressAgent()), \
        patch.object(jobs.db, "task_set", side_effect=capture_task_set):
    progress_job = jobs.submit("gewu", f"progress-{time.time_ns()}", "progress")
    deadline = time.time() + 3
    while time.time() < deadline:
        progress_state = jobs.get(progress_job["job"]["id"])
        if progress_state["state"] == "done":
            break
        time.sleep(0.02)
assert progress_state["state"] == "done"
assert any("OpenAlex 元数据检索" in str(u.get("message", ""))
           for u in progress_updates)
db.ex("DELETE FROM tasks WHERE kind='agent_run' AND ref=?",
      (progress_job["job"]["id"],))


# Identical novelty checks must reuse the cached V4-Pro judgement.
novelty_calls = []


class FakeNoveltyLLM:
    def __init__(self, *args, **kwargs):
        pass

    def ask_json(self, *args, **kwargs):
        novelty_calls.append(kwargs)
        return {"verdict": "cross_domain_new", "closest_prior": [],
                "what_is_actually_new": "test", "confidence": 0.8,
                "reasoning": "test"}


statement = f"novelty-cache-test-{time.time_ns()}"
cache_key = lit_tools._novelty_cache_key(statement, True)
with patch.object(lit_tools.index, "hybrid_search", return_value=[]), \
        patch.object(lit_tools.search, "expand_queries", return_value={"en": ["q"]}), \
        patch.object(lit_tools.search, "s_openalex", return_value=[]), \
        patch.object(lit_tools, "LLM", FakeNoveltyLLM):
    first = lit_tools.lit_novelty_check(statement, search_web=True)
    second = lit_tools.lit_novelty_check(statement, search_web=True)
assert first["completed"] is True and first["cached"] is False
assert second["completed"] is True and second["cached"] is True
assert len(novelty_calls) == 1
assert novelty_calls[0]["thinking"] is False and novelty_calls[0]["attempts"] == 1
db.ex("DELETE FROM kv WHERE k=?", (cache_key,))


# search_web is metadata-only.  Full expansion is queued only via the separate
# explicit switch and never blocks the scan tool.
scan_llm_calls = []


class FakeScanLLM:
    def __init__(self, *args, **kwargs):
        pass

    def ask_json(self, *args, **kwargs):
        scan_llm_calls.append(kwargs)
        return {"donor_concept": "test concept", "mapping": "test mapping",
                "falsifiable_prediction": "test prediction",
                "computable_descriptor": {"name": "test_descriptor"},
                "discriminating_test": "test", "why_not_already_known": "test"}


scan_domain = f"test-domain-{time.time_ns()}"
with patch("zhizhi.core.llm.LLM", FakeScanLLM), \
        patch.object(lit_tools, "lit_novelty_check",
                     return_value={"verdict": "novel", "completed": True}), \
        patch.object(lit_tools, "queue_literature_expansion",
                     return_value={"queued": True, "task_ref": "LX-test"}) as queue_mock:
    fast = disc_tools.disc_crossdomain_scan(
        domain=scan_domain, mode="manual", context="test context",
        auto_novelty_check=True, search_web=True, expand_literature=False)
    assert fast["novelty_check_completed"] is True
    assert fast["status"] == "success" and fast["mode"] == "manual"
    assert scan_llm_calls[-1]["request_timeout"] == 1000
    assert "literature_expansion_task" not in fast
    queue_mock.assert_not_called()
    queued = disc_tools.disc_crossdomain_scan(
        domain=scan_domain, mode="manual", context="test context",
        auto_novelty_check=True, search_web=True, expand_literature=True)
    assert queued["literature_expansion_task"]["queued"] is True
    queue_mock.assert_called_once()
db.ex("DELETE FROM memory WHERE agent='gewu' AND kind='crossdomain_scanned' AND content=?",
      (scan_domain,))


class TimeoutScanLLM:
    def __init__(self, *args, **kwargs):
        pass

    def ask_json(self, *args, **kwargs):
        raise TimeoutError("Request timed out")


timeout_domain = f"timeout-domain-{time.time_ns()}"
with patch("zhizhi.core.llm.LLM", TimeoutScanLLM):
    timed_out = disc_tools.disc_crossdomain_scan(
        domain=timeout_domain, mode="manual", context="test",
        auto_novelty_check=False)
assert timed_out["status"] == "timeout"
assert timed_out["stage"] == "crossdomain_reasoning"
assert timed_out["domain"] == timeout_domain and timed_out["mode"] == "manual"
assert timed_out["missing_fields"] == [] and timed_out["scan_valid"] is False
assert not db.q1("SELECT 1 FROM memory WHERE kind='crossdomain_scanned' AND content=?",
                 (timeout_domain,))


class InvalidScanLLM:
    def __init__(self, *args, **kwargs):
        pass

    def ask_json(self, *args, **kwargs):
        return {"mapping": "only one field"}


invalid_domain = f"invalid-domain-{time.time_ns()}"
with patch("zhizhi.core.llm.LLM", InvalidScanLLM):
    invalid = disc_tools.disc_crossdomain_scan(
        domain=invalid_domain, mode="manual", context="test",
        auto_novelty_check=False)
assert invalid["status"] == "invalid" and invalid["stage"] == "schema_validation"
assert "donor_concept" in invalid["missing_fields"]
assert "mapping" not in invalid["missing_fields"]
assert invalid["mode"] == "manual"
assert not db.q1("SELECT 1 FROM memory WHERE kind='crossdomain_scanned' AND content=?",
                 (invalid_domain,))


# Supplying the recommendation from step 1 must bypass autonomous proposal and
# rerun only the expensive deep-reasoning phase.
reused_domain = f"reused-domain-{time.time_ns()}"
with patch("zhizhi.core.llm.LLM", FakeScanLLM), \
        patch.object(disc_tools, "disc_propose_domains") as propose_mock:
    reused = disc_tools.disc_crossdomain_scan(
        domain=reused_domain, mode="auto_propose", context="test",
        auto_novelty_check=False)
propose_mock.assert_not_called()
assert reused["status"] == "success" and reused["domain"] == reused_domain
db.ex("DELETE FROM memory WHERE agent='gewu' AND kind='crossdomain_scanned' AND content=?",
      (reused_domain,))


with patch.object(lit_tools._EXPAND_EXECUTOR, "submit") as submit:
    background = lit_tools.lit_expand_search(
        f"background-test-{time.time_ns()}", max_papers=10, background=True)
assert background["queued"] is True and background["task_ref"].startswith("LX")
submit.assert_called_once()
db.ex("DELETE FROM tasks WHERE id=?", (background["task_id"],))
db.ex("DELETE FROM kv WHERE k=?", (lit_tools._expansion_key(background["task_ref"]),))


# Recurring expansion removes already learned papers and duplicate search hits
# before the paid V4 relevance screen.
existing = db.q1("SELECT doi,title FROM papers WHERE LENGTH(title)>30 LIMIT 1")
assert existing
new_title = f"A completely new scheduler test paper {time.time_ns()}"
candidate = {"doi": f"10.9999/scheduler-{time.time_ns()}", "title": new_title,
             "abstract": "new evidence", "year": 2026, "journal": "Test"}
existing_work = {"doi": existing["doi"] or "", "title": existing["title"],
                 "abstract": "already learned", "year": 2020, "journal": "Test"}
screened = []


def fake_score(works, topic):
    screened.extend(works)
    return [{**work, "relevance": 9.0, "evidence_type": "supports"}
            for work in works]


with patch.object(lit_tools.search, "expand_queries",
                  return_value={"en": ["scheduler query"]}), \
        patch.object(lit_tools.search, "search_many",
                     return_value=[existing_work, candidate, dict(candidate)]), \
        patch.object(lit_tools.search, "score_relevance", side_effect=fake_score), \
        patch.object(lit_tools, "_novelty_scores", side_effect=lambda works: [1.0] * len(works)), \
        patch.object(lit_tools.search, "enqueue",
                     return_value={"added": 1, "skipped_duplicate": 0,
                                   "fulltext_obtained": 1, "abstract_only": 0,
                                   "added_ids": ["scheduler-test-paper"],
                                   "duplicate_ids": []}), \
        patch.object(lit_tools.worker, "control"):
    expansion = lit_tools._lit_expand_search_sync("scheduler test", max_papers=10)
assert len(screened) == 1 and screened[0]["title"] == new_title
assert expansion["n_preexisting"] == 1
assert expansion["n_repeated_candidates"] == 1
assert expansion["enqueue"]["added_ids"] == ["scheduler-test-paper"]


# Schedule configuration is persistent and independently controllable.  The
# unit test suppresses the daemon so it cannot make external calls.
schedule_topic = f"scheduled-learning-{time.time_ns()}"
with patch.object(lit_tools, "ensure_literature_scheduler"), \
        patch.object(lit_tools._SCHEDULE_WAKE, "set"):
    scheduled = lit_tools.lit_schedule_create(
        schedule_topic, interval_minutes=60, papers_per_run=7)
    assert scheduled["created"] is True
    schedule_task = db.q1("SELECT id FROM tasks WHERE kind='lit_schedule' AND ref=?",
                          (scheduled["ref"],))
    fake_round = {
        "n_found": 12, "n_new_candidates": 4, "n_preexisting": 3,
        "n_repeated_candidates": 2, "n_accepted": 1,
        "enqueue": {"added": 1, "skipped_duplicate": 0,
                    "fulltext_obtained": 1, "abstract_only": 0,
                    "added_ids": [paper["id"]], "duplicate_ids": []},
    }
    with patch.object(lit_tools, "_lit_expand_search_sync", return_value=fake_round):
        lit_tools._run_schedule_once(int(schedule_task["id"]), scheduled["ref"])
    listed = lit_tools.lit_schedule_status()["tasks"]
    row = next(item for item in listed if item["ref"] == scheduled["ref"])
    assert row["interval_minutes"] == 60 and row["papers_per_run"] == 7
    assert row["runs_completed"] == 1 and row["cumulative_added"] == 1
    assert row["cumulative_duplicates"] == 5
    Path(row["last_result"]["result_path"]).unlink(missing_ok=True)
    assert lit_tools.lit_schedule_control(scheduled["ref"], "pause")["state"] == "paused"
    assert lit_tools.lit_schedule_control(scheduled["ref"], "resume")["state"] == "running"
    assert lit_tools.lit_schedule_control(scheduled["ref"], "delete")["deleted"] is True
assert lit_tools._schedule_config(scheduled["ref"]) is None
assert db.q1("SELECT state FROM tasks WHERE kind='lit_schedule' AND ref=?",
             (scheduled["ref"],))["state"] == "deleted"
db.ex("DELETE FROM tasks WHERE kind='lit_schedule' AND ref=?", (scheduled["ref"],))

# Pause/delete must also cancel an already-running round at its next safe
# progress boundary.  Previously a later progress update overwrote the paused
# task state and the expensive round continued to completion.
for control_action in ("pause", "delete"):
    control_topic = f"schedule-{control_action}-{time.time_ns()}"
    with patch.object(lit_tools, "ensure_literature_scheduler"), \
            patch.object(lit_tools._SCHEDULE_WAKE, "set"):
        controlled = lit_tools.lit_schedule_create(
            control_topic, interval_minutes=60, papers_per_run=3)
        controlled_task = db.q1(
            "SELECT id FROM tasks WHERE kind='lit_schedule' AND ref=?",
            (controlled["ref"],))

        def interrupting_round(*_args, progress=None, **_kwargs):
            progress(0.10, "first safe boundary")
            result = lit_tools.lit_schedule_control(controlled["ref"], control_action)
            assert not result.get("error")
            progress(0.20, "must be interrupted here")
            raise AssertionError("a controlled schedule continued after cancellation")

        lit_tools._SCHEDULE_RUNNING.add(controlled["ref"])
        with patch.object(lit_tools, "_lit_expand_search_sync",
                          side_effect=interrupting_round):
            lit_tools._run_schedule_once(
                int(controlled_task["id"]), controlled["ref"], 0)

        task_after = db.q1("SELECT state,message FROM tasks WHERE id=?",
                           (int(controlled_task["id"]),))
        if control_action == "pause":
            config_after = lit_tools._schedule_config(controlled["ref"])
            assert config_after["state"] == "paused"
            assert config_after["run_active"] is False
            assert config_after["runs_completed"] == 0
            assert task_after["state"] == "paused"
            resumed = lit_tools.lit_schedule_control(controlled["ref"], "resume")
            assert resumed["state"] == "running"
            lit_tools.lit_schedule_control(controlled["ref"], "delete")
        else:
            assert lit_tools._schedule_config(controlled["ref"]) is None
            assert task_after["state"] == "deleted"
        db.ex("DELETE FROM tasks WHERE id=?", (int(controlled_task["id"]),))

print("targeted tests: OK")
