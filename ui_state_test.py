from __future__ import annotations

import time
import tempfile
from pathlib import Path
from unittest.mock import patch

from zhizhi.core import db
from zhizhi.core.agent import (active_session, delete_session, list_sessions,
                               new_session, save_message, set_active_session)
from zhizhi.lit import worker
from zhizhi.tools import meta_tools


db.init()
agent = f"state-test-{time.time_ns()}"
s1 = new_session(agent, "one")
s2 = new_session(agent, "two")
assert active_session(agent) == s2
assert set_active_session(agent, s1) == s1
assert active_session(agent) == s1
internal = new_session(agent, "[咨询] internal", make_active=False)
assert active_session(agent) == s1
assert internal not in {s["id"] for s in list_sessions(agent)}

deletable = new_session(agent, "delete-me")
save_message(deletable, "user", "this conversation must be deleted")
replacement = delete_session(agent, deletable)
assert replacement != deletable and active_session(agent) == replacement
assert not db.q1("SELECT 1 FROM sessions WHERE id=?", (deletable,))
assert not db.q1("SELECT 1 FROM messages WHERE session_id=?", (deletable,))

status = worker.status()
assert {"activity", "active_papers", "idle_threads", "running_now"} <= status.keys()
assert status["active_papers"] == len(status["running_now"])

depth_token = meta_tools._CONSULT_DEPTH.set(1)
try:
    blocked = meta_tools.agent_consult("gewu", "should not recurse")
finally:
    meta_tools._CONSULT_DEPTH.reset(depth_token)
assert blocked.get("blocked") == "recursive_agent_consult"

previous_worker_state = db.kv_get(worker.CTL, "paused")
with tempfile.TemporaryDirectory() as td:
    sample = Path(td) / "batch-upload-test.pdf"
    sample.write_bytes(b"%PDF-1.4\n% queue-only test " + str(time.time_ns()).encode())
    with patch.object(worker, "ensure_thread") as ensure:
        uploaded = worker.register_new_pdf(sample, title=f"queue-test-{time.time_ns()}")
    assert "learning" in uploaded and "ingest" not in uploaded
    assert uploaded["learning"]["state"] == "queued"
    ensure.assert_called_once()
    task = db.q1("SELECT state FROM tasks WHERE id=?", (uploaded["learning"]["task_id"],))
    assert task and task["state"] == "queued"
    Path(uploaded["saved_to"]).unlink(missing_ok=True)
    db.ex("DELETE FROM tasks WHERE id=?", (uploaded["learning"]["task_id"],))
    db.ex("DELETE FROM papers WHERE id=?", (uploaded["paper_id"],))

    paper_id = f"attach-test-{time.time_ns()}"
    supplement = Path(td) / "supplement-test.pdf"
    supplement.write_bytes(b"%PDF-1.4\n% attach test " + str(time.time_ns()).encode())
    db.ex("INSERT INTO papers(id,source,title,abstract,evidence_level,status,added_at) "
          "VALUES(?,'test',?,'abstract','abstract','failed',?)",
          (paper_id, f"attach-title-{time.time_ns()}", time.time()))
    with patch.object(worker, "ensure_thread") as ensure:
        attached = worker.attach_fulltext(paper_id, supplement)
    assert attached["paper_id"] == paper_id
    assert attached["learning"]["state"] == "queued"
    ensure.assert_called_once()
    stored = db.q1("SELECT path,content_hash,status,evidence_level FROM papers WHERE id=?",
                   (paper_id,))
    assert stored["path"] == attached["saved_to"]
    assert stored["content_hash"] and stored["status"] == "queued"
    assert stored["evidence_level"] == "fulltext"
    Path(attached["saved_to"]).unlink(missing_ok=True)
    db.ex("DELETE FROM tasks WHERE id=?", (attached["learning"]["task_id"],))
    db.ex("DELETE FROM papers WHERE id=?", (paper_id,))
db.kv_set(worker.CTL, previous_worker_state)

db.ex("DELETE FROM messages WHERE session_id IN (?,?,?)", (s1, s2, internal))
db.ex("DELETE FROM sessions WHERE id IN (?,?,?)", (s1, s2, internal))
db.ex("DELETE FROM kv WHERE k=?", (f"agent_active_session:{agent}",))
print("agent UI state tests: OK")
