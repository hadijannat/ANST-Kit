import json
from argparse import Namespace

from anstkit.__main__ import cmd_export_audit
from anstkit.audit.events import AuditEvent, EventType
from anstkit.audit.store import AuditStore


def test_export_audit_command(tmp_path):
    db_path = tmp_path / "audit.db"
    store = AuditStore(str(db_path))

    session_id = "sess-123"
    store.append(
        AuditEvent(
            event_type=EventType.POLICY_GATE_PASS,
            session_id=session_id,
            payload={"evidence": [{"ok": True}]},
        )
    )
    store.append(
        AuditEvent(
            event_type=EventType.DECISION_MADE,
            session_id=session_id,
            payload={"approved": True},
        )
    )

    out_path = tmp_path / "export.json"
    args = Namespace(
        session_id=session_id,
        db_path=str(db_path),
        limit=1000,
        out=str(out_path),
        events_only=False,
        evidence_only=False,
    )

    cmd_export_audit(args)

    data = json.loads(out_path.read_text())
    assert data["session_id"] == session_id
    assert len(data["events"]) == 2
    assert data["evidence"]["policy"] == [{"ok": True}]
