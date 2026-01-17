"""Append-only audit event store with SQLite backend."""

import json
import sqlite3
from datetime import datetime
from typing import List, Optional

from .events import AuditEvent, EventType


class AuditStore:
    """Append-only audit event store with SQLite backend.

    This store provides non-repudiation guarantees by:
    1. Only allowing append operations (no updates or deletes)
    2. Storing all event metadata including timestamps and UUIDs
    3. Maintaining causality chains via parent_event_id

    Example:
        >>> store = AuditStore(":memory:")
        >>> event = AuditEvent(
        ...     event_type=EventType.PROPOSAL_SUBMITTED,
        ...     session_id="sess-123",
        ...     payload={"goal": "stabilize level"},
        ... )
        >>> store.append(event)
        >>> events = store.query(session_id="sess-123")
    """

    def __init__(self, db_path: str = "audit.db"):
        """Initialize the audit store.

        Args:
            db_path: Path to SQLite database file. Use ":memory:" for in-memory.
        """
        self.db_path = db_path
        self._is_memory = db_path == ":memory:"
        # For in-memory databases, maintain a single persistent connection
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection.

        For in-memory databases, returns the persistent connection.
        For file databases, creates a new connection.
        Note: check_same_thread=False allows connection sharing across threads
        (needed for FastAPI's threadpool execution of sync endpoints).
        """
        if self._is_memory:
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            return self._conn
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _close_connection(self, conn: sqlite3.Connection) -> None:
        """Close a connection if it's not the persistent in-memory connection."""
        if not self._is_memory:
            conn.close()

    def _init_db(self) -> None:
        """Create the events table if it doesn't exist."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                parent_event_id TEXT,
                payload TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id ON events(session_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)
        """)
        conn.commit()
        self._close_connection(conn)

    def append(self, event: AuditEvent) -> None:
        """Append an event to the store.

        Args:
            event: The audit event to store.
        """
        conn = self._get_connection()
        conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.event_type.value,
                event.session_id,
                event.timestamp.isoformat(),
                event.parent_event_id,
                json.dumps(event.payload),
            ),
        )
        conn.commit()
        self._close_connection(conn)

    def query(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """Query events with optional filters.

        Args:
            session_id: Filter by session ID.
            event_type: Filter by event type.
            limit: Maximum number of events to return.

        Returns:
            List of matching AuditEvent objects, ordered by timestamp descending.
        """
        conn = self._get_connection()
        query = "SELECT * FROM events WHERE 1=1"
        params: List[str] = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        cursor = conn.execute(query, params)
        events = []
        for row in cursor.fetchall():
            events.append(
                AuditEvent(
                    event_id=row[0],
                    event_type=EventType(row[1]),
                    session_id=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    parent_event_id=row[4],
                    payload=json.loads(row[5]),
                )
            )
        self._close_connection(conn)
        return events

    def delete(self, event_id: str) -> None:
        """Attempt to delete an event (always raises).

        This method exists to enforce immutability - audit events
        cannot be deleted to maintain non-repudiation guarantees.

        Args:
            event_id: The event ID to (not) delete.

        Raises:
            ValueError: Always, as audit events are immutable.
        """
        raise ValueError("Audit events are immutable and cannot be deleted")

    def close(self) -> None:
        """Close the persistent connection for in-memory databases."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
