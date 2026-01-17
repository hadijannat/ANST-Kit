"""Append-only audit event store with SQLite backend.

This module provides thread-safe audit logging using SQLite. Thread safety is
achieved through thread-local connections, ensuring each thread gets its own
database connection.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from typing import List, Optional

from .events import AuditEvent, EventType

logger = logging.getLogger(__name__)


class AuditStore:
    """Append-only audit event store with SQLite backend.

    This store provides non-repudiation guarantees by:
    1. Only allowing append operations (no updates or deletes)
    2. Storing all event metadata including timestamps and UUIDs
    3. Maintaining causality chains via parent_event_id

    Thread Safety:
        Uses thread-local connections to ensure each thread gets its own
        SQLite connection, preventing "SQLite objects created in a thread
        can only be used in that same thread" errors.

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

        # Thread-local storage for connections
        self._local = threading.local()

        # For in-memory databases, we need a shared connection
        # to preserve data across calls
        self._memory_conn: Optional[sqlite3.Connection] = None
        self._memory_lock = threading.Lock()

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection for the current thread.

        For in-memory databases, returns a shared connection (protected by lock).
        For file databases, returns a thread-local connection.
        """
        if self._is_memory:
            # In-memory databases need a single shared connection
            # to preserve data, protected by a lock
            if self._memory_conn is None:
                with self._memory_lock:
                    if self._memory_conn is None:
                        self._memory_conn = sqlite3.connect(
                            ":memory:", check_same_thread=False
                        )
            return self._memory_conn

        # File-based database: use thread-local connection
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            logger.debug(f"Created new SQLite connection for thread {threading.current_thread().name}")
        return self._local.conn

    def _close_connection(self, conn: sqlite3.Connection) -> None:
        """Close a connection if appropriate.

        For file databases, we keep thread-local connections open for reuse.
        For in-memory databases, we never close the shared connection until close() is called.
        """
        # Don't close - let connections be reused within their thread
        pass

    def _init_db(self) -> None:
        """Create the events table if it doesn't exist."""
        conn = self._get_connection()

        if self._is_memory:
            with self._memory_lock:
                self._create_schema(conn)
        else:
            self._create_schema(conn)

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create the database schema."""
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

    def append(self, event: AuditEvent) -> None:
        """Append an event to the store.

        Args:
            event: The audit event to store.
        """
        conn = self._get_connection()

        if self._is_memory:
            with self._memory_lock:
                self._insert_event(conn, event)
        else:
            self._insert_event(conn, event)

    def _insert_event(self, conn: sqlite3.Connection, event: AuditEvent) -> None:
        """Insert an event into the database."""
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

        if self._is_memory:
            with self._memory_lock:
                return self._execute_query(conn, session_id, event_type, limit)
        else:
            return self._execute_query(conn, session_id, event_type, limit)

    def _execute_query(
        self,
        conn: sqlite3.Connection,
        session_id: Optional[str],
        event_type: Optional[EventType],
        limit: int,
    ) -> List[AuditEvent]:
        """Execute the query and return results."""
        query = "SELECT * FROM events WHERE 1=1"
        params: List[str] = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)

        # Use parameterized limit for safety
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(str(limit))

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
        """Close all database connections.

        For in-memory databases, closes the shared connection (data is lost).
        For file databases, closes all thread-local connections.
        """
        if self._is_memory:
            with self._memory_lock:
                if self._memory_conn is not None:
                    self._memory_conn.close()
                    self._memory_conn = None
                    logger.debug("Closed in-memory audit store connection")
        else:
            # Close thread-local connection if it exists
            if hasattr(self._local, "conn") and self._local.conn is not None:
                self._local.conn.close()
                self._local.conn = None
                logger.debug("Closed thread-local audit store connection")
