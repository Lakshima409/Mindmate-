import sqlite3
import uuid
from datetime import datetime, timedelta


def _column_exists(conn, table_name, column_name):
    """Check if a column exists in a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def _ensure_column(conn, table_name, column_name, column_def):
    """Add a column to a table if it doesn't already exist."""
    if not _column_exists(conn, table_name, column_name):
        cursor = conn.cursor()
        cursor.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}"
        )


def create_connection(db_path="chatbot.db"):
    """Create a database connection with best practices.

    Uses WAL mode for better concurrency and foreign keys for data integrity.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_tables(conn=None):
    """Create all necessary tables if they don't exist.

    Creates three tables:
    - training_data: stores minimal interaction data with session tracking
    - sessions: tracks session metadata for context management
    - feedback: stores user feedback on responses

    Args:
        conn: A sqlite3 connection object, or a path string to create one.
    """
    if conn is None:
        conn = create_connection()
        should_close = True
    elif isinstance(conn, str):
        conn = create_connection(conn)
        should_close = True
    else:
        should_close = False

    try:
        cursor = conn.cursor()

        # Training data table - stores minimal info for analysis with session tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_input TEXT,
                predicted_tag TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Session metadata table - tracks active sessions for context management
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_preferences TEXT DEFAULT '{}'
            )
        """)

        # Feedback table - stores user feedback on responses
        cursor.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            response_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            feedback INTEGER NOT NULL CHECK(feedback IN (1, -1)),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(response_id, session_id)
        )""")

        # Migrate existing databases: add missing columns
        _ensure_column(conn, "training_data", "session_id", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(conn, "sessions", "last_activity",
                       "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        _ensure_column(conn, "sessions", "user_preferences",
                       "TEXT DEFAULT '{}'")

        # Response ownership table - links server-generated response IDs to sessions
        cursor.execute("""CREATE TABLE IF NOT EXISTS response_ownership (
            response_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )""")

        conn.commit()
    finally:
        if should_close:
            conn.close()


def record_feedback(conn, response_id, session_id, feedback_val):
    """Record user feedback on a chatbot response.

    Returns True if feedback was recorded, False if a duplicate submission
    for the same (response_id, session_id) already exists.
    """
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO feedback (response_id, session_id, feedback)
               VALUES (?, ?, ?)""",
            (response_id, session_id, feedback_val),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # UNIQUE constraint on (response_id, session_id) - duplicate submission
        conn.rollback()
        return False


def clear_user_feedback(conn, session_id):
    """Clear all feedback associated with a given session."""
    cursor = conn.cursor()
    cursor.execute(
        """DELETE FROM feedback WHERE session_id = ?""",
        (session_id,),
    )
    conn.commit()


def save_interaction(conn, session_id, user_input, predicted_tag=None, confidence=None):
    """Save an interaction with proper parameterized queries and session tracking.

    Uses parameterized SQL queries to prevent SQL injection.
    Truncates user_input to 500 chars to protect privacy.
    Updates session last_activity timestamp.
    """
    cursor = conn.cursor()

    # Truncate long inputs to protect privacy
    safe_input = user_input[:500] if user_input else None

    cursor.execute(
        """INSERT INTO training_data
           (session_id, user_input, predicted_tag, confidence)
           VALUES (?, ?, ?, ?)""",
        (session_id, safe_input, predicted_tag, confidence),
    )

    # Update session last_activity
    cursor.execute(
        """INSERT INTO sessions (session_id, last_activity)
           VALUES (?, CURRENT_TIMESTAMP)
           ON CONFLICT(session_id) DO UPDATE SET last_activity = CURRENT_TIMESTAMP""",
        (session_id,),
    )

    conn.commit()


def get_recent_interactions(conn, session_id, limit=10):
    """Get recent interactions for a specific session only.

    Prevents one conversation's context from leaking into another.
    Only returns data for the specified session_id.
    """
    cursor = conn.cursor()
    cursor.execute(
        """SELECT user_input, predicted_tag, confidence, created_at
           FROM training_data
           WHERE session_id = ?
           ORDER BY created_at DESC
           LIMIT ?""",
        (session_id, limit),
    )
    return cursor.fetchall()


def cleanup_old_interactions(conn, days=90):
    """Remove interactions older than specified days for privacy retention.

    Default: 90 days. Can be configured based on privacy requirements.
    Only deletes from training_data, not from sessions metadata.
    """
    cursor = conn.cursor()
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """DELETE FROM training_data WHERE created_at < ?""",
        (cutoff_date,),
    )
    conn.commit()


def register_response(conn, response_id, session_id):
    """Persist server-side ownership of a generated response ID."""
    cursor = conn.cursor()
    cursor.execute(
        """INSERT OR IGNORE INTO response_ownership (response_id, session_id)
           VALUES (?, ?)""",
        (response_id, session_id),
    )
    conn.commit()
    return True


def get_response_owner(conn, response_id):
    """Return the owning session ID for a response, or None if unknown."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT session_id FROM response_ownership WHERE response_id = ?",
        (response_id,),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def cleanup_old_response_ownership(conn, days=30):
    """Remove response-ownership records older than the feedback window.

    This is a manual/administrative helper only. It is not scheduled
    automatically. Ownership records should be retained at least as long
    as users are expected to submit feedback; deleting them only prevents
    future feedback for those old responses, not existing feedback rows.
    """
    cursor = conn.cursor()
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    cursor.execute(
        "DELETE FROM response_ownership WHERE created_at < ?", (cutoff_date,)
    )
    conn.commit()
    return cursor.rowcount


def generate_session_id():
    """Generate a random, non-guessable session identifier.

    Uses UUID4 for cryptographically random session IDs.
    Prevents session ID guessing attacks.
    """
    return str(uuid.uuid4())
