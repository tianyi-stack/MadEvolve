"""
Database schema for program storage.
"""

import sqlite3
from typing import Optional

SCHEMA_VERSION = 1

PROGRAMS_TABLE = """
CREATE TABLE IF NOT EXISTS programs (
    program_id TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    parent_id TEXT,
    generation INTEGER NOT NULL,
    combined_score REAL,
    public_metrics TEXT,
    private_metrics TEXT,
    text_feedback TEXT,
    embedding BLOB,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_id) REFERENCES programs(program_id)
)
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_programs_generation ON programs(generation)",
    "CREATE INDEX IF NOT EXISTS idx_programs_score ON programs(combined_score DESC)",
    "CREATE INDEX IF NOT EXISTS idx_programs_parent ON programs(parent_id)",
]

METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT
)
"""


def create_schema(conn: sqlite3.Connection):
    """
    Create or upgrade database schema.

    Args:
        conn: SQLite connection
    """
    cursor = conn.cursor()

    # Create tables
    cursor.execute(PROGRAMS_TABLE)
    cursor.execute(METADATA_TABLE)

    # Create indexes
    for idx in INDEXES:
        cursor.execute(idx)

    # Set schema version
    cursor.execute(
        "INSERT OR REPLACE INTO schema_metadata (key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )

    conn.commit()


def get_schema_version(conn: sqlite3.Connection) -> Optional[int]:
    """Get current schema version."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        )
        row = cursor.fetchone()
        return int(row[0]) if row else None
    except sqlite3.OperationalError:
        return None
