"""User database for storing Telegram user data and statistics.

This module provides a lightweight SQLite-based persistence layer for
tracking Telegram users who interact with the bot.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class UserDatabase:
    """SQLite database manager for user data and statistics."""

    # Current schema version
    SCHEMA_VERSION = 2

    def __init__(self, db_path: Path, logger=None):
        """Initialize the user database.

        Parameters
        ----------
        db_path:
            Path to the SQLite database file.
        logger:
            Optional logger instance.
        """
        self.db_path = Path(db_path)
        self.log = logger
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new database connection."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_existing_columns(self, conn: sqlite3.Connection, table_name: str) -> Set[str]:
        """Get set of existing column names for a table."""
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return {row[1] for row in cursor.fetchall()}

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Migrate database schema to current version."""
        cursor = conn.cursor()

        # Check if users table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            # No users table yet, nothing to migrate
            return

        existing_columns = self._get_existing_columns(conn, "users")

        # List of columns that should exist with their definitions
        required_columns = {
            "user_id": "INTEGER PRIMARY KEY",
            "username": "TEXT",
            "first_name": "TEXT",
            "last_name": "TEXT",
            "language_code": "TEXT",
            "is_bot": "INTEGER DEFAULT 0",
            "phone_number": "TEXT",
            "first_seen_at": "TEXT",
            "last_activity_at": "TEXT",
            "pdfs_sent": "INTEGER DEFAULT 0",
            "messages_count": "INTEGER DEFAULT 0",
        }

        # Add missing columns
        for col_name, col_def in required_columns.items():
            if col_name not in existing_columns:
                # Extract just the type and default for ALTER TABLE
                # SQLite ALTER TABLE only supports simple column additions
                if "DEFAULT" in col_def:
                    parts = col_def.split("DEFAULT")
                    col_type = parts[0].strip()
                    default_val = parts[1].strip()
                    alter_sql = f"ALTER TABLE users ADD COLUMN {col_name} {col_type} DEFAULT {default_val}"
                else:
                    col_type = col_def.replace("PRIMARY KEY", "").strip()
                    alter_sql = f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"

                if self.log:
                    self.log.info("Migrating: adding column %s to users table", col_name)

                try:
                    cursor.execute(alter_sql)
                except sqlite3.OperationalError as e:
                    if self.log:
                        self.log.warning("Could not add column %s: %s", col_name, e)

        conn.commit()

    def _init_db(self) -> None:
        """Create database tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if self.log:
            self.log.info("Creating/opening database at: %s", self.db_path)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                language_code TEXT,
                is_bot INTEGER DEFAULT 0,
                phone_number TEXT,
                first_seen_at TEXT NOT NULL DEFAULT '',
                last_activity_at TEXT NOT NULL DEFAULT '',
                pdfs_sent INTEGER DEFAULT 0,
                messages_count INTEGER DEFAULT 0
            )
        ''')

        # User chats table (tracks which chats user has interacted from)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                chat_id INTEGER NOT NULL,
                chat_type TEXT,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                UNIQUE(user_id, chat_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')

        conn.commit()

        # Run migrations to ensure all columns exist
        self._migrate_schema(conn)

        conn.close()

        if self.log:
            self.log.info("User database initialized successfully at %s", self.db_path)

    def record_user_activity(
        self,
        user_id: int,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        language_code: Optional[str] = None,
        is_bot: bool = False,
        chat_id: Optional[int] = None,
        chat_type: Optional[str] = None,
        is_pdf: bool = False,
        phone_number: Optional[str] = None,
    ) -> None:
        """Record or update user activity.

        Parameters
        ----------
        user_id:
            Telegram user ID.
        username:
            Telegram username (without @).
        first_name:
            User's first name.
        last_name:
            User's last name.
        language_code:
            User's language code (e.g., 'en', 'ru').
        is_bot:
            Whether the user is a bot.
        chat_id:
            Chat ID where interaction occurred.
        chat_type:
            Type of chat ('private', 'group', 'supergroup', 'channel').
        is_pdf:
            Whether this activity is a PDF upload.
        phone_number:
            Phone number if explicitly provided by user.
        """
        now = datetime.utcnow().isoformat()

        if self.log:
            self.log.info(
                "Recording activity for user_id=%s, username=%s, is_pdf=%s",
                user_id, username, is_pdf
            )

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            try:
                # Check if user exists
                cursor.execute(
                    "SELECT user_id, pdfs_sent, messages_count, phone_number FROM users WHERE user_id = ?",
                    (user_id,)
                )
                row = cursor.fetchone()

                if row:
                    # Update existing user
                    pdfs_sent = (row["pdfs_sent"] or 0) + (1 if is_pdf else 0)
                    messages_count = (row["messages_count"] or 0) + 1
                    # Preserve existing phone if new one not provided
                    existing_phone = row["phone_number"]
                    final_phone = phone_number if phone_number else existing_phone

                    cursor.execute('''
                        UPDATE users SET
                            username = COALESCE(?, username),
                            first_name = COALESCE(?, first_name),
                            last_name = COALESCE(?, last_name),
                            language_code = COALESCE(?, language_code),
                            is_bot = ?,
                            phone_number = COALESCE(?, phone_number),
                            last_activity_at = ?,
                            pdfs_sent = ?,
                            messages_count = ?
                        WHERE user_id = ?
                    ''', (
                        username, first_name, last_name, language_code,
                        1 if is_bot else 0, final_phone, now,
                        pdfs_sent, messages_count, user_id
                    ))

                    if self.log:
                        self.log.info(
                            "Updated user %s: pdfs_sent=%s, messages_count=%s",
                            user_id, pdfs_sent, messages_count
                        )
                else:
                    # Insert new user
                    cursor.execute('''
                        INSERT INTO users (
                            user_id, username, first_name, last_name, language_code,
                            is_bot, phone_number, first_seen_at, last_activity_at,
                            pdfs_sent, messages_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        user_id, username, first_name, last_name, language_code,
                        1 if is_bot else 0, phone_number, now, now,
                        1 if is_pdf else 0, 1
                    ))

                    if self.log:
                        self.log.info("Inserted new user %s", user_id)

                # Record chat if provided
                if chat_id is not None:
                    cursor.execute('''
                        INSERT INTO user_chats (user_id, chat_id, chat_type, first_seen_at, last_seen_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(user_id, chat_id) DO UPDATE SET
                            chat_type = COALESCE(excluded.chat_type, chat_type),
                            last_seen_at = excluded.last_seen_at
                    ''', (user_id, chat_id, chat_type, now, now))

                conn.commit()

                if self.log:
                    self.log.info("Successfully committed user activity for user_id=%s", user_id)

            except Exception as e:
                if self.log:
                    self.log.error("Failed to record user activity for user_id=%s: %s", user_id, e)
                conn.rollback()
                raise
            finally:
                conn.close()

    def update_phone_number(self, user_id: int, phone_number: str) -> None:
        """Update user's phone number (when explicitly shared via contact).

        Parameters
        ----------
        user_id:
            Telegram user ID.
        phone_number:
            Phone number string.
        """
        now = datetime.utcnow().isoformat()

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            try:
                cursor.execute('''
                    UPDATE users SET phone_number = ?, last_activity_at = ?
                    WHERE user_id = ?
                ''', (phone_number, now, user_id))
                conn.commit()
            except Exception as e:
                if self.log:
                    self.log.error("Failed to update phone for user_id=%s: %s", user_id, e)
                conn.rollback()
            finally:
                conn.close()

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Retrieve all users with their statistics.

        Returns
        -------
        List[Dict[str, Any]]
            List of user records with all fields.
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT
                    u.user_id,
                    u.username,
                    u.first_name,
                    u.last_name,
                    u.language_code,
                    u.is_bot,
                    u.phone_number,
                    u.first_seen_at,
                    u.last_activity_at,
                    u.pdfs_sent,
                    u.messages_count
                FROM users u
                ORDER BY u.last_activity_at DESC
            ''')

            users = []
            for row in cursor.fetchall():
                user = dict(row)
                user["is_bot"] = bool(user["is_bot"])

                # Get associated chats
                cursor.execute('''
                    SELECT chat_id, chat_type, first_seen_at, last_seen_at
                    FROM user_chats WHERE user_id = ?
                ''', (user["user_id"],))
                user["chats"] = [dict(c) for c in cursor.fetchall()]

                users.append(user)

            return users
        finally:
            conn.close()

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single user by ID.

        Parameters
        ----------
        user_id:
            Telegram user ID.

        Returns
        -------
        Optional[Dict[str, Any]]
            User record or None if not found.
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()

            if not row:
                return None

            user = dict(row)
            user["is_bot"] = bool(user["is_bot"])

            cursor.execute('''
                SELECT chat_id, chat_type, first_seen_at, last_seen_at
                FROM user_chats WHERE user_id = ?
            ''', (user_id,))
            user["chats"] = [dict(c) for c in cursor.fetchall()]

            return user
        finally:
            conn.close()

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get aggregate statistics.

        Returns
        -------
        Dict[str, Any]
            Summary statistics.
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT COUNT(*) as total_users FROM users")
            total_users = cursor.fetchone()["total_users"]

            cursor.execute("SELECT SUM(pdfs_sent) as total_pdfs FROM users")
            row = cursor.fetchone()
            total_pdfs = row["total_pdfs"] if row["total_pdfs"] else 0

            cursor.execute("SELECT SUM(messages_count) as total_messages FROM users")
            row = cursor.fetchone()
            total_messages = row["total_messages"] if row["total_messages"] else 0

            cursor.execute("SELECT COUNT(DISTINCT chat_id) as total_chats FROM user_chats")
            total_chats = cursor.fetchone()["total_chats"]

            return {
                "total_users": total_users,
                "total_pdfs": total_pdfs,
                "total_messages": total_messages,
                "total_chats": total_chats,
            }
        finally:
            conn.close()