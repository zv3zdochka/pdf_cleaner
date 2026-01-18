from __future__ import annotations

import sqlite3
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class UserDatabase:
    SCHEMA_VERSION = 2

    def __init__(self, db_path: Path, logger=None):
        self.db_path = Path(db_path)
        self.log = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_existing_columns(self, conn: sqlite3.Connection, table_name: str) -> Set[str]:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return {row[1] for row in cursor.fetchall()}

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            return

        existing_columns = self._get_existing_columns(conn, "users")

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

        for col_name, col_def in required_columns.items():
            if col_name not in existing_columns:
                if "DEFAULT" in col_def:
                    parts = col_def.split("DEFAULT")
                    col_type = parts[0].strip()
                    default_val = parts[1].strip()
                    alter_sql = f"ALTER TABLE users ADD COLUMN {col_name} {col_type} DEFAULT {default_val}"
                else:
                    col_type = col_def.replace("PRIMARY KEY", "").strip()
                    alter_sql = f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"

                try:
                    cursor.execute(alter_sql)
                except sqlite3.OperationalError:
                    pass

        conn.commit()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

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
        self._migrate_schema(conn)
        conn.close()

    def has_phone_number(self, user_id: int) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT phone_number FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            result = bool(row and row["phone_number"])
            self.log.info("has_phone_number(%s) = %s", user_id, result)
            return result
        finally:
            conn.close()

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
        now = datetime.utcnow().isoformat()
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            try:
                cursor.execute(
                    "SELECT user_id, pdfs_sent, messages_count, phone_number FROM users WHERE user_id = ?",
                    (user_id,)
                )
                row = cursor.fetchone()

                if row:
                    pdfs_sent = (row["pdfs_sent"] or 0) + (1 if is_pdf else 0)
                    messages_count = (row["messages_count"] or 0) + 1
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
                else:
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

                if chat_id is not None:
                    cursor.execute('''
                        INSERT INTO user_chats (user_id, chat_id, chat_type, first_seen_at, last_seen_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(user_id, chat_id) DO UPDATE SET
                            chat_type = COALESCE(excluded.chat_type, chat_type),
                            last_seen_at = excluded.last_seen_at
                    ''', (user_id, chat_id, chat_type, now, now))

                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def update_phone_number(self, user_id: int, phone_number: str) -> None:
        self.log.info("update_phone_number called: user_id=%s, phone=%s", user_id, phone_number)
        now = datetime.utcnow().isoformat()
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
                if cursor.fetchone():
                    cursor.execute('''
                        UPDATE users SET phone_number = ?, last_activity_at = ?
                        WHERE user_id = ?
                    ''', (phone_number, now, user_id))
                    self.log.info("Updated phone for existing user %s", user_id)
                else:
                    cursor.execute('''
                        INSERT INTO users (user_id, phone_number, first_seen_at, last_activity_at, pdfs_sent, messages_count)
                        VALUES (?, ?, ?, ?, 0, 1)
                    ''', (user_id, phone_number, now, now))
                    self.log.info("Created new user %s with phone", user_id)
                conn.commit()
            except Exception as e:
                self.log.error("Failed to update phone: %s", e)
                conn.rollback()
            finally:
                conn.close()

    def get_all_users(self) -> List[Dict[str, Any]]:
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