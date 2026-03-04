"""
SQLite によるチャット履歴永続化モジュール。
スキーマ管理・CRUD を集約し、ビジネスロジックは持たない。
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, List

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """チャット履歴の1レコードを表すドメインモデル。"""
    role: str          # "user" | "assistant"
    content: str
    timestamp: str     # ISO 8601
    id: int | None = None


class StorageError(Exception):
    """ストレージ操作の失敗を表す例外。"""


class ChatStorage:
    """
    SQLite ベースのチャット履歴ストレージ。

    Args:
        db_path: SQLite データベースファイルパス。
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS chat_history (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        role      TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
        content   TEXT    NOT NULL,
        timestamp TEXT    NOT NULL
    );
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._initialize()

    def _initialize(self) -> None:
        """テーブルが存在しなければ作成する。"""
        try:
            with self._connect() as conn:
                conn.execute(self._CREATE_TABLE)
        except sqlite3.Error as e:
            raise StorageError(f"DB 初期化失敗: {e}") from e

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """
        スレッドセーフな DB 接続コンテキストマネージャ。
        check_same_thread=False でマルチスレッド対応。
        """
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=10,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise StorageError(f"DB 操作失敗: {e}") from e
        finally:
            conn.close()

    def save_message(self, role: str, content: str) -> None:
        """
        メッセージを履歴に保存する。

        Args:
            role: "user" または "assistant"。
            content: メッセージ本文。

        Raises:
            StorageError: 保存失敗時。
        """
        ts = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_history (role, content, timestamp) VALUES (?, ?, ?)",
                (role, content, ts),
            )
        logger.debug("メッセージ保存: role=%s, len=%d", role, len(content))

    def get_recent_messages(self, limit: int = 20) -> List[ChatMessage]:
        """
        最近のメッセージを取得する。

        Args:
            limit: 取得件数上限。

        Returns:
            古い順に並んだ ChatMessage リスト。
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, timestamp
                FROM chat_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        # 古い順（昇順）に変換して返す
        return [
            ChatMessage(
                id=r["id"],
                role=r["role"],
                content=r["content"],
                timestamp=r["timestamp"],
            )
            for r in reversed(rows)
        ]

    def get_all_messages(self) -> List[ChatMessage]:
        """
        全チャット履歴を取得する（UIの初期表示用）。

        Returns:
            古い順の ChatMessage リスト。
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, role, content, timestamp FROM chat_history ORDER BY id ASC"
            ).fetchall()
        return [
            ChatMessage(id=r["id"], role=r["role"], content=r["content"], timestamp=r["timestamp"])
            for r in rows
        ]

    def clear_all(self) -> None:
        """
        全履歴を削除する（デバッグ・リセット用）。
        本番では慎重に使用すること。
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM chat_history")
        logger.warning("チャット履歴を全削除しました")