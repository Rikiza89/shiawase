"""
ChromaDB によるベクトル記憶管理モジュール。
会話の意味的検索と永続化を担当する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import chromadb
from chromadb.config import Settings

from ai.embeddings import EmbeddingService, EmbeddingError
from config import MemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """検索結果の記憶エントリ。"""
    document: str
    distance: float
    metadata: dict


class MemoryError(Exception):
    """記憶操作の失敗を表す例外。"""


class MemoryService:
    """
    ChromaDB + Ollama 埋め込みによる長期記憶管理サービス。
    再起動を跨いで会話コンテキストを保持する。

    Args:
        cfg: メモリ設定。
        embedding_svc: 埋め込み生成サービス。
    """

    def __init__(self, cfg: MemoryConfig, embedding_svc: EmbeddingService) -> None:
        self._cfg = cfg
        self._embedder = embedding_svc
        self._client = chromadb.PersistentClient(
            path=str(cfg.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB コレクション準備完了: %s (件数=%d)",
            cfg.collection_name,
            self._collection.count(),
        )

    def store(self, text: str, role: str, turn_id: str) -> None:
        """
        会話ターンをベクトル化して記憶に保存する。

        Args:
            text: 保存するテキスト。
            role: "user" または "assistant"。
            turn_id: ターンの一意 ID（重複防止）。

        Raises:
            MemoryError: 保存失敗時。
        """
        try:
            embedding = self._embedder.embed(text)
            self._collection.upsert(
                ids=[turn_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{"role": role}],
            )
            logger.debug("記憶保存: turn_id=%s, role=%s", turn_id, role)
        except EmbeddingError as e:
            raise MemoryError(f"埋め込み生成失敗による記憶保存エラー: {e}") from e
        except Exception as e:
            raise MemoryError(f"ChromaDB 保存エラー: {e}") from e

    def retrieve(self, query: str, top_k: int | None = None) -> List[MemoryEntry]:
        """
        クエリに意味的に近い記憶を取得する。

        Args:
            query: 検索クエリテキスト。
            top_k: 取得件数上限（未指定は config 値使用）。

        Returns:
            距離順の MemoryEntry リスト。

        Raises:
            MemoryError: 検索失敗時。
        """
        k = top_k or self._cfg.top_k_memories
        count = self._collection.count()
        if count == 0:
            return []

        # 実際の件数を超えないよう調整
        actual_k = min(k, count)

        try:
            embedding = self._embedder.embed(query)
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=actual_k,
                include=["documents", "distances", "metadatas"],
            )
        except EmbeddingError as e:
            raise MemoryError(f"検索クエリの埋め込み生成失敗: {e}") from e
        except Exception as e:
            raise MemoryError(f"ChromaDB 検索エラー: {e}") from e

        entries: List[MemoryEntry] = []
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        for doc, dist, meta in zip(docs, dists, metas):
            entries.append(MemoryEntry(document=doc, distance=dist, metadata=meta))

        logger.debug("記憶検索完了: query_len=%d, hits=%d", len(query), len(entries))
        return entries