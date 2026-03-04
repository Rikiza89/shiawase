"""
Ollama nomic-embed-text を使ったベクトル埋め込み生成モジュール。
"""

from __future__ import annotations

import logging
from typing import List

import requests

from config import OllamaConfig

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """埋め込み生成の失敗を表す例外。"""


class EmbeddingService:
    """
    Ollama HTTP API 経由でテキスト埋め込みを生成するサービス。

    Args:
        cfg: Ollama 接続設定。
    """

    def __init__(self, cfg: OllamaConfig) -> None:
        self._cfg = cfg
        self._endpoint = f"{cfg.base_url}/api/embeddings"

    def embed(self, text: str) -> List[float]:
        """
        テキストをベクトルに変換する。

        Args:
            text: 埋め込み対象テキスト。

        Returns:
            埋め込みベクトル（float のリスト）。

        Raises:
            EmbeddingError: API 呼び出し失敗時。
        """
        if not text.strip():
            raise EmbeddingError("空のテキストは埋め込みできません")

        try:
            resp = requests.post(
                self._endpoint,
                json={"model": self._cfg.embed_model, "prompt": text},
                timeout=self._cfg.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise EmbeddingError(f"Ollama 埋め込み API エラー: {e}") from e

        data = resp.json()
        embedding = data.get("embedding")
        if not embedding or not isinstance(embedding, list):
            raise EmbeddingError(f"埋め込みレスポンスが不正: {data}")

        logger.debug("埋め込み生成完了: dim=%d", len(embedding))
        return embedding