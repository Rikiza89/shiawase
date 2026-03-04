"""
faster-whisper による日本語音声書き起こしモジュール。
CUDA 使用不可時は CPU にフォールバックする。
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from huggingface_hub.utils import are_progress_bars_disabled
import huggingface_hub

from config import WhisperConfig

# モデルリポジトリ名（faster-whisper large-v3 公式）
_MODEL_REPO = "Systran/faster-whisper-large-v3"


def ensure_whisper_model_downloaded(model_size: str) -> None:
    """
    Whisper モデルを HuggingFace Hub から明示的にダウンロードする。
    進捗バーをコンソールに表示し、ダウンロード済みならスキップする。

    Args:
        model_size: モデルサイズ文字列（例: "large-v3"）。
    """
    repo_id = f"Systran/faster-whisper-{model_size}"
    print(f"\n[Whisper] モデル確認中: {repo_id}")
    print("[Whisper] 初回のみダウンロードが必要です（約3GB）。しばらくお待ちください...\n")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            # 進捗バーを強制的に有効化
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
        print(f"\n[Whisper] モデルの準備完了: {repo_id}\n")
    except Exception as e:
        # ダウンロード失敗は WhisperModel ロード時に再度エラーになるため警告のみ
        print(f"[Whisper] 警告: 事前ダウンロード失敗 ({e})。モデルロード時に再試行します。")

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """書き起こし処理の失敗を表す例外。"""


class Transcriber:
    """
    faster-whisper large-v3 による日本語音声書き起こしサービス。
    CUDA 使用不可時は自動的に CPU / int8 へフォールバックする。

    Args:
        cfg: Whisper 設定。
    """

    def __init__(self, cfg: WhisperConfig) -> None:
        self._cfg = cfg
        self._model = self._load_model()
        self._using_cpu = False  # CUDA 障害後の CPU フォールバック状態

    def _load_model(self) -> WhisperModel:
        """
        Whisper モデルをロードする。
        未ダウンロードの場合は進捗表示付きで取得してからロードする。
        CUDA 初期化失敗時は CPU フォールバックを試みる。

        Returns:
            WhisperModel インスタンス。

        Raises:
            TranscriptionError: モデルロード全失敗時。
        """
        # 進捗表示付きで事前ダウンロード（キャッシュ済みなら即終了）
        ensure_whisper_model_downloaded(self._cfg.model_size)
        try:
            model = WhisperModel(
                self._cfg.model_size,
                device=self._cfg.device,
                compute_type=self._cfg.compute_type,
            )
            logger.info(
                "Whisper モデルロード完了: device=%s, compute_type=%s",
                self._cfg.device,
                self._cfg.compute_type,
            )
            return model
        except Exception as e:
            logger.warning(
                "CUDA でのモデルロード失敗。CPU にフォールバック: %s", e
            )
            try:
                model = WhisperModel(
                    self._cfg.model_size,
                    device=self._cfg.fallback_device,
                    compute_type=self._cfg.fallback_compute_type,
                )
                logger.info("Whisper モデル（CPU フォールバック）ロード完了")
                return model
            except Exception as e2:
                raise TranscriptionError(f"Whisper モデルロード全失敗: {e2}") from e2

    def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        PCM バイト列を日本語テキストに書き起こす。

        Args:
            pcm_bytes: int16 PCM 音声データ。
            sample_rate: サンプリングレート（Hz）。

        Returns:
            書き起こしテキスト。無音や認識不可なら空文字列。

        Raises:
            TranscriptionError: 書き起こし処理中の例外。
        """
        if not pcm_bytes:
            return ""

        # int16 → float32 正規化（Whisper 入力仕様）
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            segments, info = self._model.transcribe(
                audio,
                language=self._cfg.language,
                beam_size=5,
                vad_filter=True,           # 内蔵 VAD で二重フィルタリング
                vad_parameters={"threshold": 0.5},
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            # cublas / CUDA ライブラリが見つからない場合は CPU に切り替えて再試行
            err_str = str(e).lower()
            if not self._using_cpu and any(k in err_str for k in ("cublas", "cuda", "cudnn", "dll")):
                logger.warning("CUDA 推論エラー、CPU にフォールバックします: %s", e)
                try:
                    self._model = WhisperModel(
                        self._cfg.model_size,
                        device=self._cfg.fallback_device,
                        compute_type=self._cfg.fallback_compute_type,
                    )
                    self._using_cpu = True
                    logger.info("CPU フォールバックモデルロード完了。再試行します。")
                    segments, info = self._model.transcribe(
                        audio,
                        language=self._cfg.language,
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters={"threshold": 0.5},
                    )
                    text = " ".join(seg.text.strip() for seg in segments).strip()
                except Exception as e2:
                    raise TranscriptionError(f"書き起こし処理エラー（CPU フォールバック失敗）: {e2}") from e2
            else:
                raise TranscriptionError(f"書き起こし処理エラー: {e}") from e

        logger.debug(
            "書き起こし完了: lang=%s, prob=%.2f, text='%s'",
            info.language,
            info.language_probability,
            text[:50],
        )
        return text