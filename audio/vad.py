"""
WebRTC VAD を使った音声区間検出モジュール。
無音区間に基づいて発話セグメントを切り出す。
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional

import numpy as np
import webrtcvad

from config import VADConfig

logger = logging.getLogger(__name__)


class VADProcessor:
    """
    WebRTC VAD による発話区間検出プロセッサ。
    フレーム単位で音声を受け取り、発話完了セグメントを返す。

    Args:
        cfg: VAD 設定。
    """

    def __init__(self, cfg: VADConfig) -> None:
        self._cfg = cfg
        self._vad = webrtcvad.Vad(cfg.aggressiveness)

        # フレームサイズ（サンプル数）
        self._frame_samples = int(cfg.sample_rate * cfg.frame_duration_ms / 1000)
        # 無音フレーム数でタイムアウト判定
        self._silence_frames = int(cfg.silence_threshold_ms / cfg.frame_duration_ms)
        # 最小発話フレーム数
        self._min_speech_frames = int(cfg.min_speech_ms / cfg.frame_duration_ms)

        # 状態管理
        self._speech_frames: List[bytes] = []
        self._silence_count: int = 0
        self._is_speaking: bool = False
        # リングバッファ：発話前の前置フレームを保持
        self._pre_buffer: deque[bytes] = deque(maxlen=10)

    @property
    def frame_samples(self) -> int:
        """1フレームのサンプル数。"""
        return self._frame_samples

    def process_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """
        1フレーム分の音声を処理し、発話完了時に PCM バイト列を返す。

        Args:
            frame: int16 numpy 配列（frame_samples 個）。

        Returns:
            発話セグメントの PCM バイト列。発話未完了なら None。
        """
        # webrtcvad は bytes を要求
        frame_bytes = frame.tobytes()

        try:
            is_speech = self._vad.is_speech(frame_bytes, self._cfg.sample_rate)
        except Exception as e:
            logger.warning("VAD 処理エラー（スキップ）: %s", e)
            return None

        if is_speech:
            if not self._is_speaking:
                # 発話開始：前置バッファを含めて音声収集を開始
                self._is_speaking = True
                self._speech_frames = list(self._pre_buffer)
            self._silence_count = 0
            self._speech_frames.append(frame_bytes)
        else:
            self._pre_buffer.append(frame_bytes)
            if self._is_speaking:
                self._silence_count += 1
                self._speech_frames.append(frame_bytes)

                # 無音が閾値を超えたら発話終了と判定
                if self._silence_count >= self._silence_frames:
                    return self._finalize_segment()
        return None

    def _finalize_segment(self) -> Optional[bytes]:
        """
        収集したフレームを結合してセグメントを確定する。
        最小発話長に満たない場合はノイズとして破棄する。

        Returns:
            有効セグメントの PCM バイト列、または None。
        """
        frames = self._speech_frames
        self._speech_frames = []
        self._silence_count = 0
        self._is_speaking = False

        if len(frames) < self._min_speech_frames:
            logger.debug("短すぎる発話をノイズとして破棄: %d frames", len(frames))
            return None

        segment = b"".join(frames)
        logger.debug("発話セグメント確定: %.2f 秒", len(segment) / 2 / self._cfg.sample_rate)
        return segment

    def reset(self) -> None:
        """状態をリセットする（ミュート時・エラー回復時に使用）。"""
        self._speech_frames = []
        self._silence_count = 0
        self._is_speaking = False
        self._pre_buffer.clear()