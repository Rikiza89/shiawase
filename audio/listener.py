"""
マイク入力の継続的ストリームとキュー転送モジュール。
VADProcessor と連携し、発話セグメントを非同期キューへ送る。
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from audio.vad import VADProcessor
from config import AudioConfig, VADConfig

logger = logging.getLogger(__name__)


class MicrophoneError(Exception):
    """マイク操作の失敗を表す例外。"""


class MicrophoneListener:
    """
    sounddevice を使った継続的マイク入力リスナー。
    発話セグメントを segment_queue へ投入する。

    Args:
        audio_cfg: マイク設定。
        vad_cfg: VAD 設定。
        segment_queue: 発話セグメント（bytes）の出力キュー。
        status_callback: 状態テキストを UI へ通知するコールバック。
    """

    def __init__(
        self,
        audio_cfg: AudioConfig,
        vad_cfg: VADConfig,
        segment_queue: queue.Queue,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._audio_cfg = audio_cfg
        self._vad_processor = VADProcessor(vad_cfg)
        self._segment_queue = segment_queue
        self._status_cb = status_callback or (lambda _: None)

        self._muted: bool = False
        self._running: bool = False
        self._stream: Optional[sd.InputStream] = None
        self._thread: Optional[threading.Thread] = None

        # 残余サンプルバッファ（フレームサイズ整合用）
        self._remainder = np.array([], dtype=np.int16)

    @property
    def is_muted(self) -> bool:
        return self._muted

    def mute(self) -> None:
        """
        マイクをミュートし、VAD バッファをリセットする。
        TTS 再生中に呼び出してハウリングを防ぐ。
        """
        self._muted = True
        self._vad_processor.reset()
        self._remainder = np.array([], dtype=np.int16)
        # VAD が発話中と判断していたセグメントをキューから除去
        while not self._segment_queue.empty():
            try:
                self._segment_queue.get_nowait()
            except Exception:
                break
        self._status_cb("🔇 ミュート中")
        logger.info("マイク ミュート（TTS 再生中）")

    def unmute(self) -> None:
        """マイクのミュートを解除する。"""
        self._muted = False
        self._status_cb("🎙 聴いています...")
        logger.info("マイク ミュート解除")

    def toggle_mute(self) -> bool:
        """
        ミュート状態を切り替える。

        Returns:
            切り替え後のミュート状態。
        """
        self._muted = not self._muted
        if self._muted:
            self._vad_processor.reset()
            self._status_cb("🔇 ミュート中")
        else:
            self._status_cb("🎙 聴いています...")
        logger.info("ミュート状態: %s", self._muted)
        return self._muted

    def start(self) -> None:
        """
        マイク入力ストリームを開始する。

        Raises:
            MicrophoneError: マイクが使用不可の場合。
        """
        if self._running:
            return

        try:
            self._stream = sd.InputStream(
                samplerate=self._audio_cfg.sample_rate,
                channels=self._audio_cfg.channels,
                dtype=self._audio_cfg.dtype,
                blocksize=self._audio_cfg.blocksize,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            self._status_cb("🎙 聴いています...")
            logger.info("マイク入力開始")
        except sd.PortAudioError as e:
            raise MicrophoneError(f"マイクを開けません: {e}") from e

    def stop(self) -> None:
        """マイク入力を停止する。"""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._vad_processor.reset()
        logger.info("マイク入力停止")

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """
        sounddevice コールバック。
        フレームを VAD に渡し、発話完了時にキューへ投入する。
        このメソッドはオーディオスレッドから呼び出される点に注意。
        """
        if not self._running or self._muted:
            return

        if status:
            logger.warning("sounddevice ステータス: %s", status)

        # フラット化して残余バッファと結合
        samples = np.concatenate([self._remainder, indata.flatten()])
        frame_size = self._vad_processor.frame_samples

        # フレームサイズ単位で VAD 処理
        i = 0
        while i + frame_size <= len(samples):
            frame = samples[i : i + frame_size]
            segment = self._vad_processor.process_frame(frame)
            if segment is not None:
                try:
                    self._segment_queue.put_nowait(segment)
                    logger.debug("セグメントをキューに投入")
                except queue.Full:
                    logger.warning("セグメントキューが満杯。セグメントを破棄")
            i += frame_size

        # 処理できなかった残余を保持
        self._remainder = samples[i:]