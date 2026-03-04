"""
Piper TTS を使ったローカル英語音声合成モジュール。
ONNX モデルをオフラインで読み込み、sounddevice で再生する。
"""

from __future__ import annotations

import io
import logging
import threading
import wave

import sounddevice as sd
import soundfile as sf

from config import PiperConfig

logger = logging.getLogger(__name__)


class TTSError(Exception):
    """TTS 処理の失敗を表す例外。"""


class TTSService:
    """
    Piper TTS による英語音声合成・再生サービス。

    Args:
        piper_cfg: Piper モデル設定。
    """

    def __init__(self, piper_cfg: PiperConfig) -> None:
        self._model_path = piper_cfg.model_path
        self._use_cuda = piper_cfg.use_cuda
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._voice = self._load_voice()

    def _load_voice(self):
        """Piper ONNX モデルをロードする。"""
        # piper-tts 1.4+ は espeak-ng を内包。piper-phonemize は不要。
        # https://github.com/OHF-Voice/piper1-gpl
        try:
            from piper.voice import PiperVoice  # type: ignore
        except ImportError as e:
            raise TTSError(
                "piper-tts がインストールされていません。"
                " pip install piper-tts>=1.4.0 を実行してください。"
            ) from e

        if not self._model_path.exists():
            raise TTSError(f"Piper モデルファイルが見つかりません: {self._model_path}")

        try:
            voice = PiperVoice.load(str(self._model_path), use_cuda=self._use_cuda)
            logger.info("Piper モデルロード完了: %s", self._model_path)
            return voice
        except Exception as e:
            raise TTSError(f"Piper モデルのロードに失敗しました: {e}") from e

    def speak(self, text: str) -> None:
        """
        テキストを音声合成して再生する（ブロッキング）。
        前の再生が進行中であれば停止してから新規再生する。

        Args:
            text: 読み上げるテキスト。

        Raises:
            TTSError: 合成または再生に失敗した場合。
        """
        if not text.strip():
            return

        self.stop()

        wav_bytes = self._synthesize(text)
        self._play(wav_bytes)

    def _synthesize(self, text: str) -> bytes:
        """
        Piper でテキストを WAV バイト列に変換する。

        Args:
            text: 合成テキスト。

        Returns:
            WAV 音声データ。

        Raises:
            TTSError: 合成失敗時。
        """
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file)
            wav_data = buf.getvalue()
            if len(wav_data) <= 44:  # WAV header only = no audio frames
                raise TTSError("音声合成結果が空です（Piper モデルエラー）")
            logger.info("TTS 合成完了: %d bytes", len(wav_data))
            return wav_data
        except TTSError:
            raise
        except Exception as e:
            raise TTSError(f"Piper 音声合成失敗: {e}") from e

    def _play(self, wav_bytes: bytes) -> None:
        """
        WAV バイト列を sounddevice でブロッキング再生する。
        stop() 呼び出しで途中停止可能。

        Args:
            wav_bytes: 再生する WAV データ。

        Raises:
            TTSError: 再生失敗時。
        """
        try:
            data, samplerate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            self._stop_event.clear()
            sd.play(data, samplerate)
            # Poll until playback finishes or stop is requested.
            # sd.get_stream() raises PortAudioError when the stream has already
            # ended, so we catch it to exit the loop gracefully.
            while True:
                if self._stop_event.is_set():
                    sd.stop()
                    break
                try:
                    if not sd.get_stream().active:
                        break
                except Exception:
                    break  # stream ended before we could check
                sd.sleep(50)
            logger.debug("TTS 再生完了: %d samples @ %d Hz", len(data), samplerate)
        except Exception as e:
            raise TTSError(f"音声再生失敗: {e}") from e

    def stop(self) -> None:
        """現在の再生を停止する（次の発話開始前に呼び出す）。"""
        self._stop_event.set()
        sd.stop()
