"""
アプリケーション全体の設定管理モジュール。
全設定は外部化・型付け・起動時検証済み。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ── ベースディレクトリ ──────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent.resolve()


@dataclass(frozen=True)
class PiperConfig:
    """Piper TTS 設定。"""
    # en_US-amy-medium.onnx (22050 Hz) を既定ボイスとする
    model_path: Path = BASE_DIR / "models" / "en_US-amy-medium.onnx"
    use_cuda: bool = False


@dataclass(frozen=True)
class WhisperConfig:
    """faster-whisper 音声認識設定。"""
    model_size: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "ja"
    # CUDA 使用不可時のフォールバック
    fallback_device: str = "cpu"
    fallback_compute_type: str = "int8"


@dataclass(frozen=True)
class VADConfig:
    """WebRTC VAD 設定。"""
    # 感度: 0(低) ～ 3(高)
    aggressiveness: int = 2
    sample_rate: int = 16000
    frame_duration_ms: int = 30        # 30ms フレーム（VAD 仕様）
    silence_threshold_ms: int = 1200   # 発話終了判定の無音時間
    min_speech_ms: int = 300           # 最小発話長（ノイズ除去）


@dataclass(frozen=True)
class AudioConfig:
    """マイク入力設定。"""
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    blocksize: int = 480   # 30ms @ 16kHz


@dataclass(frozen=True)
class OllamaConfig:
    """Ollama HTTP API 設定。"""
    base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b-instruct-q4_K_M"
    embed_model: str = "nomic-embed-text:latest"
    timeout: int = 120
    # LLM 生成パラメータ
    temperature: float = 0.75
    top_p: float = 0.9
    max_tokens: int = 512


@dataclass(frozen=True)
class MemoryConfig:
    """ベクトルメモリ・チャット履歴設定。"""
    db_dir: Path = BASE_DIR / "database"
    sqlite_path: Path = BASE_DIR / "database" / "chat_history.db"
    chroma_dir: Path = BASE_DIR / "database" / "chroma"
    collection_name: str = "companion_memory"
    # コンテキスト注入する類似記憶の上限
    top_k_memories: int = 5


@dataclass(frozen=True)
class UIConfig:
    """UI テーマ・レイアウト設定。"""
    title: str = "AI 心の伴走者"
    width: int = 900
    height: int = 700
    bg_color: str = "#1e1e2e"
    surface_color: str = "#2a2a3e"
    accent_color: str = "#cba6f7"
    ai_header_color: str = "#89b4fa"   # AI 発話者ラベル色（Catppuccin ブルー）
    text_color: str = "#cdd6f4"
    subtext_color: str = "#a6adc8"
    user_bubble_color: str = "#313244"
    ai_bubble_color: str = "#45475a"
    font_family: str = "Yu Gothic UI"
    font_size: int = 11
    font_size_small: int = 9


@dataclass(frozen=True)
class AppConfig:
    """アプリケーション統合設定。起動時に検証実行。"""
    piper: PiperConfig = field(default_factory=PiperConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    def validate(self) -> None:
        """
        設定の整合性を検証。
        起動時に必ず呼び出すこと。

        Raises:
            ValueError: 設定値が無効な場合。
        """
        if self.vad.aggressiveness not in (0, 1, 2, 3):
            raise ValueError(f"VAD aggressiveness は 0-3 の範囲: {self.vad.aggressiveness}")
        if self.audio.sample_rate != self.vad.sample_rate:
            raise ValueError("audio.sample_rate と vad.sample_rate は一致必須")
        if not self.ollama.base_url.startswith("http"):
            raise ValueError(f"無効な Ollama URL: {self.ollama.base_url}")

    def ensure_directories(self) -> None:
        """必要なディレクトリを一括作成。"""
        dirs = [
            self.memory.db_dir,
            self.memory.chroma_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# ── シングルトン設定インスタンス ────────────────────────────────────
CONFIG = AppConfig()
