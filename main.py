"""
AI 心の伴走者 - メインエントリポイント。
全コンポーネントの初期化・スレッド管理・ライフサイクル制御を担う。
"""

from __future__ import annotations

import logging
import os
import queue
import re
import signal
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional


# ====================================================================
# CUDA DLLパス解決（他モジュールimport前に実行必須）
# ctranslate2がcublas64_12.dll等を検索できるよう、
# pip installされたnvidia-*パッケージのlib/binディレクトリを
# DLL検索パスに追加する
# ====================================================================
def _register_nvidia_dll_paths() -> None:
    """
    nvidia-cublas-cu12等のpipパッケージに含まれるDLLパスを
    WindowsのDLL検索パスに登録する。

    Notes:
        ctranslate2はC言語レベルでLoadLibrary()を使用してDLLを読み込むため、
        Pythonのos.add_dll_directory()だけでは不十分。
        PATH環境変数への追加が最も確実な方法。
        両方の手法を併用して互換性を最大化する。
    """
    if sys.platform != "win32":
        return
    # venvのsite-packages内のnvidiaディレクトリを探索
    venv_sp = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not venv_sp.is_dir():
        print("[CasaAI] WARNING: nvidia package directory not found")
        return
    # DLLを含むディレクトリを収集
    dll_dirs: list[str] = []
    for bin_dir in venv_sp.rglob("bin"):
        if bin_dir.is_dir() and any(bin_dir.glob("*.dll")):
            dll_dirs.append(str(bin_dir))
    for lib_dir in venv_sp.rglob("lib"):
        if lib_dir.is_dir() and any(lib_dir.glob("*.dll")):
            dll_dirs.append(str(lib_dir))
    if not dll_dirs:
        print("[CasaAI] WARNING: No NVIDIA DLL files found")
        return
    # 方法1: PATH環境変数に追加（ctranslate2のLoadLibrary()対応）
    current_path = os.environ.get("PATH", "")
    new_entries = [d for d in dll_dirs if d not in current_path]
    if new_entries:
        os.environ["PATH"] = ";".join(new_entries) + ";" + current_path
    # 方法2: os.add_dll_directory()も併用（Python側のimport対応）
    for d in dll_dirs:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
    print(f"[CasaAI] Registered {len(dll_dirs)} NVIDIA DLL directories to PATH")
    for d in dll_dirs:
        print(f"  -> {d}")


# DLLパス登録を最初に実行
_register_nvidia_dll_paths()



from config import CONFIG
from installer import run_installer, InstallerError
from database.storage import ChatStorage
from ai.embeddings import EmbeddingService
from ai.memory import MemoryService
from ai.llm import LLMService
from audio.listener import MicrophoneListener, MicrophoneError
from audio.tts import TTSService, TTSError
from audio.transcriber import Transcriber, TranscriptionError
from ui.app import CompanionApp

# ── ロギング設定 ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("companion.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def _extract_english(text: str) -> str:
    """Return only the [EN] portion of a bilingual response for TTS."""
    match = re.search(r'\[EN\]\s*(.+?)(?=\[JA\]|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # fallback: speak full text if no tag found


class CompanionController:
    """
    全コンポーネントを統合するアプリケーションコントローラ。
    スレッド間の協調とライフサイクルを管理する。
    """

    def __init__(self) -> None:
        self._running = False
        # 発話セグメントの受け渡しキュー（リスナー → トランスクライバー）
        self._segment_queue: queue.Queue = queue.Queue(maxsize=10)
        # 書き起こしテキストの受け渡しキュー（トランスクライバー → LLM）
        self._text_queue: queue.Queue = queue.Queue(maxsize=5)

        # ── サービス初期化（TTS は installer 実行後に遅延初期化） ──
        self._storage = ChatStorage(CONFIG.memory.sqlite_path)
        self._embedding_svc = EmbeddingService(CONFIG.ollama)
        self._memory_svc = MemoryService(CONFIG.memory, self._embedding_svc)
        self._llm_svc = LLMService(CONFIG.ollama)
        self._tts_svc: Optional[TTSService] = None  # run() 内で初期化
        self._transcriber = Transcriber(CONFIG.whisper)

        # UI は後で設定する（コールバック必要なため）
        self._app: Optional[CompanionApp] = None
        self._listener: Optional[MicrophoneListener] = None

    def _check_prerequisites(self) -> None:
        """
        起動前提条件を検証する。
        失敗時はユーザーフレンドリーなメッセージを出力して終了。
        """
        if not self._llm_svc.check_availability():
            logger.error(
                "Ollama が起動していないか、モデル '%s' が未インストールです。\n"
                "対処: ollama serve && ollama pull %s",
                CONFIG.ollama.llm_model,
                CONFIG.ollama.llm_model,
            )
            sys.exit(1)

    def _on_mute_toggle(self) -> bool:
        """ミュートトグルコールバック。"""
        if self._listener:
            return self._listener.toggle_mute()
        return False

    def _on_exit(self) -> None:
        """終了コールバック。全スレッドを安全に停止する。"""
        logger.info("終了シグナル受信。シャットダウン開始...")
        self._running = False
        if self._listener:
            self._listener.stop()
        # ワーカースレッドを解放するため番兵値をキューへ投入
        self._segment_queue.put(None)
        self._text_queue.put(None)

    def _transcription_worker(self) -> None:
        """
        セグメントキューを監視し、Whisper で書き起こすワーカースレッド。
        書き起こしテキストをテキストキューへ転送する。
        """
        logger.info("書き起こしワーカー開始")
        while self._running:
            try:
                segment = self._segment_queue.get(timeout=1.0)
                if segment is None:  # 番兵値で終了
                    break

                if self._app:
                    self._app.set_status("⚙️  書き起こし中...")
                    self._app.set_transcription_preview("...")

                text = self._transcriber.transcribe(segment, CONFIG.audio.sample_rate)
                if text.strip():
                    logger.info("書き起こし: '%s'", text)
                    if self._app:
                        self._app.set_transcription_preview(text)
                    self._text_queue.put(text)
                else:
                    if self._app:
                        self._app.set_status("🎙 聴いています...")
                        self._app.set_transcription_preview("")

            except queue.Empty:
                continue
            except TranscriptionError as e:
                logger.error("書き起こしエラー: %s", e)
                if self._app:
                    self._app.set_status("⚠️  書き起こし失敗")
        logger.info("書き起こしワーカー終了")

    def _response_worker(self) -> None:
        """
        テキストキューを監視し、記憶検索・LLM 生成・TTS 再生を実行するワーカースレッド。
        """
        logger.info("応答ワーカー開始")
        while self._running:
            try:
                user_text = self._text_queue.get(timeout=1.0)
                if user_text is None:  # 番兵値で終了
                    break

                if self._app:
                    self._app.set_status("💭 考えています...")
                    self._app.set_transcription_preview("")
                    self._app.add_message("user", user_text)

                # ── 記憶検索とコンテキスト組み立て ──────────
                memories = []
                try:
                    memories = self._memory_svc.retrieve(user_text)
                except Exception as e:
                    logger.warning("記憶検索失敗（スキップ）: %s", e)

                # 直近の会話履歴を LLM コンテキストとして取得
                recent_msgs = self._storage.get_recent_messages(limit=10)
                history = [
                    {"role": m.role, "content": m.content} for m in recent_msgs
                ]

                # ── LLM 応答生成 ──────────────────────────
                try:
                    response = self._llm_svc.generate(user_text, history, memories)
                except Exception as e:
                    logger.error("LLM 生成エラー: %s", e)
                    response = "ごめんなさい、少し調子が悪いみたいです。もう一度話しかけてみてください。"

                if self._app:
                    self._app.add_message("assistant", response)
                    self._app.set_status("🔊 話しています...")

                # ── 記憶保存 ──────────────────────────────
                turn_id_user = f"user_{uuid.uuid4().hex}"
                turn_id_ai = f"ai_{uuid.uuid4().hex}"
                self._storage.save_message("user", user_text)
                self._storage.save_message("assistant", response)

                try:
                    self._memory_svc.store(user_text, "user", turn_id_user)
                    self._memory_svc.store(response, "assistant", turn_id_ai)
                except Exception as e:
                    logger.warning("記憶保存失敗（スキップ）: %s", e)

                # ── TTS 再生（English のみ読み上げ） ─────
                # マイクをミュートして TTS 音声の自己入力を防ぐ
                if self._listener:
                    self._listener.mute()
                try:
                    if self._tts_svc:
                        self._tts_svc.speak(_extract_english(response))
                except TTSError as e:
                    logger.error("TTS 再生エラー: %s", e)
                finally:
                    if self._listener:
                        self._listener.unmute()

                if self._app:
                    self._app.set_status("🎙 聴いています...")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error("応答ワーカー予期せぬエラー: %s", e, exc_info=True)
                if self._app:
                    self._app.set_status("⚠️  エラーが発生しました")
        logger.info("応答ワーカー終了")

    def run(self) -> None:
        """アプリケーションを起動する。"""
        CONFIG.validate()
        CONFIG.ensure_directories()

        # ── 自動インストール ──────────────────────────
        try:
            run_installer()
        except InstallerError as e:
            logger.error("インストール失敗: %s", e)
            sys.exit(1)

        # ── TTS 初期化（Piper モデルロード） ────────────
        try:
            self._tts_svc = TTSService(CONFIG.piper)
        except TTSError as e:
            logger.error("TTS 初期化失敗: %s", e)
            sys.exit(1)

        # ── 前提条件確認 ──────────────────────────────
        self._check_prerequisites()

        # ── UI 構築 ───────────────────────────────────
        self._app = CompanionApp(
            cfg=CONFIG.ui,
            on_mute_toggle=self._on_mute_toggle,
            on_exit=self._on_exit,
        )

        # 過去の会話履歴を UI に復元
        history = self._storage.get_all_messages()
        if history:
            self._app.load_history(history)
            self._app.show_system_message("過去の会話を読み込みました")
        else:
            self._app.show_system_message(
                "はじめまして。いつでも話しかけてください。"
            )

        # ── マイクリスナー初期化 ──────────────────────
        self._listener = MicrophoneListener(
            audio_cfg=CONFIG.audio,
            vad_cfg=CONFIG.vad,
            segment_queue=self._segment_queue,
            status_callback=self._app.set_status,
        )

        # ── ワーカースレッド開始 ──────────────────────
        self._running = True
        threads = [
            threading.Thread(target=self._transcription_worker, daemon=True, name="transcriber"),
            threading.Thread(target=self._response_worker, daemon=True, name="responder"),
        ]
        for t in threads:
            t.start()

        # ── マイク開始 ────────────────────────────────
        try:
            self._listener.start()
        except MicrophoneError as e:
            logger.error("マイク起動失敗: %s", e)
            self._app.show_system_message(f"⚠️ マイクエラー: {e}")

        # ── UI メインループ（ブロッキング） ───────────
        logger.info("アプリケーション起動完了")
        self._app.run()

        # ── シャットダウン待機 ────────────────────────
        for t in threads:
            t.join(timeout=5.0)
        logger.info("アプリケーション終了")


if __name__ == "__main__":
    controller = CompanionController()
    controller.run()