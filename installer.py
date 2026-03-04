"""
起動前チェックモジュール。
Piper TTS モデルの存在確認・自動ダウンロードを行う。
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from config import CONFIG

logger = logging.getLogger(__name__)

# HuggingFace リポジトリ上のモデルパス (rhasspy/piper-voices)
_HF_REPO = "rhasspy/piper-voices"
_MODEL_FILES = [
    "en/en_US/amy/medium/en_US-amy-medium.onnx",
    "en/en_US/amy/medium/en_US-amy-medium.onnx.json",
]


class InstallerError(Exception):
    """起動前チェックの失敗を表す例外。"""


def ensure_piper_model(model_path: Path) -> None:
    """
    Piper ONNX モデルと設定ファイルが存在するか確認し、
    なければ HuggingFace Hub から自動ダウンロードする。

    Args:
        model_path: モデル .onnx ファイルのパス。

    Raises:
        InstallerError: ダウンロード失敗時。
    """
    config_path = Path(str(model_path) + ".json")
    models_dir = model_path.parent
    models_dir.mkdir(parents=True, exist_ok=True)

    if model_path.exists() and config_path.exists():
        logger.info("Piper モデル確認完了: %s", model_path)
        return

    logger.info("Piper モデルが見つかりません。ダウンロードを開始します...")
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as e:
        raise InstallerError(
            "huggingface-hub がインストールされていません。"
            " pip install huggingface-hub を実行してください。"
        ) from e

    try:
        for hf_path in _MODEL_FILES:
            filename = hf_path.split("/")[-1]
            dest = models_dir / filename
            if dest.exists():
                continue
            logger.info("ダウンロード中: %s", filename)
            downloaded = hf_hub_download(
                repo_id=_HF_REPO,
                filename=hf_path,
                local_dir=str(models_dir),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download はサブディレクトリに保存する場合があるので移動
            downloaded_path = Path(downloaded)
            if downloaded_path != dest:
                downloaded_path.rename(dest)
        logger.info("Piper モデルダウンロード完了: %s", models_dir)
    except Exception as e:
        raise InstallerError(
            f"Piper モデルのダウンロードに失敗しました。\n"
            f"手動でモデルを {models_dir} に配置してください。\n"
            f"ダウンロード元: https://huggingface.co/{_HF_REPO}\n"
            f"詳細: {e}"
        ) from e


def ensure_piper_tts() -> None:
    """
    piper-tts 1.4.0+ がインストールされているか確認し、古ければアップグレードする。

    piper-tts 1.4+ は espeak-ng を内包しており piper-phonemize は不要。
    https://github.com/OHF-Voice/piper1-gpl

    Raises:
        InstallerError: アップグレード失敗時。
    """
    try:
        import importlib.metadata
        ver = importlib.metadata.version("piper-tts")
        major, minor, *_ = (int(x) for x in ver.split("."))
        if (major, minor) >= (1, 4):
            logger.info("piper-tts %s: OK", ver)
            return
        logger.warning("piper-tts %s は古すぎます。1.4.0+ にアップグレードします...", ver)
    except Exception:
        logger.info("piper-tts バージョン確認不可。インストールを試みます...")

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "piper-tts>=1.4.0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        logger.info("piper-tts アップグレード完了")
    except subprocess.CalledProcessError as e:
        raise InstallerError(
            f"piper-tts のアップグレードに失敗しました: {e}"
        ) from e


def run_installer() -> None:
    """
    起動前チェックを実行する。
    main.py の起動時に呼び出す。

    Raises:
        InstallerError: チェック失敗時。
    """
    CONFIG.ensure_directories()
    ensure_piper_tts()
    ensure_piper_model(CONFIG.piper.model_path)
    logger.info("起動前チェック完了。")
