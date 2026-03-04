"""
Tkinter ダークモード UI モジュール。
チャット表示・状態ラベル・ミュートボタンを提供する。
全 I/O 操作はコールバック経由で疎結合を維持する。
"""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from datetime import datetime
from tkinter import font as tkfont
from tkinter import scrolledtext
from typing import Callable, Optional

from config import UIConfig

logger = logging.getLogger(__name__)


class CompanionApp:
    """
    AI 心の伴走者 メインウィンドウ。

    Args:
        cfg: UI テーマ設定。
        on_mute_toggle: ミュートボタン押下コールバック。
        on_exit: 終了ボタン押下コールバック。
    """

    def __init__(
        self,
        cfg: UIConfig,
        on_mute_toggle: Callable[[], bool],
        on_exit: Callable[[], None],
    ) -> None:
        self._cfg = cfg
        self._on_mute_toggle = on_mute_toggle
        self._on_exit = on_exit

        # UI スレッドからの安全な更新用キュー
        self._ui_queue: queue.Queue = queue.Queue()

        self._root = tk.Tk()
        self._build_ui()
        self._poll_ui_queue()

    def _build_ui(self) -> None:
        """ウィジェットを構築する。"""
        c = self._cfg
        root = self._root
        root.title(c.title)
        root.configure(bg=c.bg_color)
        root.protocol("WM_DELETE_WINDOW", self._handle_exit)

        # ── ウィンドウを画面中央に配置・最小サイズ設定 ──────────
        root.update_idletasks()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = max(0, (screen_w - c.width) // 2)
        y = max(0, (screen_h - c.height) // 2)
        root.geometry(f"{c.width}x{c.height}+{x}+{y}")
        root.minsize(700, 500)

        # ── キーボードショートカット ──────────────────────────
        root.bind("<Control-m>", lambda _: self._handle_mute())
        root.bind("<Control-M>", lambda _: self._handle_mute())

        # ── フォント定義 ──────────────────────────────
        base_font = tkfont.Font(family=c.font_family, size=c.font_size)
        small_font = tkfont.Font(family=c.font_family, size=c.font_size_small)
        bold_font = tkfont.Font(family=c.font_family, size=c.font_size, weight="bold")
        title_font = tkfont.Font(family=c.font_family, size=14, weight="bold")

        # ── タイトルバー ──────────────────────────────
        title_frame = tk.Frame(root, bg=c.surface_color, pady=10)
        title_frame.pack(fill=tk.X)
        tk.Label(
            title_frame,
            text="💛  AI 心の伴走者",
            bg=c.surface_color,
            fg=c.accent_color,
            font=title_font,
        ).pack()
        tk.Label(
            title_frame,
            text="あなたの心に、そっと寄り添います",
            bg=c.surface_color,
            fg=c.subtext_color,
            font=small_font,
        ).pack()

        # ── タイトル下のアクセントライン ─────────────────
        tk.Frame(root, bg=c.accent_color, height=1).pack(fill=tk.X)

        # ── チャットウィンドウ ────────────────────────
        chat_frame = tk.Frame(root, bg=c.bg_color)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 4))

        self._chat_box = scrolledtext.ScrolledText(
            chat_frame,
            state=tk.DISABLED,
            wrap=tk.WORD,
            bg=c.surface_color,
            fg=c.text_color,
            font=base_font,
            relief=tk.FLAT,
            insertbackground=c.text_color,
            selectbackground=c.accent_color,
            padx=12,
            pady=10,
            spacing3=4,  # 段落後の追加スペース
        )
        self._chat_box.pack(fill=tk.BOTH, expand=True)

        # テキストタグ定義
        self._chat_box.tag_configure(
            "user",
            foreground=c.accent_color,  # ユーザー：紫（Catppuccin mauve）
            font=bold_font,
        )
        self._chat_box.tag_configure(
            "ai_header",
            foreground=c.ai_header_color,  # AI：青（Catppuccin blue）
            font=bold_font,
        )
        self._chat_box.tag_configure("ai", foreground=c.text_color, font=base_font)
        self._chat_box.tag_configure(
            "system",
            foreground=c.subtext_color,
            font=small_font,
            justify=tk.CENTER,
        )
        self._chat_box.tag_configure(
            "timestamp",
            foreground=c.subtext_color,
            font=small_font,
        )

        # ── ライブ書き起こしプレビュー（背景付きフレーム） ──
        preview_frame = tk.Frame(root, bg=c.surface_color, pady=6)
        preview_frame.pack(fill=tk.X, padx=12, pady=(0, 0))
        self._transcription_var = tk.StringVar(value="　")  # 空白で高さ維持
        tk.Label(
            preview_frame,
            textvariable=self._transcription_var,
            bg=c.surface_color,
            fg=c.subtext_color,
            font=small_font,
            anchor="w",
            wraplength=c.width - 50,
        ).pack(fill=tk.X, padx=10)

        # ── ステータスラベル ──────────────────────────
        self._status_var = tk.StringVar(value="🎙 聴いています...")
        tk.Label(
            root,
            textvariable=self._status_var,
            bg=c.bg_color,
            fg=c.accent_color,
            font=small_font,
            anchor="w",
        ).pack(fill=tk.X, padx=14, pady=(4, 2))

        # ── コントロールボタン ────────────────────────
        btn_frame = tk.Frame(root, bg=c.bg_color)
        btn_frame.pack(pady=(2, 12))

        self._mute_btn = tk.Button(
            btn_frame,
            text="🎙 ミュート",
            command=self._handle_mute,
            bg=c.surface_color,
            fg=c.text_color,
            activebackground=c.accent_color,
            activeforeground=c.bg_color,
            font=base_font,
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2",
            bd=0,
        )
        self._mute_btn.pack(side=tk.LEFT, padx=8)

        tk.Label(
            btn_frame,
            text="(Ctrl+M)",
            bg=c.bg_color,
            fg=c.subtext_color,
            font=small_font,
        ).pack(side=tk.LEFT, padx=0)

        tk.Button(
            btn_frame,
            text="✕ 終了",
            command=self._handle_exit,
            bg=c.surface_color,
            fg="#f38ba8",
            activebackground="#f38ba8",
            activeforeground=c.bg_color,
            font=base_font,
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2",
            bd=0,
        ).pack(side=tk.LEFT, padx=8)

    # ── 非同期 UI 更新キュー ─────────────────────────────────────

    def _poll_ui_queue(self) -> None:
        """
        メインスレッドで UI 更新キューをポーリングする。
        tkinter はメインスレッドからのみ操作可能なため必須。
        """
        try:
            while True:
                fn = self._ui_queue.get_nowait()
                fn()
        except queue.Empty:
            pass
        self._root.after(50, self._poll_ui_queue)

    def _schedule(self, fn: Callable) -> None:
        """UI 更新をメインスレッドのキューへスケジュールする。"""
        self._ui_queue.put(fn)

    # ── ヘルパー ─────────────────────────────────────────────────

    @staticmethod
    def _format_timestamp(ts: str) -> str:
        """ISO 8601 タイムスタンプを読みやすいフォーマットに変換する。"""
        try:
            dt = datetime.fromisoformat(ts)
            return dt.strftime("%Y/%m/%d %H:%M")
        except Exception:
            return ""

    # ── 公開 API ────────────────────────────────────────────────

    def add_message(self, role: str, content: str, timestamp: str = "") -> None:
        """
        チャットウィンドウにメッセージを追加する（スレッドセーフ）。

        Args:
            role: "user" または "assistant"。
            content: メッセージ本文。
            timestamp: ISO 8601 タイムスタンプ（省略時は非表示）。
        """
        def _update() -> None:
            self._chat_box.configure(state=tk.NORMAL)
            if timestamp:
                ts_str = self._format_timestamp(timestamp)
                if ts_str:
                    self._chat_box.insert(tk.END, f"  {ts_str}\n", "timestamp")
            if role == "user":
                self._chat_box.insert(tk.END, "\nあなた：\n", "user")
                self._chat_box.insert(tk.END, f"  {content}\n", "ai")
            else:
                self._chat_box.insert(tk.END, "\n伴走者：\n", "ai_header")
                self._chat_box.insert(tk.END, f"  {content}\n", "ai")
            self._chat_box.configure(state=tk.DISABLED)
            self._chat_box.see(tk.END)
        self._schedule(_update)

    def set_status(self, text: str) -> None:
        """ステータスラベルを更新する（スレッドセーフ）。"""
        self._schedule(lambda: self._status_var.set(text))

    def set_transcription_preview(self, text: str) -> None:
        """ライブ書き起こしプレビューを更新する（スレッドセーフ）。"""
        display = f"📝 {text}" if text else "　"  # 全角スペースで高さを維持
        self._schedule(lambda: self._transcription_var.set(display))

    def show_system_message(self, text: str) -> None:
        """システム通知をチャットに表示する（スレッドセーフ）。"""
        def _update() -> None:
            self._chat_box.configure(state=tk.NORMAL)
            self._chat_box.insert(tk.END, f"\n── {text} ──\n", "system")
            self._chat_box.configure(state=tk.DISABLED)
            self._chat_box.see(tk.END)
        self._schedule(_update)

    def load_history(self, messages: list) -> None:
        """
        起動時に過去の会話履歴を一括表示する。

        Args:
            messages: ChatMessage オブジェクトのリスト。
        """
        for msg in messages:
            self.add_message(msg.role, msg.content, timestamp=msg.timestamp)

    # ── ボタンハンドラ ───────────────────────────────────────────

    def _handle_mute(self) -> None:
        """ミュートボタン処理。ミュート中は赤背景で視覚的に明示する。"""
        is_muted = self._on_mute_toggle()
        if is_muted:
            self._mute_btn.config(
                text="🔇 ミュート解除",
                bg="#f38ba8",
                fg=self._cfg.bg_color,
            )
        else:
            self._mute_btn.config(
                text="🎙 ミュート",
                bg=self._cfg.surface_color,
                fg=self._cfg.text_color,
            )

    def _handle_exit(self) -> None:
        """終了ボタン・ウィンドウ閉じる処理。"""
        self._on_exit()
        self._root.destroy()

    def run(self) -> None:
        """Tkinter メインループを開始する。"""
        self._root.mainloop()
