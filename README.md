# 💛 AI 心の伴走者

ローカル完結型 AI 感情サポートコンパニオン。常時マイク待機・日本語音声認識・温かい応答・日本語 TTS 読み上げ。

---

## 動作要件

| 項目 | 要件 |
|------|------|
| OS | Windows 11 |
| Python | 3.11+ |
| GPU | NVIDIA RTX（CUDA 12.1+）推奨 |
| RAM | 16GB 以上（32GB 推奨） |
| インターネット | 初回起動時のみ（音声モデル自動ダウンロード） |

---

## セットアップ手順

### 1. Python インストール
[python.org](https://www.python.org/downloads/) から Python 3.11+ をインストール。  
インストール時に **「Add Python to PATH」にチェック** を入れること。

### 2. CUDA ツールキット（GPU 使用時）
[CUDA Toolkit 12.1+](https://developer.nvidia.com/cuda-downloads) をインストール。

### 3. Ollama インストール
[ollama.com](https://ollama.com/) からインストール後、以下を実行：

```bash
# Ollama サーバー起動
ollama serve

# 別ターミナルでモデルをプル
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull nomic-embed-text:latest
```

### 4. 依存パッケージインストール

```bash
cd ai_companion　(又は保存したフォルダーの名前)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU 版 PyTorch** を使用する場合は、公式コマンドで上書きインストール：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

> **piper-tts について**
> バージョン 1.4+ から espeak-ng が Python ホイールに内包されました。
> `piper-phonemize` の個別インストールは不要です。詳細: [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)

### 5. 起動

```bash
python main.py
```

**初回起動時**、以下が自動実行されます：
1. `voices/ja_female/` へ日本語女性音声モデル（nakamura medium）をダウンロード
2. Whisper 音声認識モデル（デフォルト: large-v3、約3GB）を HuggingFace からダウンロード

> piper-tts 1.4+ は Python ホイールとして pip でインストール済みのため、バイナリの個別ダウンロードは不要です。

2回目以降はスキップされます。

> ⚠️ **Whisper モデルのダウンロードについて**
>
> デフォルトの `large-v3` は約 3GB あり、回線速度によっては 15 分以上かかります。
> 以下のいずれかで短縮できます。
>
> **方法 A — 軽量モデルに変更する（推奨）**
>
> `config.py` の 58 行目を編集してください：
>
> ```python
> # 変更前（デフォルト）
> model_size: str = "large-v3"        # 約 3.0GB
>
> # 推奨（品質をほぼ維持しつつ高速）
> model_size: str = "large-v3-turbo"  # 約 0.8GB・推論速度 4 倍
>
> # さらに軽量にする場合
> model_size: str = "medium"          # 約 0.5GB
> model_size: str = "small"           # 約 0.25GB（精度は低下）
> ```
>
> **方法 B — HuggingFace トークンを設定する**
>
> 未認証だとダウンロード速度が制限されます。無料アカウントのトークンを設定すると制限が緩和されます：
>
> 1. [huggingface.co](https://huggingface.co/) でアカウント作成 → Settings → Access Tokens → New token（read）
> 2. アプリ起動前に以下を実行：
>
> ```cmd
> set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx
> python main.py
> ```

---

## プロジェクト構造

```
ai_companion/
├── main.py              # エントリポイント・オーケストレーター
├── config.py            # 全設定の一元管理
├── installer.py         # 音声モデル自動インストール
├── ui/
│   └── app.py           # Tkinter ダークモード UI
├── audio/
│   ├── listener.py      # マイク入力・VAD 統合（TTS 再生中は自動ミュート）
│   ├── vad.py           # WebRTC VAD 発話検出
│   ├── transcriber.py   # faster-whisper 書き起こし
│   └── tts.py           # Piper TTS 合成・再生
├── ai/
│   ├── llm.py           # Ollama LLM 応答生成
│   ├── memory.py        # ChromaDB ベクトル記憶
│   └── embeddings.py    # nomic-embed-text 埋め込み
├── database/
│   └── storage.py       # SQLite チャット履歴
├── voices/
│   └── ja_female/       # 自動ダウンロード先
├── requirements.txt
└── README.md
```

---

## 機能概要

- **常時マイク待機**：WebRTC VAD による自動発話検出（ウェイクワード不要）
- **日本語音声認識**：faster-whisper large-v3（GPU/CPU 自動切替）
- **AI 応答**：Ollama qwen2.5:7b-instruct（温かく共感的な日本語応答）
- **長期記憶**：ChromaDB + nomic-embed-text（再起動後も記憶保持）
- **音声読み上げ**：Piper TTS 日本語女性音声（完全ローカル・espeak-ng 内蔵）
- **フィードバックループ防止**：TTS 再生中はマイクを自動ミュート（VAD バッファもリセット）
- **ダークモード UI**：リアルタイム書き起こしプレビュー・ステータス表示

---

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| `Ollama が起動していない` | `ollama serve` を先に実行 |
| `モデルが見つからない` | `ollama pull qwen2.5:7b-instruct-q4_K_M` を実行 |
| `マイクエラー` | サウンド設定でマイクのアクセス許可を確認 |
| `CUDA エラー` | CPU フォールバックで自動継続（速度は低下） |
| `piper-tts インストールエラー` | `pip install --upgrade piper-tts>=1.4.0` を実行（1.4+ で espeak-ng 内蔵、piper-phonemize 不要） |
| Whisper ダウンロードが遅い・止まる | `config.py` の `model_size` を `"large-v3-turbo"` に変更するか、`HF_TOKEN` を設定する（上記参照） |
| AI の声をマイクが拾って無限ループする | piper-tts 1.4+ + 自動ミュート対応済み。古いバージョンで発生する場合は `pip install --upgrade piper-tts>=1.4.0` |

ログは `companion.log` に出力されます。

---

## 深刻な悩みを抱えているときは

このアプリは AI による会話サポートを提供しますが、専門家の代替にはなりません。
深刻な悩みや危機的な状況では、カウンセラー・医師・公的相談機関などの専門家にご相談ください。
