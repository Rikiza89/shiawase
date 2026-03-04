"""
Ollama LLM 呼び出しモジュール。
心理的サポートシステムプロンプトと記憶コンテキストを統合する。
"""
from __future__ import annotations
import logging
from typing import List
import requests
from ai.memory import MemoryEntry
from config import OllamaConfig
logger = logging.getLogger(__name__)
# ── 心理的サポート用システムプロンプト（ハードコード必須仕様） ──────
_SYSTEM_PROMPT = """You are a warm, gentle, and supportive emotional companion.
[Absolute Rules]
- Always respond in BOTH English and Japanese using this exact format:
  [EN] <English response here>
  [JA] <日本語の応答はここ>
- Never skip either language tag
- Maintain a warm, calm, and empathetic tone at all times
- Never blame or shame the user
- Gently reframe negative thoughts in a positive, constructive way
- Encourage self-compassion
- Keep each language response short to medium length (not too long)
- Speak naturally and conversationally
[If the user mentions suicide or self-harm]
- Do not abruptly reject or cut off the conversation
- Respond with empathy and let them know they are not alone
- Gently encourage them to seek professional support
- Recommend speaking with a counselor, doctor, or support service (do not provide specific phone numbers or URLs)
[Tone Guidelines]
- Calm, soft, and warm
- Non-imposing
- Non-judgmental
- Collaborative and understanding
"""
class LLMError(Exception):
    """LLM 呼び出しの失敗を表す例外。"""
class LLMService:
    """
    Ollama qwen2.5 モデルへの生成リクエストを管理するサービス。
    Args:
        cfg: Ollama 接続設定。
    """
    def __init__(self, cfg: OllamaConfig) -> None:
        self._cfg = cfg
        self._endpoint = f"{cfg.base_url}/api/chat"
    def generate(
        self,
        user_input: str,
        recent_history: List[dict],
        memories: List[MemoryEntry],
    ) -> str:
        """
        ユーザー入力・会話履歴・記憶コンテキストから応答を生成する。
        Args:
            user_input: 最新のユーザー発言。
            recent_history: 直近の会話履歴（{"role": ..., "content": ...} 形式）。
            memories: 意味的に類似した過去の記憶エントリ。
        Returns:
            生成された日本語応答テキスト。
        Raises:
            LLMError: API 呼び出し失敗時。
        """
        # 関連記憶をシステムプロンプトに注入
        system = _SYSTEM_PROMPT
        if memories:
            mem_text = "\n".join(
                f"- [{m.metadata.get('role', '?')}] {m.document}" for m in memories
            )
            system += f"\n\n【関連する過去の会話（参考）】\n{mem_text}"
        messages = [{"role": "system", "content": system}]
        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_input})
        payload = {
            "model": self._cfg.llm_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._cfg.temperature,
                "top_p": self._cfg.top_p,
                "num_predict": self._cfg.max_tokens,
            },
        }
        try:
            resp = requests.post(
                self._endpoint,
                json=payload,
                timeout=self._cfg.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"Ollama LLM API エラー: {e}") from e
        data = resp.json()
        content = data.get("message", {}).get("content", "").strip()
        if not content:
            raise LLMError(f"LLM 応答が空です: {data}")
        logger.debug("LLM 応答生成完了: len=%d", len(content))
        return content
    def check_availability(self) -> bool:
        """
        Ollama サーバーと対象モデルの稼働確認。
        Returns:
            利用可能なら True。
        """
        try:
            resp = requests.get(
                f"{self._cfg.base_url}/api/tags",
                timeout=5,
            )
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            available = any(self._cfg.llm_model in m for m in models)
            if not available:
                logger.warning(
                    "モデル '%s' が Ollama に見つかりません。利用可能: %s",
                    self._cfg.llm_model,
                    models,
                )
            return available
        except requests.RequestException:
            return False

