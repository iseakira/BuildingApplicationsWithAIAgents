"""
llm.py
どのLLMを使うかを1か所で切り替えるためのヘルパー。

    from src.common.llm import get_llm
    llm = get_llm().bind_tools(TOOLS)

切り替え方（環境変数 LLM_PROVIDER、または .env に書く）:
    fake      : APIキー不要のダミーLLM（キーワードでツールを選ぶだけ）
    openai    : ChatOpenAI（OPENAI_API_KEY が必要）
    anthropic : ChatAnthropic（ANTHROPIC_API_KEY が必要）
未指定なら、キーがあればそれを使い、なければ fake になる。
"""
from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# トレース送信は10章まで使わないので、明示的に有効化しない限りオフにする
os.environ.setdefault("TRACELOOP_TRACING_ENABLED", "false")


def get_llm(provider: Optional[str] = None, **kwargs: Any) -> BaseChatModel:
    provider = (provider or os.getenv("LLM_PROVIDER") or "").lower()
    if not provider:
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "fake"

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0.0, **kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"), temperature=0.0, **kwargs)
    if provider == "fake":
        print("[llm] No API key -> using FakeToolCallingModel (dummy LLM)")
        return FakeToolCallingModel()
    raise ValueError(f"unknown LLM_PROVIDER: {provider}")


# ---------------------------------------------------------------------------
# ダミーLLM
# ---------------------------------------------------------------------------

_REFUND_WORDS = ["refund", "broken", "cracked", "damaged", "返金", "壊れ", "割れ"]
_CANCEL_WORDS = ["cancel", "キャンセル"]
_ADDRESS_WORDS = ["address", "ship to", "住所", "届け先"]
_SHIPPED_STATUSES = {"delivered", "shipped"}


class FakeToolCallingModel(BaseChatModel):
    """本物のLLMの代わりに、キーワードで「それっぽい」ツール呼び出しを返すモデル。

    - 最後のメッセージが人間の発言なら → 業務ツール + send_customer_message を呼ぶ
    - 最後のメッセージがツール結果なら → 確認文を返して終わる
    賢さはゼロなので、評価の数字は本物のLLMとは比べられない。仕組みの確認用。
    """

    bound_tool_names: List[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools: list, **kwargs: Any) -> "FakeToolCallingModel":
        names = []
        for t in tools:
            name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
            names.append(name or getattr(t, "__name__", str(t)))
        return self.model_copy(update={"bound_tool_names": names})

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        last = messages[-1]
        if isinstance(last, ToolMessage):
            msg = AIMessage(content="Done. I've taken care of your request.")
        else:
            msg = self._decide(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # --- 中身 ---------------------------------------------------------------

    def _decide(self, messages: List[BaseMessage]) -> AIMessage:
        text = " ".join(str(m.content) for m in messages if isinstance(m, HumanMessage)).lower()
        order = self._find_order(messages)
        order_id = order.get("order_id", "UNKNOWN")
        status = str(order.get("status", "")).lower()

        def has(words):
            return any(w in text for w in words)

        if has(_ADDRESS_WORDS):
            call = ("update_address_for_order", {"order_id": order_id, "shipping_address": order.get("shipping_address", {})})
            reply = f"The shipping address for order {order_id} has been updated."
        elif has(_CANCEL_WORDS) and status not in _SHIPPED_STATUSES:
            call = ("cancel_order", {"order_id": order_id})
            reply = f"Your order {order_id} has been cancelled."
        elif has(_CANCEL_WORDS) or has(_REFUND_WORDS):
            call = ("issue_refund", {"order_id": order_id, "amount": float(order.get("total", 0.0))})
            reply = f"A refund for order {order_id} has been issued."
        else:
            call = None
            reply = "Could you tell me a bit more about what you need help with?"

        tool_calls = []
        if call and self._is_bound(call[0]):
            tool_calls.append(self._tool_call(*call))
        if self._is_bound("send_customer_message"):
            tool_calls.append(self._tool_call("send_customer_message", {"order_id": order_id, "text": reply}))
        return AIMessage(content="" if tool_calls else reply, tool_calls=tool_calls)

    def _is_bound(self, name: str) -> bool:
        return not self.bound_tool_names or name in self.bound_tool_names

    @staticmethod
    def _tool_call(name: str, args: dict) -> dict:
        return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}

    @staticmethod
    def _find_order(messages: List[BaseMessage]) -> dict:
        """システムプロンプト中の 'ORDER: {...}' から注文情報を取り出す。"""
        for m in messages:
            if isinstance(m, SystemMessage):
                match = re.search(r"ORDER:\s*(\{.*\})", str(m.content), re.S)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except json.JSONDecodeError:
                        pass
        return {}
