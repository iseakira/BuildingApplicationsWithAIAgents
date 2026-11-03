"""
ステップ① 最初のエージェント（本 2.1節）

できること: 注文のキャンセルだけ。
課題:       返金ツール issue_refund を追加して、採点スクリプトの点数を上げる。

動かす:  .venv\\Scripts\\python -m my_agents.step01_cancel_agent
採点:    .venv\\Scripts\\python -m my_agents.eval_tool_calls my_agents.step01_cancel_agent
"""
import json
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from src.common.llm import get_llm


# ---------------------------------------------------------------------------
# 1. ツール: @tool を付けた関数。docstring が「このツールは何をするか」の説明として LLM に渡る
# ---------------------------------------------------------------------------
@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order that has not shipped yet."""
    print(f"  [TOOL] cancel_order(order_id={order_id})")
    return "cancelled"


@tool
def send_customer_message(order_id: str, text: str) -> str:
    """Send a plain-text message to the customer."""
    print(f"  [TOOL] send_customer_message -> {text}")
    return "sent"

@tool
def issue_refund(order_id:str, amount:float):
    """Issue a refund for an order that was already delivered or shipped"""
    print(f"""[TOOL] issue_refund(order_id={order_id}, amount={amount})""")
    return "refund_question"


# TODO(課題): ここに issue_refund(order_id: str, amount: float) を追加して、下の TOOLS に入れる


TOOLS = [cancel_order, issue_refund,send_customer_message]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# 2. LLM にツール一覧を渡す（bind_tools）。これで LLM が「どのツールを呼ぶか」を返せるようになる
llm = get_llm().bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# 3. 状態（State）: グラフの中を流れるデータ。messages は operator.add で「追記」される
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    order: dict
    messages: Annotated[Sequence[BaseMessage], operator.add]


SYSTEM_PROMPT = (
    "You are a helpful e-commerce support agent.\n"
    "First call the one business tool that fits the request, "
    "then call send_customer_message to confirm with the customer."
)


# ---------------------------------------------------------------------------
# 4. ノード: LLM を呼ぶ → ツールを実行 → 結果を見て LLM がもう一度返事をする
# ---------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    order_json = json.dumps(state.get("order") or {}, ensure_ascii=False)
    prompt = [SystemMessage(content=f"{SYSTEM_PROMPT}\n\nORDER: {order_json}")] + list(state["messages"])

    ai = llm.invoke(prompt)
    new_messages: list[BaseMessage] = [ai]

    for tc in ai.tool_calls:
        tool_fn = TOOLS_BY_NAME.get(tc["name"])
        result = tool_fn.invoke(tc["args"]) if tool_fn else f"error: unknown tool {tc['name']}"
        new_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"]))

    if ai.tool_calls:
        new_messages.append(llm.invoke(prompt + new_messages))

    return {"messages": new_messages}


# ---------------------------------------------------------------------------
# 5. グラフ: 今はノード1つだけ（START → agent → END）。5章で分岐を足していく
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.set_entry_point("agent")
    g.add_edge("agent", END)
    return g.compile()


graph = build_graph()


if __name__ == "__main__":
    examples = [
        ({"order_id": "B111", "status": "pending", "total": 50.0}, "Please cancel my order."),
        ({"order_id": "A222", "status": "delivered", "total": 19.99}, "My mug arrived broken. Refund?"),
    ]
    for order, text in examples:
        print(f"\n=== {text}  (order={order['order_id']}, status={order['status']})")
        result = graph.invoke({"order": order, "messages": [HumanMessage(content=text)]})
        print(f"  [AI] {result['messages'][-1].content}")
