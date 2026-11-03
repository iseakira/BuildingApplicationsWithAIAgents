"""
採点スクリプト: 評価セットを流して「正しい業務ツールが呼ばれたか」を数える。

    .venv\\Scripts\\python -m my_agents.eval_tool_calls my_agents.step01_cancel_agent
    .venv\\Scripts\\python -m my_agents.eval_tool_calls my_agents.step01_cancel_agent --limit 5 -v

対象のモジュールは `graph` を持ち、graph.invoke({"order": ..., "messages": [...]}) で
動くこと（ステップ①のエージェントと同じ形）。
"""
import argparse
import contextlib
import importlib
import io
import json
from collections import defaultdict
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

EVAL_SET = (
    Path(__file__).resolve().parents[1]
    / "src/common/evaluation/scenarios/ecommerce_customer_support_evaluation_set.jsonl"
)
# 返信用ツールは「業務ツール」ではないので採点から外す
IGNORE = {"send_customer_message"}


def load_samples(limit=None):
    with open(EVAL_SET, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    return samples[:limit] if limit else samples


def to_messages(turns):
    return [HumanMessage(content=t["content"]) if t["role"] == "customer" else AIMessage(content=t["content"])
            for t in turns]


def called_tools(messages):
    names = []
    for m in messages:
        if isinstance(m, AIMessage):
            names += [tc["name"] for tc in m.tool_calls if tc["name"] not in IGNORE]
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_module", help="例: my_agents.step01_cancel_agent")
    parser.add_argument("--limit", type=int, help="先頭から何件だけ流すか")
    parser.add_argument("-v", "--verbose", action="store_true", help="エージェントの print も表示する")
    args = parser.parse_args()

    graph = importlib.import_module(args.agent_module).graph
    per_tool = defaultdict(lambda: [0, 0])  # expected -> [正解数, 件数]

    for s in load_samples(args.limit):
        expected = s["expected_function_call"]["name"]
        quiet = contextlib.nullcontext() if args.verbose else contextlib.redirect_stdout(io.StringIO())
        with quiet:
            result = graph.invoke({"order": s["order"], "messages": to_messages(s["input"])})
        got = called_tools(result["messages"])
        ok = expected in got
        per_tool[expected][0] += ok
        per_tool[expected][1] += 1
        print(f"{'OK ' if ok else 'NG '} {s['sample_id']:<22} expected={expected:<15} got={got}")

    print("\n--- ツール別の正解率 ---")
    total_ok = total = 0
    for name, (ok, n) in sorted(per_tool.items()):
        print(f"  {name:<15} {ok}/{n}")
        total_ok += ok
        total += n
    print(f"  {'合計':<14} {total_ok}/{total}  ({total_ok / total:.0%})")


if __name__ == "__main__":
    main()
