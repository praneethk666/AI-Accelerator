"""Drive the agent-executor from the terminal — a fast way to see the tool-picking
loop work without the API/UI. One-shot or interactive.

    python scripts/agent_chat.py "what does the warranty policy say about returns?"
    python scripts/agent_chat.py "ingest ./uploads/some.pdf"
    python scripts/agent_chat.py                     # interactive REPL, keeps memory

On a pending write (e.g. ingest_document), it prints what the agent wants to run
and asks y/n right there before re-invoking with approval — same shape the API/UI
will use, just done inline.

Thin CLI over backend.agent.executor.run_agent — no recipe duplicated here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from backend.agent.executor import run_agent  # noqa: E402
from backend.agent_tools import build_agent_registry  # noqa: E402
from backend.core.config import load_config  # noqa: E402

CONFIG_PATH = os.getenv("CONFIG_PATH", "config/global.yaml")


def _print_tool_calls(tool_calls: list[dict]) -> None:
    for call in tool_calls:
        result = call.get("result", "<blocked>")
        preview = result if len(str(result)) <= 300 else f"{str(result)[:300]}..."
        print(f"  → {call['name']}({call['args']})\n    = {preview}")


def ask(message: str, config: dict, registry: dict, history: list) -> list:
    """Run one turn, handling an approval prompt inline. Returns the updated history."""
    result = run_agent(message, config=config, registry=registry, conversation_history=history)

    if result["status"] == "needs_approval":
        if result["tool_calls"]:
            print("(tool calls this turn:)")
            _print_tool_calls(result["tool_calls"])
        for p in result["pending"]:
            print(f"\nagent wants to run: {p['name']}({p['args']})")
        reply = input("approve? [y/N] ").strip().lower()
        if reply != "y":
            print("declined — not running it.")
            return history
        result = run_agent(message, config=config, registry=registry,
                            conversation_history=history, approved_writes=True)

    if result["tool_calls"]:
        print("(tools called:)")
        _print_tool_calls(result["tool_calls"])
    print(f"\nagent: {result['answer']}\n")
    return result["messages"]


def main() -> None:
    config = load_config(CONFIG_PATH)
    registry = build_agent_registry()
    print(f"tools available: {', '.join(registry)}\n")

    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]), config, registry, [])
        return

    print("interactive mode — Ctrl+C to quit")
    history: list = []
    while True:
        try:
            message = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        history = ask(message, config, registry, history)


if __name__ == "__main__":
    main()
