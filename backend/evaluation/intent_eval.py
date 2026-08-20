"""Accuracy evaluation for agent intent classification.

Runs the labelled set in intent_dataset.py through classify_intent() against a real
model and reports what the routing decision actually got right.

Two accuracy numbers, because they answer different questions:

  label accuracy    — did it pick the exact intent? Useful for tuning the prompt.
  routing accuracy  — did it pick the right PATH (force a search vs allow a direct
                      answer)? This is what changes agent behaviour. Four labels
                      collapse into two routes, so a label can be wrong while the
                      agent still behaves correctly.

And one safety counter that matters more than either:

  grounding misses  — a message that needed the documents was allowed to answer
                      without them. This is the failure mode that produces confident,
                      unsourced answers, so it is reported separately and should be 0.

Usage:
    python -m backend.evaluation.intent_eval
    python -m backend.evaluation.intent_eval --provider groq --model openai/gpt-oss-20b
    python -m backend.evaluation.intent_eval --json results.json --delay 2.2

--delay paces requests for rate-limited free tiers (Groq allows ~30 req/min).
Requires a working provider key; nothing here runs in CI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from backend.evaluation.intent_dataset import (
    ALL_LABELS,
    CASES,
    TOOL_REQUIRING_LABELS,
    to_messages,
)

PASS_BAR = 0.95  # accuracy the classifier is expected to hold


def evaluate(llm, cases=None, delay: float = 0.0, progress=True) -> dict:
    """Classify every case and return metrics + per-case rows.

    llm: a chat model (built by the caller so this stays provider-agnostic).
    """
    from backend.agent.intent_classifier import classify_intent

    cases = list(cases if cases is not None else CASES)
    rows = []

    for i, case in enumerate(cases, 1):
        history = to_messages(case.history) if case.history else None
        t0 = time.time()
        res = classify_intent(case.message, conversation_history=history, llm=llm)
        ms = round((time.time() - t0) * 1000)

        correct = res.intent == case.expected
        rows.append({
            "message": case.message,
            "expected": case.expected,
            "predicted": res.intent,
            "correct": correct,
            "fallback": res.fallback,
            "requires_tools": res.requires_tools,
            "had_history": bool(case.history),
            "note": case.note,
            "ms": ms,
        })
        if progress:
            flag = "ok  " if correct else "MISS"
            print(f"{i:>3}. {flag} {case.expected:<18} -> {res.intent:<18} "
                  f"{ms:>5}ms  {case.message[:52]}", flush=True)
        if delay:
            time.sleep(delay)

    return _summarise(rows)


def _summarise(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(r["correct"] for r in rows)

    per_class = {}
    for label in ALL_LABELS:
        tp = sum(1 for r in rows if r["expected"] == label and r["predicted"] == label)
        fp = sum(1 for r in rows if r["expected"] != label and r["predicted"] == label)
        support = sum(1 for r in rows if r["expected"] == label)
        per_class[label] = {
            "support": support,
            "recall": tp / support if support else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "tp": tp, "fp": fp, "fn": support - tp,
        }

    matrix = {e: {p: 0 for p in ALL_LABELS} for e in ALL_LABELS}
    for r in rows:
        if r["predicted"] in matrix.get(r["expected"], {}):
            matrix[r["expected"]][r["predicted"]] += 1

    def needs_tools(label: str) -> bool:
        return label in TOOL_REQUIRING_LABELS

    routing_correct = sum(
        1 for r in rows if needs_tools(r["expected"]) == needs_tools(r["predicted"])
    )
    grounding_misses = [
        r for r in rows if needs_tools(r["expected"]) and not needs_tools(r["predicted"])
    ]

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "routing_accuracy": routing_correct / total if total else 0.0,
        "grounding_misses": len(grounding_misses),
        "grounding_miss_rows": grounding_misses,
        "fallbacks": sum(r["fallback"] for r in rows),
        "avg_ms": round(sum(r["ms"] for r in rows) / total) if total else 0,
        "per_class": per_class,
        "matrix": matrix,
        "labels": ALL_LABELS,
        "rows": rows,
    }


def print_report(s: dict) -> None:
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"label accuracy    : {s['correct']}/{s['total']} = {s['accuracy']:.1%}"
          f"   ({'PASS' if s['accuracy'] >= PASS_BAR else 'BELOW'} {PASS_BAR:.0%} bar)")
    print(f"routing accuracy  : {s['routing_accuracy']:.1%}   (search vs direct answer)")
    print(f"grounding misses  : {s['grounding_misses']}   "
          f"(needed documents, allowed a direct answer)")
    print(f"fallbacks         : {s['fallbacks']}   (classification failed -> forced search)")
    print(f"avg latency       : {s['avg_ms']}ms per turn")
    print(f"\n{'class':<20} {'recall':>8} {'precision':>10} {'n':>5}")
    for label in s["labels"]:
        c = s["per_class"][label]
        print(f"{label:<20} {c['recall']:>7.0%} {c['precision']:>10.0%} {c['support']:>5}")

    misses = [r for r in s["rows"] if not r["correct"]]
    if misses:
        print(f"\nmisclassified ({len(misses)}):")
        for r in misses:
            route = "SAME route" if (
                (r["expected"] in TOOL_REQUIRING_LABELS)
                == (r["predicted"] in TOOL_REQUIRING_LABELS)
            ) else "ROUTE CHANGED"
            print(f"  {r['expected']:<18} -> {r['predicted']:<18} [{route}]  {r['message']}")
            if r["note"]:
                print(f"        note: {r['note']}")
    print(bar)


def build_llm(provider: str | None, model: str | None, config_path: str):
    """Build the classifier model: explicit flags win, else the configured agent.intent."""
    from backend.core.config import load_config
    from backend.core.llm_client import get_llm_for

    config = load_config(config_path)
    if provider or model:
        key_env = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY",
                   "google": "GOOGLE_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
        section = {"provider": provider, "model": model, "temperature": 0}
        if provider in key_env:
            section["api_key"] = os.getenv(key_env[provider])
        return get_llm_for(config, section, max_tokens=12), f"{provider} / {model}"

    agent_cfg = (config.get("query") or {}).get("agent") or {}
    intent_cfg = agent_cfg.get("intent") or {}
    merged = {**agent_cfg, **intent_cfg}
    llm = get_llm_for(config, merged, max_tokens=intent_cfg.get("max_tokens", 12))
    return llm, f"{merged.get('provider')} / {merged.get('model')}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate agent intent classification.")
    p.add_argument("--provider", help="groq | openai | google | anthropic (default: from config)")
    p.add_argument("--model", help="model id (default: from config)")
    p.add_argument("--config", default=os.getenv("CONFIG_PATH", "config/global.yaml"))
    p.add_argument("--json", dest="json_out", help="write full results to this path")
    p.add_argument("--delay", type=float, default=0.0,
                   help="seconds between calls, for rate-limited free tiers (e.g. 2.2)")
    p.add_argument("--quiet", action="store_true", help="suppress per-case lines")
    args = p.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    llm, label = build_llm(args.provider, args.model, args.config)

    # Preflight, because classify_intent() catches everything and falls back to
    # document_question: an unusable model doesn't crash the run, it scores 75
    # fallbacks and still prints a routing number that looks survivable. One call up
    # front, built at the same max_tokens the real run uses, distinguishes the two
    # ways that happens — a retired id raises, a reasoning model returns empty.
    try:
        probe = llm.invoke("Reply with the single word: ok")
    except Exception as e:
        print(f"model check failed for {label}:\n  {e}\n\n"
              "Providers retire model ids without notice — confirm this one is still "
              "listed by the provider, or pass a different --model.", file=sys.stderr)
        return 2

    if not (probe.content or "").strip():
        print(f"{label} returned empty content under this token budget.\n\n"
              "Reasoning models (e.g. openai/gpt-oss-*) spend the whole budget on "
              "reasoning tokens before emitting any, so every case would score as a "
              "fallback. Raise intent.max_tokens well past a label's length, or pick a "
              "model that answers directly.", file=sys.stderr)
        return 2

    print(f"Evaluating {len(CASES)} cases against {label}\n")

    summary = evaluate(llm, delay=args.delay, progress=not args.quiet)
    summary["model"] = label
    print_report(summary)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.json_out}")

    # Non-zero exit if the bar is missed or grounding regressed — usable as a gate.
    ok = summary["accuracy"] >= PASS_BAR and summary["grounding_misses"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
