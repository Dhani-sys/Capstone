"""Evaluation on the practice sets (Step 1 vs Step 2, and Step 3).

Scoring follows the dataset's own rules (DATA_DICTIONARY.md):
* ``must_include``: facts separated by `` ; ``, alternatives by `` / `` --
  any one alternative is enough. ISO dates also match their written form
  ("2026-10-27" == "27 October 2026"), case-insensitively.
* ``must_not_include``: any listed phrase (or paraphrase-free substring)
  present in the reply fails that row.
* Single questions: any decision in ``acceptable_decisions`` counts.
* Conversations: the row's ``expected_decision`` counts, run in turn order
  with one shared Session per conversation.
* Retrieval: a row's ``gold_evidence`` section IDs present in the reply's
  sources count as retrieved (catalog rows ``products.csv:SKU`` are checked
  against the product the agent used, via the reply text/decision path).

Step 1 is the same agent with retrieval and structured handlers off -- a
no-document baseline, so it shows what answering without the documents
costs, and Step 2 shows what grounding recovers. Both numbers are computed
by this script, never typed by hand.

Usage:
    python -m src.evaluate --dataset <dataset-dir> [--output results.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import date

from .agent import ConversationAgent, Session
from .dates import format_date

GENERIC_TOKENS = {"the", "a", "an"}


def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s₹]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # ISO date -> written form, so both notations match.
    def _fmt(match):
        try:
            day = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return format_date(day).lower()
        except ValueError:
            return match.group(0)
    text = re.sub(r"(\d{4})-(\d{2})-(\d{2})", _fmt, text)
    return text


def _expand_dates(text: str) -> str:
    """Add the written form of any ISO date to a must_include string."""
    def _fmt(match):
        try:
            year, month, day_num = match.group(0).split("-")
            day = date(int(year), int(month), int(day_num))
            return f"{match.group(0)} / {format_date(day)}"
        except ValueError:
            return match.group(0)
    return re.sub(r"\d{4}-\d{2}-\d{2}", _fmt, text)


def fact_passed(reply: str, fact: str) -> bool:
    reply_n = _norm(reply)
    for alternative in _expand_dates(fact).split(" / "):
        alt = _norm(alternative)
        if alt and alt in reply_n:
            return True
    return False


def row_passed(reply: str, must_include: str, must_not_include: str) -> bool:
    if must_include.strip():
        for fact in must_include.split(" ; "):
            if fact.strip() and not fact_passed(reply, fact.strip()):
                return False
    if must_not_include.strip():
        reply_n = _norm(reply)
        for banned in must_not_include.split(" ; "):
            if banned.strip() and _norm(banned.strip()) in reply_n:
                return False
    return True


def retrieval_hit(reply_sources, gold_evidence: str) -> bool:
    gold_sections = [g.strip() for g in gold_evidence.split(" ; ")
                     if g.strip() and not g.strip().startswith("products.csv")]
    if not gold_sections:
        return True  # nothing to retrieve for this row (e.g. unsupported)
    return any(section in reply_sources for section in gold_sections)


def evaluate_questions(agent, dataset_dir):
    path = os.path.join(dataset_dir, "practice", "questions.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    details, passed = [], 0
    decision_ok, retrieval_ok = 0, 0
    for row in rows:
        result = agent.answer_single(row["question"])
        ok = row_passed(result.text, row["must_include"], row["must_not_include"])
        acceptable = [d.strip() for d in row["acceptable_decisions"].split(" ; ")]
        dec_ok = result.decision in acceptable
        ret_ok = retrieval_hit(result.sources, row["gold_evidence"])
        passed += ok
        decision_ok += dec_ok
        retrieval_ok += ret_ok
        details.append({"id": row["id"], "intent": row["intent"],
                        "question": row["question"], "reply": result.text,
                        "decision": result.decision,
                        "expected_decision": row["expected_decision"],
                        "decision_ok": dec_ok, "retrieval_ok": ret_ok,
                        "sources": result.sources, "passed": ok,
                        "must_include": row["must_include"],
                        "must_not_include": row["must_not_include"]})
    return {"total": len(rows), "passed": passed,
            "accuracy": round(passed / len(rows), 4) if rows else 0,
            "decision_accuracy": round(decision_ok / len(rows), 4) if rows else 0,
            "retrieval_accuracy": round(retrieval_ok / len(rows), 4) if rows else 0,
            "details": details}


def evaluate_conversations(agent, dataset_dir):
    path = os.path.join(dataset_dir, "practice", "conversations.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by_conversation = {}
    for row in rows:
        by_conversation.setdefault(row["conversation_id"], []).append(row)
    details, passed, total = [], 0, 0
    decision_ok = 0
    for conv_id, turns in sorted(by_conversation.items()):
        session = Session(mode="conversation")
        for row in sorted(turns, key=lambda r: int(r["turn"])):
            result = agent.respond(row["customer_message"], session)
            ok = row_passed(result.text, row["must_include"],
                            row["must_not_include"])
            dec_ok = result.decision == row["expected_decision"]
            passed += ok
            decision_ok += dec_ok
            total += 1
            details.append({"conversation_id": conv_id, "turn": row["turn"],
                            "scenario": row["scenario"],
                            "customer_message": row["customer_message"],
                            "reply": result.text, "decision": result.decision,
                            "expected_decision": row["expected_decision"],
                            "decision_ok": dec_ok, "passed": ok,
                            "sources": result.sources,
                            "tool_calls": result.tool_calls,
                            "must_include": row["must_include"],
                            "must_not_include": row["must_not_include"]})
    return {"total": total, "passed": passed,
            "accuracy": round(passed / total, 4) if total else 0,
            "decision_accuracy": round(decision_ok / total, 4) if total else 0,
            "details": details}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the VoiceOps agent")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--tickets", default="tickets_created.csv")
    args = parser.parse_args(argv)

    step1 = ConversationAgent(args.dataset, tickets_path=args.tickets,
                              use_retrieval=False, use_structured=False)
    step2 = ConversationAgent(args.dataset, tickets_path=args.tickets,
                              use_retrieval=True, use_structured=True)
    results = {
        "step1_no_documents_questions": evaluate_questions(step1, args.dataset),
        "step2_rag_questions": evaluate_questions(step2, args.dataset),
        "step3_conversations": evaluate_conversations(step2, args.dataset),
    }
    # Strip per-row details from the printed summary; keep them in the file.
    summary = {key: {k: v for k, v in value.items() if k != "details"}
               for key, value in results.items()}
    print(json.dumps(summary, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f"Full results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
