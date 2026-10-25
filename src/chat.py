"""Interactive chat demo for the Step 3 agent.

Run:  python -m src.chat --dataset <path-to-extracted-dataset>
Type your messages; the agent introduces itself as ZentroMart's AI
assistant, asks for an order ID, verifies you, and then answers from the
documents. Type 'quit' to exit. Tickets are written to tickets_created.csv.
"""
from __future__ import annotations

import argparse
import sys

from .agent import ConversationAgent, Session


def main(argv=None):
    parser = argparse.ArgumentParser(description="ZentroMart support chat demo")
    parser.add_argument("--dataset", required=True,
                        help="Path to the extracted Capstone dataset folder")
    parser.add_argument("--tickets", default="tickets_created.csv")
    args = parser.parse_args(argv)
    agent = ConversationAgent(args.dataset, tickets_path=args.tickets)
    session = Session(mode="conversation")
    print("ZentroMart support (type 'quit' to exit).")
    while True:
        try:
            message = input("You: ").strip()
        except EOFError:
            break
        if message.lower() in ("quit", "exit"):
            break
        if not message:
            continue
        result = agent.respond(message, session)
        print(f"Agent: {result.text}")
        if result.sources:
            print(f"  [sources: {', '.join(result.sources)}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
