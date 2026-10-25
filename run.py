"""Batch runner -- the organizers' run command (see README).

Takes a folder of customer audio files, runs the Step 3 agent on each, and
saves for every file:
    <stem>.txt   -- the agent's reply text
    <stem>.wav   -- the spoken reply (Gnani TTS, when a key is available)
    <stem>.json  -- transcript, reply, retrieved section IDs, decision,
                    tool calls and ticket ID (if any)

Usage:
    python run.py --input-dir audio/ --output-dir outputs/ --dataset <dataset-dir>
    python run.py --input-dir transcripts/ --output-dir outputs/ --dataset <dir> --text

--text mode reads .txt transcripts instead of audio, for testing the agent
without a Gnani key. In audio mode, if Gnani STT is unavailable, a sidecar
<stem>.txt transcript is used when present; otherwise the reply is the
"didn't catch that" recovery line, so every input file still gets an output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent import ConversationAgent  # noqa: E402

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="VoiceOps batch runner")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True,
                        help="Path to the extracted Capstone dataset folder")
    parser.add_argument("--text", action="store_true",
                        help="Treat input .txt files as transcripts (no STT)")
    parser.add_argument("--language", default="en-IN")
    parser.add_argument("--voice", default="Kaveri")
    parser.add_argument("--model", default="timbre-v2.5")
    parser.add_argument("--tickets", default="tickets_created.csv")
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    agent = ConversationAgent(args.dataset, tickets_path=args.tickets)

    voice = None
    if not args.text and os.environ.get("GNANI_API_KEY"):
        try:
            from src.voice import GnaniVoice
            voice = GnaniVoice(language=args.language, voice=args.voice,
                               model=args.model)
        except RuntimeError as exc:
            print(f"Voice unavailable, continuing with text only: {exc}",
                  file=sys.stderr)

    files = sorted(os.listdir(args.input_dir))
    summary = []
    for filename in files:
        stem, ext = os.path.splitext(filename)
        path = os.path.join(args.input_dir, filename)
        if not os.path.isfile(path):
            continue
        if args.text:
            if ext.lower() != ".txt":
                continue
            with open(path, encoding="utf-8") as handle:
                transcript = handle.read().strip()
            result = agent.answer_single(transcript) if transcript else None
            reply = result.text if result else "Sorry, I didn't catch that."
        else:
            if ext.lower() not in AUDIO_EXTS:
                continue
            transcript = ""
            if voice is not None:
                try:
                    transcript = voice.transcribe(path)
                except Exception as exc:  # noqa: BLE001 - keep going per file
                    print(f"STT failed for {filename}: {exc}", file=sys.stderr)
            if not transcript:
                sidecar = os.path.join(args.input_dir, stem + ".txt")
                if os.path.exists(sidecar):
                    with open(sidecar, encoding="utf-8") as handle:
                        transcript = handle.read().strip()
            if transcript:
                result = agent.answer_single(transcript)
                reply = result.text
            else:
                result = None
                reply = "Sorry, I didn't catch that. Could you say it again?"
        out_base = os.path.join(args.output_dir, stem)
        with open(out_base + ".txt", "w", encoding="utf-8") as handle:
            handle.write(reply)
        record = {
            "file": filename, "transcript": transcript, "reply": reply,
            "decision": result.decision if result else "clarify",
            "sources": result.sources if result else [],
            "tool_calls": result.tool_calls if result else [],
            "ticket_id": result.ticket_id if result else None,
        }
        with open(out_base + ".json", "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
        if voice is not None:
            try:
                voice.synthesize(reply, out_base + ".wav")
                record["audio"] = stem + ".wav"
            except Exception as exc:  # noqa: BLE001
                print(f"TTS failed for {filename}: {exc}", file=sys.stderr)
        summary.append(record)
        print(f"{filename}: {reply[:100]}")
    with open(os.path.join(args.output_dir, "summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Done: {len(summary)} files -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
