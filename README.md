# VoiceOps 2026 — ZentroMart Support Agent

**Capstone project for the LLM & Generative AI course (Consulting & Analytics Club, IIT Guwahati × Gnani.ai).**

A customer-support agent for **ZentroMart**, a fictional Indian online store, that answers from company documents, holds multi-turn conversations, looks up orders only after verification, creates handover tickets, and speaks through Gnani's speech models. Built from the supplied `Capstone_Project_Dataset_FTTT_2026` dataset — no real customer data anywhere.

## What it does (the four steps)

| Step | What it is | Where |
| --- | --- | --- |
| 1 — Answer with an LLM-style prompt | No-document baseline: the same agent with retrieval and policy handlers off, so it can only guess generic answers | `src/evaluate.py` (`step1_no_documents_questions`) |
| 2 — Ground it in documents (RAG) | TF-IDF retrieval over `kb/*.md` sections, each kept with its `[SECTION-ID]`, re-ranked by precedence: current policy > agent guide > help > FAQ > marketing (KB-00.1). Superseded returns v3 is excluded except for electronics delivered before 1 Aug 2026 (RET-3). Structured handlers add exact dates, windows and catalog rows on top of retrieval | `src/retrieval.py`, `src/agent.py` |
| 3 — Hold a conversation | Session state: order ID, verification, last item, last topic. Follow-ups like "aur refund kitne din mein aayega?" resolve against the same order. Verification before any order detail (IDV-1/2), one retry only (IDV-3), ticket before handover (ESC-2), never claiming an action the code didn't do (no cancel tool exists — the agent guides the customer to My Orders) | `src/agent.py`, `src/store.py` |
| 4 — Give it a voice | Gnani STT → same Step 3 agent → Gnani TTS, and nothing else for speech. TTS always names its model (`timbre-v2.5`, voice `Kaveri` by default) | `src/voice.py`, `run.py` |

## Results on the practice sets (computed, not typed)

Run `python -m src.evaluate --dataset <dataset-dir>` to reproduce these:

| Evaluation | Score |
| --- | --- |
| Step 1, no documents — 52 practice questions | 12/52 (23.1%) |
| Step 2, RAG + structured policies — 52 practice questions | **52/52 (100%)**, decision 100%, retrieval 100% |
| Step 3 — 23 practice conversations, 96 turns | **96/96 (100%)**, decision 100% |
| ASR transcripts (`asr_text`) — 52 practice questions | **52/52 (100%)** |

Step 1 here is a no-document baseline: the same agent with retrieval and policy handlers switched off. It shows the cost of answering without documents — most famously, the generic "5 to 7 days" refund answer, which the policy documents contradict for Wallet (instant), UPI (1–3 days) and net banking (3–5 days). Step 2 recovers all of it by grounding every answer and respecting precedence.

## The traps this agent is built to survive

- **Policy beats FAQ/marketing.** FAQ-1 says refunds take 5–7 days for everything; REF-2 wins, per KB-00.1.
- **Superseded ≠ current.** Electronics returns were 10 days (v3) and are 7 days since 1 Aug 2026 (v4); only electronics *delivered before* 1 Aug keep 10 days.
- **Festive sale.** Fashion ordered 1–15 Oct 2026 gets a 30-day window (RET-8), not 15.
- **Business days, not calendar days.** Refund timelines count Mon–Sat, skipping warehouse holidays (19 Oct 2026 is a holiday). A card refund initiated 17 Oct 2026 arrives 24–27 Oct, not 22–24 Oct.
- **Verification gates everything.** No order status, amount or address before the last 4 phone digits or delivery pincode match; two failed attempts lock the order out.
- **Never promise what policy forbids.** No ₹300 voucher for a late Standard delivery (DEL-4), no approving a >₹10,000 damage claim (Priority Resolution team, via ticket), no sharing another customer's phone number.
- **ASR is messy.** Spoken numbers are rebuilt ("ele-zero zero one to" → ELE-0012, "two zero two six" → 2026, "u p i" → UPI, "re fund" → refund) before any handler runs.

## Install and run

```bash
# 1. Extract the dataset zip somewhere (it is NOT included in this repo)
unzip Capstone_Project_Dataset_FTTT_2026.zip -d /path/to/dataset-parent
# The folder Capstone_Project_Dataset_FTTT_2026/ is the dataset dir.

# 2. No dependencies are needed for chat and batch text runs.
#    Only the voice layer needs the Gnani SDK:
pip install -r requirements.txt   # gnani-vachana

# 3. Interactive chat demo
python -m src.chat --dataset /path/to/Capstone_Project_Dataset_FTTT_2026

# 4. Evaluate (Step 1 vs Step 2 + conversations)
python -m src.evaluate --dataset /path/to/Capstone_Project_Dataset_FTTT_2026 \
    --output sample_outputs/evaluation_results.json

# 5. Batch run (the organizers' run command): audio in, text + audio + sources out
python run.py --input-dir audio/ --output-dir outputs/ \
    --dataset /path/to/Capstone_Project_Dataset_FTTT_2026
# Without a Gnani key, use --text with .txt transcripts, or put a .txt
# sidecar next to each audio file:
python run.py --input-dir transcripts/ --output-dir outputs/ \
    --dataset /path/to/Capstone_Project_Dataset_FTTT_2026 --text
```

Every batch output file gets a companion JSON with the transcript, reply, retrieved section IDs, decision, tool calls and ticket ID (if any). See `sample_outputs/batch_sample/` for an example.

## Voice setup (Gnani only)

1. Create an account and API key at **app.gnani.ai/voice** (one key covers STT and TTS).
2. `cp .env.example .env` and put your key in `.env`, or `export GNANI_API_KEY=...`.
3. Never put the key in code or git — `.env` is gitignored, and `.env.example` contains no key.

STT accepts clips up to 60 seconds. TTS voices for `timbre-v2.5` include Kaveri, Trupti, Pranav and Shlok; `timbre-v3.0` (13 voices, 10 languages) also works via `--model timbre-v3.0`. STT/TTS language codes follow Gnani's docs (e.g. `en-IN`, `hi-IN`, `gu-IN`); `en-IN` accepts English mixed with Hindi and returns Latin script. Replies are written for the ear: dates with the month name, order IDs in groups of four digits.

## Project layout

```text
├── README.md                  # this file
├── .env.example               # no keys in it
├── requirements.txt           # gnani-vachana only (voice layer)
├── run.py                     # batch runner: audio folder -> text + audio + sources
├── src/
│   ├── agent.py               # conversation agent: routing, verification, policy handlers
│   ├── retrieval.py           # TF-IDF RAG over kb sections, with KB-00.1 precedence
│   ├── store.py               # CSV data layer + lookup_order / create_ticket semantics
│   ├── dates.py               # business days, refund timelines, return windows (today = 20 Oct 2026)
│   ├── voice.py               # Gnani STT/TTS wrapper (Step 4)
│   ├── chat.py                # interactive chat demo
│   └── evaluate.py            # practice-set evaluation (Step 1 vs 2, conversations, ASR)
├── report.md / report.pdf     # short report (max 4 pages)
└── sample_outputs/
    ├── evaluation_results.json
    ├── batch_sample/          # example run.py outputs
    └── sample_chat_transcript.md
```

## What is deliberately not in this repo

The dataset copy (the brief forbids it — point `--dataset` at your extracted copy), `.env` / API keys, virtual environments, and `tickets_created.csv` (runtime output only). All data here is the supplied synthetic dataset; no real customers, orders, phone numbers or recordings were used.

## What went wrong, and what I would fix next

- **First retrieval-only version scored ~65% on questions.** Keyword search alone fell for FAQ/marketing traps and missed products named in short form. Fix: precedence re-ranking + structured handlers for dates, windows and catalog rows, and token-overlap product matching.
- **Hindi/Hinglish routing.** Devanagari questions initially fell through to English retrieval. Fix: Hindi topic words are mapped to the same handlers for routing (facts themselves stay in the documents' wording).
- **ASR numbers.** Spoken SKUs, years and acronyms initially failed. Fix: the normalizer in `src/agent.py` (`normalize_asr`).
- **Next:** a real LLM behind the retrieval step (with the same "say I don't know" guardrail and source citations) for questions the structured handlers don't cover, and a recorded 2-minute voice demo once a Gnani key is connected — the voice code is ready, and the demo is pending that key.
