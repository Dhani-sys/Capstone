# VoiceOps 2026 — Short Report: ZentroMart Support Agent

## What I built

A support agent for ZentroMart, a fictional Indian online store, that answers customers from company documents, holds multi-turn conversations, and speaks through Gnani's speech models. The same agent serves chat, batch audio runs and the interactive demo: audio goes through Gnani STT into the agent, and the reply comes back through Gnani TTS. No other speech service is used, and no real customer data was used anywhere — every record is from the supplied synthetic dataset.

The agent is deterministic and dependency-free (Python standard library only, plus `gnani-vachana` for voice). Retrieval is TF-IDF over the knowledge-base sections, each kept with its `[SECTION-ID]`, re-ranked by the precedence in KB-00.1: current policy first, then agent guide, help, FAQ, and marketing last. Superseded documents are excluded from retrieval, and support logs are never treated as policy. On top of retrieval sit structured handlers that compute exact answers: business-day refund timelines (Monday–Saturday, skipping warehouse holidays, with "today" fixed at 20 October 2026), calendar-day return windows, festive-sale and pre-August electronics rules, catalog lookups by SKU or product name, and order lookups gated by verification.

Conversation state is kept per session: order ID, verification status, last item and last topic. A customer is asked for their order ID, then for the last 4 digits of their registered phone number or their delivery pincode, and no order detail is shared until that check passes. Two failed attempts lock the order out. Follow-ups resolve against the same order ("aur refund kitne din mein aayega?" after a return question uses that order's payment method and Plus status). Tickets are created before any handover, and a handover is only claimed when the ticket tool returns `created=True`. The agent never claims to have cancelled an order — no cancel tool exists, so it guides the customer to cancel free in My Orders — and never promises compensation, approvals or outcomes the policies do not allow.

## Step 1 vs Step 2 scores

| Evaluation | Result |
| --- | --- |
| Step 1, no documents — 52 practice questions | 12/52 (23.1%) |
| Step 2, RAG + structured policies — 52 practice questions | 52/52 (100%), retrieval 100% |
| Step 3 — 23 practice conversations, 96 turns | 96/96 (100%) |
| ASR transcripts — 52 practice questions | 52/52 (100%) |

Step 1 is the same agent with retrieval and policy handlers switched off — a no-document baseline. Its value is that it measures exactly what grounding is worth. Without documents it guesses the generic answer, and the dataset is designed to punish that: FAQ-1's "refunds take 5 to 7 days" is wrong for Wallet (instant, within 2 hours), UPI (1 to 3 business days) and net banking (3 to 5 business days); the old 10-day electronics window is wrong since 1 August 2026; and the festive 30-day fashion window is invisible without RET-8. Step 2 recovers all of it, because every answer is tied to a retrieved section or a computed date, and unanswerable questions get an honest "I don't have information about that" plus a human handover instead of an invented policy.

## What went wrong and what I would fix

Three failures shaped the final design. First, plain keyword retrieval scored about 65% on the questions: it fell for FAQ and marketing traps and missed products named in short form ("Kaafi Running Sneakers" versus the catalog's "Kaafi Running Sneakers, White"). The fix was precedence re-ranking plus structured handlers and token-overlap product matching. Second, Hindi and Hinglish questions initially fell through to English retrieval; mapping Hindi topic words to the same handlers for routing fixed routing without mistranslating policy facts. Third, ASR transcripts broke on spoken numbers ("ele-zero zero one to", "two zero two six", "u p i", "re fund"); a normalizer that rebuilds digits, years, SKUs and acronyms before routing brought ASR accuracy to 100% on the practice transcripts.

What I would fix next: put a real LLM behind retrieval for questions the structured handlers do not cover, keeping the same guardrails — answer only from retrieved sections, cite them, and say "I don't know" when they do not cover the question. And record the 2-minute voice demo: the voice code is complete and follows the brief's sample (Gnani SDK, `timbre-v2.5`, voice Kaveri), and the demo is pending a Gnani key. Connecting a key at `app.gnani.ai/voice` and setting `GNANI_API_KEY` is the only remaining step for Step 4 to run end to end.

## How to run it

```bash
unzip Capstone_Project_Dataset_FTTT_2026.zip -d /path/to/dataset-parent
pip install -r requirements.txt   # gnani-vachana, voice only
python -m src.chat --dataset /path/to/Capstone_Project_Dataset_FTTT_2026
python -m src.evaluate --dataset /path/to/Capstone_Project_Dataset_FTTT_2026
python run.py --input-dir audio/ --output-dir outputs/ \
    --dataset /path/to/Capstone_Project_Dataset_FTTT_2026
```

The repository contains no dataset copy, no API keys and no `.env` file, as the brief requires. Full evaluation details are in `sample_outputs/evaluation_results.json`, with a sample chat transcript and batch outputs alongside it.
