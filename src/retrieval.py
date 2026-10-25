"""Knowledge-base retrieval (Step 2: RAG, no external dependencies).

Every ``kb/*.md`` section keeps its ``[SECTION-ID]``. Chunks are scored with
TF-IDF cosine similarity over a small synonym-expanded vocabulary, then
re-ranked by the precedence in KB-00.1:

    current policy > agent guide > help > FAQ > marketing

Superseded documents (e.g. returns v3) are excluded from normal retrieval:
they only apply to electronics delivered before 1 August 2026 (RET-3), which
the agent handles explicitly. Support logs are never indexed -- they are not
policy authority. Marketing pages are indexed last and never used to answer
a policy question on their own.
"""
from __future__ import annotations

import glob
import math
import os
import re
from collections import Counter

# doc_type -> (precedence rank, score bonus). Lower rank wins ties.
PRECEDENCE = {
    "policy": (0, 0.18),
    "agent_guide": (1, 0.10),
    "help": (2, 0.05),
    "faq": (3, 0.0),
    "marketing": (4, -0.25),
    "meta": (5, -0.25),
}

STOPWORDS = set(
    "a an the is are was were be been being am do does did have has had "
    "i you he she it we they me him her us them my your his its our their "
    "this that these those what when where which who whom how why can could "
    "should would will shall may might please tell me about for to of in on "
    "at by with from as and or but if then so than too very just also".split()
)

# Customer words -> document words. Customers say "send it back", documents
# say "reverse pickup"; "nobody came to collect it" is a missed pickup, etc.
SYNONYMS = {
    "return": ["return", "send", "back", "reverse", "pickup"],
    "send": ["return", "reverse", "pickup"],
    "back": ["return", "reverse", "pickup"],
    "collect": ["reverse", "pickup"],
    "pickup": ["reverse", "pickup", "collection"],
    "cancel": ["cancel", "cancellation"],
    "refund": ["refund", "money", "back"],
    "money": ["refund"],
    "late": ["late", "delivery", "delay"],
    "delay": ["late", "delivery"],
    "broken": ["damaged", "defective", "crack"],
    "damaged": ["damaged", "defective"],
    "torn": ["damaged", "defective"],
    "wrong": ["wrong", "incorrect", "missing"],
    "missing": ["missing", "wrong"],
    "warranty": ["warranty", "assured"],
    "assured": ["assured", "warranty"],
    "charge": ["fee", "charges"],
    "fee": ["fee", "charges"],
    "free": ["free", "waived"],
    "coupon": ["coupon", "offer"],
    "discount": ["coupon", "discount"],
    "deliver": ["delivery", "shipping"],
    "delivery": ["delivery", "shipping"],
    "ship": ["shipping", "shipped", "delivery"],
    "track": ["tracking", "status", "shipped"],
    "status": ["status", "tracking"],
    "human": ["human", "colleague", "agent", "escalation"],
    "person": ["human", "colleague"],
    "speak": ["human", "colleague"],
    "complaint": ["escalation", "ticket"],
    "court": ["legal", "consumer", "escalation"],
    "police": ["legal", "escalation"],
    "battery": ["safety", "swelling", "battery"],
    "smell": ["safety", "burning"],
    "sparks": ["safety", "sparks"],
}


def tokenize(text: str):
    words = re.findall(r"[\w\u0900-\u097f]+", text.lower())
    out = []
    for word in words:
        if word in STOPWORDS:
            continue
        out.append(word)
        out.extend(SYNONYMS.get(word, []))
    return out


class Chunk:
    def __init__(self, section_id, doc_id, doc_type, title, text, superseded):
        self.section_id = section_id
        self.doc_id = doc_id
        self.doc_type = doc_type
        self.title = title
        self.text = text
        self.superseded = superseded
        self.tokens = Counter(tokenize(title + " " + text))
        self.norm = math.sqrt(sum(v * v for v in self.tokens.values())) or 1.0


def parse_kb(kb_dir: str):
    chunks = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.md"))):
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
        header, _, body = raw.partition("\n---\n")
        meta = {}
        for line in header.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        doc_type = meta.get("doc_type", "help")
        superseded = meta.get("status") == "superseded"
        current_id, current_title, current_lines = None, "", []
        for line in body.splitlines():
            m = re.match(r"##\s+\[([^\]]+)\]\s*(.*)", line)
            if m:
                if current_id:
                    chunks.append(Chunk(current_id, meta.get("doc_id", ""),
                                        doc_type, current_title,
                                        "\n".join(current_lines).strip(),
                                        superseded))
                current_id, current_title, current_lines = (
                    m.group(1), m.group(2), [])
            elif current_id:
                current_lines.append(line)
        if current_id:
            chunks.append(Chunk(current_id, meta.get("doc_id", ""),
                                doc_type, current_title,
                                "\n".join(current_lines).strip(), superseded))
    return chunks


class Retriever:
    def __init__(self, kb_dir: str):
        self.chunks = parse_kb(kb_dir)
        # Only current documents take part in normal retrieval.
        self.index = [c for c in self.chunks if not c.superseded]
        doc_count = len(self.index) or 1
        df = Counter()
        for chunk in self.index:
            for token in chunk.tokens:
                df[token] += 1
        self.idf = {
            token: math.log((doc_count + 1) / (count + 1)) + 1.0
            for token, count in df.items()
        }
        self.by_id = {c.section_id: c for c in self.chunks}

    def get(self, section_id: str):
        return self.by_id.get(section_id)

    def retrieve(self, query: str, top_k: int = 5):
        query_tokens = Counter(tokenize(query))
        q_norm = math.sqrt(sum(v * v for v in query_tokens.values())) or 1.0
        scored = []
        for chunk in self.index:
            dot = sum(
                count * chunk.tokens.get(token, 0) * self.idf.get(token, 1.0)
                for token, count in query_tokens.items()
            )
            cosine = dot / (q_norm * chunk.norm)
            rank, bonus = PRECEDENCE.get(chunk.doc_type, (2, 0.0))
            scored.append((cosine + bonus, -rank, chunk))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [
            {"section_id": c.section_id, "doc_type": c.doc_type,
             "title": c.title, "text": c.text, "score": round(score, 4)}
            for score, _, c in scored[:top_k]
        ]
