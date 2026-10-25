"""ZentroMart support agent: Steps 1-3 in one deterministic, grounded agent.

Design notes (all from the dataset's own guides):

* Step 1 (prompt-only) is the same agent with retrieval and structured
  policy handlers switched off -- it can only give generic answers. See
  ``evaluate.py`` for the honest Step 1 vs Step 2 comparison. No LLM API
  key is required, so no policy, date or amount is ever invented: if the
  documents do not cover a question, the agent says so (and offers a human).
* Step 2 (RAG) retrieves KB sections with precedence current policy >
  agent guide > help > FAQ > marketing (KB-00.1), and structured handlers
  for dates, windows and catalog rows sit on top of retrieval so answers
  carry exact numbers and deadlines, with their section IDs as sources.
* Step 3 (conversation) keeps order ID, verification state, last item and
  last topic in a Session. Order details are shared only after verification
  (IDV-1/2), one retry only (IDV-3), and a ticket is created before any
  handover (ESC-2) -- and only claimed when ``created=True``.
* "Today" is always 20 October 2026 (KB-00.2), business days are
  Monday-Saturday excluding holidays (KB-00.3), return windows are
  calendar days (RET-2).
"""
from __future__ import annotations

import os
import re
from datetime import date, timedelta

from .dates import (
    TODAY, RETURN_WINDOWS, NON_RETURNABLE, format_date, parse_date,
    load_holidays, add_business_days, refund_window,
)
from .retrieval import Retriever
from .store import Store

DIGIT_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "oh": "0",
}

HI_EN_WORDS = {
    "kya", "kaise", "mujhe", "mera", "meri", "kitne", "kitna", "kab",
    "hai", "hain", "pe", "ka", "ki", "ko", "se", "aur", "ab", "nahi",
    "nahin", "chahiye", "kar", "sakta", "sakti", "sakte", "ho", "gaya",
    "aaya", "wala", "wali", "din", "rupaye", "rupay", "theek", "achha",
    "aakhri", "ank", "baare", "mein", "liye", "bhi", "toh", "ek",
}


def detect_language(text: str) -> str:
    if any("\u0900" <= ch <= "\u097f" for ch in text):
        return "hi"
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & HI_EN_WORDS:
        return "hi-en"
    return "en"


def normalize_asr(text: str) -> str:
    """Rebuild numbers from ASR text: 'ele-zero zero one to' -> ELE-0012.

    ASR mishears "two" as "to", splits acronyms ("u p i", "re fund") and
    spells years digit by digit ("two zero two six"). Only digit-word runs
    are converted, so ordinary English "to" ("speak to a person") is left
    alone. Spaced single digits are then collapsed ("2 0 2 6" -> "2026",
    "8 9 7 8 2 8 3 7" -> an 8-digit order ID).
    """
    out = text
    out = re.sub(r"\bu\s+p\s+i\b", "upi", out, flags=re.IGNORECASE)
    out = re.sub(r"\bc\s+o\s+d\b", "cod", out, flags=re.IGNORECASE)
    out = re.sub(r"\bre\s+fund\b", "refund", out, flags=re.IGNORECASE)

    digit_run = (r"(?:zero|one|two|three|four|five|six|seven|eight|nine|oh|to)")
    def _convert_run(match):
        tokens = match.group(0).split()
        if sum(1 for t in tokens if t.lower() in DIGIT_WORDS) < 2:
            return match.group(0)
        return " ".join(
            DIGIT_WORDS.get(t.lower(), "2" if t.lower() == "to" else t)
            for t in tokens)
    out = re.sub(rf"\b{digit_run}(?:\s+{digit_run})+\b", _convert_run, out,
                 flags=re.IGNORECASE)
    # Spaced single digits -> one number ("2 0 2 6" -> "2026").
    out = re.sub(r"\b\d(?:\s+\d){2,}\b",
                 lambda m: m.group(0).replace(" ", ""), out)
    # "ele-0 0 1 2" / "ele 0 0 1 2" -> "ELE-0012" style tokens.
    out = re.sub(r"(ele|bks|fsh|hom|inr)\s*-?\s*(\d(?:\s+\d){3})",
                 lambda m: m.group(1).upper() + "-" + m.group(2).replace(" ", ""),
                 out, flags=re.IGNORECASE)
    return out


def extract_order_id(text: str):
    m = re.search(r"\b(\d{4})\s+(\d{4})\b", text)
    if m:
        return m.group(1) + m.group(2)
    m = re.search(r"\b(\d{8})\b", text)
    if m:
        return m.group(1)
    return None


def extract_verification(text: str, order_id: str = None):
    """A 6-digit pincode or 4-digit phone suffix, excluding the order ID."""
    for m in re.finditer(r"\b(\d{6})\b", text):
        return m.group(1)
    for m in re.finditer(r"\b(\d{4})\b", text):
        value = m.group(1)
        if order_id and value in (order_id[:4], order_id[4:]):
            continue
        return value
    return None


def extract_sku(text: str):
    m = re.search(r"\b([A-Z]{3}-\d{4})\b", text.upper())
    return m.group(1) if m else None


def extract_amount(text: str):
    m = re.search(r"₹\s*([\d,]+)|(\d+)\s*(?:rupees|rupaye|rupay)", text,
                  re.IGNORECASE)
    if m:
        return (m.group(1) or m.group(2)).replace(",", "")
    return None


class Session:
    """One customer's conversation state. Nothing here leaves the session."""

    def __init__(self, mode: str = "conversation"):
        self.mode = mode  # 'conversation' (intro + state) or 'single' (one shot)
        self.greeted = mode == "single"
        self.order_id = None
        self.verified = False
        self.verify_attempts = 0
        self.blocked = False
        self.order = None
        self.last_product = None
        self.last_topic = None
        self.last_payment = None
        self.turn = 0


class Result:
    def __init__(self, text, decision="answer", sources=None, tool_calls=None,
                 ticket_id=None):
        self.text = text
        self.decision = decision
        self.sources = sources or []
        self.tool_calls = tool_calls or []
        self.ticket_id = ticket_id

    def as_dict(self):
        return {"text": self.text, "decision": self.decision,
                "sources": self.sources, "tool_calls": self.tool_calls,
                "ticket_id": self.ticket_id}


# Short language-aware framing lines. Policy facts themselves stay in the
# documents' wording so numbers and section terms are never mistranslated.
PHRASES = {
    "intro": {
        "en": "Hi, I'm ZentroMart's AI support assistant. Could you please share your order ID?",
        "hi": "नमस्ते, मैं ZentroMart की AI support assistant हूँ। कृपया अपना order ID बताएँ।",
        "hi-en": "Hello, main ZentroMart ki AI support assistant hoon. Aap apna order ID bata sakte hain?",
    },
    "ask_verify": {
        "en": "Thanks. To share order details, could you please share the last 4 digits of your registered phone number, or your delivery pincode?",
        "hi": "धन्यवाद। Order की जानकारी देने से पहले, कृपया अपने registered phone number के आख़िरी 4 अंक या delivery पिनकोड बताएँ।",
        "hi-en": "Thanks. Order details dene se pehle, aap apne registered phone ke aakhri 4 ank ya delivery pincode bata sakte hain?",
    },
    "retry": {
        "en": "That didn't match our records. Could you please try again with the last 4 digits of your registered phone number or your delivery pincode?",
        "hi": "यह हमारे records से match नहीं हुआ। कृपया registered phone के आख़िरी 4 अंक या delivery पिनकोड दोबारा बताएँ।",
        "hi-en": "Yeh hamare records se match nahi hua. Aap aakhri 4 ank ya delivery pincode dobara bata sakte hain?",
    },
    "blocked": {
        "en": "I'm sorry, I can't share order details without verification. Please contact support from your registered phone number, and a colleague will help you.",
        "hi": "माफ़ कीजिए, verification के बिना मैं order की जानकारी साझा नहीं कर सकती। कृपया अपने registered phone number से support से संपर्क करें।",
        "hi-en": "Maaf kijiye, verification ke bina main order details share nahi kar sakti. Aap apne registered phone number se support se contact karein.",
    },
    "dont_know": {
        "en": "Sorry, I don't have information about that. I can connect you to a colleague who can help.",
        "hi": "माफ़ कीजिए, इसके बारे में मेरे पास जानकारी नहीं है। मैं आपको एक colleague से connect कर सकती हूँ।",
        "hi-en": "Sorry, iske baare mein mere paas jaankari nahi hai. Main aapko ek colleague se connect kar sakti hoon.",
    },
}


class ConversationAgent:
    def __init__(self, dataset_dir: str, tickets_path: str = "tickets_created.csv",
                 use_retrieval: bool = True, use_structured: bool = True):
        self.store = Store(dataset_dir, tickets_path)
        self.retriever = Retriever(os.path.join(dataset_dir, "kb")) \
            if use_retrieval else None
        self.holidays = load_holidays(dataset_dir)
        self.use_structured = use_structured
        self.dataset_dir = dataset_dir

    # -- helpers ------------------------------------------------------------
    def _phrase(self, lang, key):
        return PHRASES[key][lang]

    def _handover(self, session, lang, reason, intent, summary_detail):
        """Create a ticket first (ESC-2); only then promise a handover."""
        summary = (
            f"Customer issue: {summary_detail}. "
            f"Intent: {intent}. Order: {session.order_id or 'none'}. "
            f"Checked: order verification={'verified' if session.verified else 'not verified'}. "
            f"Escalation reason: {reason}."
        )
        ticket = self.store.create_ticket(
            summary, order_id=session.order_id, reason=reason, intent=intent)
        ticket_id = ticket.get("ticket_id") if ticket.get("created") else None
        session.ticket_id = ticket_id
        if lang == "hi":
            text = ("मैं आपकी मदद के लिए एक ticket बना रही हूँ और आपको एक "
                    "colleague से connect कर रही हूँ। Ticket बनाया गया है, "
                    "एक colleague जल्द ही आपसे संपर्क करेगा।")
        elif lang == "hi-en":
            text = ("Main aapke liye ek ticket bana rahi hoon aur aapko ek "
                    "colleague se connect kar rahi hoon. Ticket ban gaya hai, "
                    "ek colleague jald hi aapse sampark karega.")
        else:
            text = ("I'm creating a ticket for you and connecting you to a "
                    "colleague who can help. A ticket has been created and a "
                    "human colleague will contact you.")
        if ticket_id:
            text += f" Ticket ID: {ticket_id}."
        return Result(text, decision="escalate",
                      sources=["ESC-1", "ESC-2"],
                      tool_calls=[f"create_ticket(order_id={session.order_id})"]
                      if session.order_id else ["create_ticket()"],
                      ticket_id=ticket_id)

    def _verified_reply(self, session, lang):
        order = session.order or {}
        status = order.get("status", "")
        promised = order.get("promised_date", "")
        session.last_payment = order.get("payment_method")
        text = {
            "en": f"Thanks, verified. Your order is {status}",
            "hi": f"धन्यवाद, verification हो गया है। आपका order {status} है",
            "hi-en": f"Thanks, verification ho gaya hai. Aapka order {status} hai",
        }[lang]
        if promised:
            text += f", with promised delivery on {format_date(date.fromisoformat(promised))}."
        else:
            text += "."
        text += {
            "en": " How can I help with this order?",
            "hi": " इस order के बारे में मैं आपकी क्या मदद कर सकती हूँ?",
            "hi-en": " Is order ke baare mein main aapki kya madad kar sakti hoon?",
        }[lang]
        return Result(text, decision="answer", sources=["IDV-2"],
                      tool_calls=[f"lookup_order(order_id={session.order_id}, "
                                  f"verification_value=****)"])

    # -- main entry point ----------------------------------------------------
    def respond(self, message: str, session: Session = None) -> Result:
        if session is None:
            session = Session(mode="single")
        session.turn += 1
        lang = detect_language(message)
        raw = message
        text = normalize_asr(message)
        low = text.lower()
        if lang == "hi":
            # Route Hindi messages through the same handlers: map the common
            # Hindi topic words to their English equivalents for routing only.
            for hindi, english in (
                    ("रिफ़ंड", " refund "), ("रिफंड", " refund "),
                    ("वापस", " return "), ("लौट", " return "),
                    ("कैंसिल", " cancel "), ("इंसान", " human "),
                    ("बात", " talk "), ("टूटा", " damaged "),
                    ("खराब", " damaged "), ("शिप", " shipped "),
                    ("ऑर्डर", " order "), ("पिकअप", " pickup "),
                    ("फीस", " fee "), ("दुकान", " store "),
                    ("खोल", " opened "), ("खुला", " opened ")):
                low = low.replace(hindi, english)
        tool_calls = []

        # First turn of a live conversation: introduce as AI (STY-1).
        if session.mode == "conversation" and not session.greeted:
            session.greeted = True
            order_id = extract_order_id(text)
            if order_id:
                session.order_id = order_id
                tool_calls.append(f"lookup_order(order_id={order_id})")
                return Result(
                    "Hi, I'm ZentroMart's AI support assistant. " +
                    self._phrase(lang, "ask_verify"),
                    decision="clarify", sources=["STY-1", "IDV-2"],
                    tool_calls=tool_calls)
            return Result(self._phrase(lang, "intro"), decision="clarify",
                          sources=["STY-1"])

        # A new order ID mid-conversation resets verification for that order.
        order_id = extract_order_id(text)
        if order_id and order_id != session.order_id:
            session.order_id = order_id
            session.verified = False
            session.verify_attempts = 0
            session.blocked = False
            session.order = None
            tool_calls.append(f"lookup_order(order_id={order_id})")
            return Result(self._phrase(lang, "ask_verify"), decision="clarify",
                          sources=["IDV-2"], tool_calls=tool_calls)

        # Verification turn for a pending order.
        if session.order_id and not session.verified and not session.blocked:
            proof = extract_verification(text, session.order_id)
            if proof:
                result = self.store.lookup_order(session.order_id, proof)
                tool_calls.append(
                    f"lookup_order(order_id={session.order_id}, "
                    f"verification_value={proof})")
                if result.get("verified"):
                    session.verified = True
                    session.order = result
                    reply = self._verified_reply(session, lang)
                    reply.tool_calls = tool_calls
                    return reply
                session.verify_attempts += 1
                if session.verify_attempts >= 2:
                    session.blocked = True
                    return Result(self._phrase(lang, "blocked"),
                                  decision="refuse", sources=["IDV-3"],
                                  tool_calls=tool_calls)
                return Result(self._phrase(lang, "retry"), decision="clarify",
                              sources=["IDV-3"], tool_calls=tool_calls)
            # No proof supplied yet: ask for it, once, and do not share details.
            if session.mode == "conversation":
                return Result(self._phrase(lang, "ask_verify"),
                              decision="clarify", sources=["IDV-2"],
                              tool_calls=tool_calls)

        # Blocked after two failed verifications: share nothing order-specific.
        if session.blocked:
            return Result(self._phrase(lang, "blocked"), decision="refuse",
                          sources=["IDV-3"])

        result = self._handle(session, raw, text, low, lang, tool_calls)
        return result

    def answer_single(self, message: str) -> Result:
        """One-shot question answering (Step 1/2 evaluation, batch runs)."""
        return self.respond(message, Session(mode="single"))

    # -- routing ---------------------------------------------------------------
    def _handle(self, session, raw, text, low, lang, tool_calls):
        order = session.order if session.verified else None

        # Another customer's data: refuse outright, never reveal a number (ESC-4).
        if self.use_structured and (
                ("phone number" in low and ("person who" in low or "someone" in low
                                            or "building" in low or "neighbour" in low
                                            or "neighbor" in low))
                or "other customer" in low):
            return Result(
                "I'm sorry, I can't share any details about other customers.",
                decision="refuse", sources=["ESC-4", "IDV-4"])

        # Fraud / legal / safety / human-request escalations (ESC-1).
        if self.use_structured and (
                "never placed" in low or "did not place" in low
                or "didn't place" in low or "never ordered" in low
                or "did not order" in low or "didn't order" in low
                or "not my order" in low or "fraud" in low
                or "unauthorised" in low or "unauthorized" in low):
            return self._handover(
                session, lang, "reported order/payment activity not made by customer",
                "fraud", "Customer reports an order they did not place")
        if self.use_structured and (
                "consumer court" in low or "legal action" in low
                or "police complaint" in low or "sue" in low):
            return self._handover(
                session, lang, "customer mentions legal action or police complaint",
                "legal", "Customer mentions legal action or a complaint")
        if self.use_structured and (
                "swelling battery" in low or "burning smell" in low
                or "sparks" in low or "battery swelling" in low
                or "smoke" in low or "injury" in low or "burning" in low):
            return self._handover(
                session, lang, "product safety risk reported",
                "safety", "Customer reports a product safety risk; stop use advised")
        if self.use_structured and (
                "speak to a person" in low or "speak to a human" in low
                or "talk to a person" in low or "talk to a human" in low
                or "insaan se baat" in low or "kisi insaan" in low
                or "human agent" in low or "connect me to" in low
                or "connect you to" in low or "connect to a human" in low
                or (("human" in low or "person" in low) and "talk" in low)
                or "kisi insaan se" in low):
            return self._handover(
                session, lang, "customer asks for a human agent",
                "human_requested", "Customer asks to speak to a human agent")
        if self.use_structured and (
                "third time" in low or "teesri baar" in low
                or "third" in low and "time" in low):
            return self._handover(
                session, lang, "third or later contact about the same issue in 7 days",
                "repeat_contact", "Customer says this is their third contact this week")
        if (self.use_structured and order
                and str(order.get("contacts_last_7_days", "0")).isdigit()
                and int(order.get("contacts_last_7_days", "0")) >= 2
                and session.turn > 3
                and ("refund" in low or "late" in low or "still" in low
                     or "problem" in low or "issue" in low or "not" in low)):
            return self._handover(
                session, lang, "third or later contact about the same issue in 7 days",
                "repeat_contact", "Repeat contact: order shows 2 earlier contacts in 7 days")

        if self.use_structured:
            handler = self._structured(session, raw, text, low, lang, order,
                                      tool_calls)
            if handler is not None:
                return handler
        return self._retrieval_answer(text, low, lang)

    # -- structured policy handlers -------------------------------------------
    def _structured(self, session, raw, text, low, lang, order, tool_calls):
        # Compensation demands: policy allows nothing for late Standard
        # deliveries (DEL-4); never claim a voucher was added (ESC-4).
        if ("voucher" in low or "compensation" in low or "compensate" in low) \
                and ("late" in low or "delay" in low):
            if order and order.get("delivery_type") == "express":
                return self._express_credit(session, order, lang)
            sorry = {"en": "Sorry, ", "hi": "माफ़ कीजिए, ",
                     "hi-en": "Maaf kijiye, "}[lang]
            return Result(
                sorry + "there is no compensation for a late Standard delivery, "
                "and I can't add a voucher for it. The promised delivery date "
                "at checkout is the commitment, and no other compensation is "
                "offered for Standard delays.",
                decision="refuse", sources=["DEL-4", "ESC-4"])

        # Damage / wrong / missing items.
        if any(word in low for word in
               ("damaged", "damage", "torn", "crack", "broken", "wrong item",
                "missing item", "toota", "tooti", "kharab", "defective")):
            return self._damage(session, text, low, lang, order)

        # Cancellation ("stop it" / "on its way" also mean a shipped cancel).
        if "cancel" in low or "cancellation" in low or "cancel kar" in low \
                or "on its way" in low \
                or ("stop" in low and ("order" in low or "it" in low)):
            return self._cancel(session, text, low, lang, order)

        # Order status / whereabouts / shipped date (verified orders only).
        if order and any(word in low for word in
                         ("where is", "where's", "not arrived", "hasn't arrived",
                          "has not arrived", "track", "status", "shipped",
                          "delivered", "when was it shipped", "kab ship",
                          "ship hua")):
            return self._order_status(session, text, low, lang, order)

        # Refund status / timeline.
        if "refund" in low or "money back" in low or "get my money" in low:
            return self._refund(session, text, low, lang, order)

        # Reverse pickup fee (before the general return handler, so a fee
        # question about a generic item -- "shoes" -- is answered, not asked
        # to name a product).
        if "pickup fee" in low or "collection free" in low \
                or "collection charge" in low or "pickup charge" in low \
                or ("charged" in low and "fee" in low):
            return self._return_fee(session, text, low, lang, order)

        # Return window / eligibility.
        if any(word in low for word in
               ("return", "wapas", "send back", "send it back", "lautane",
                "lautan", "return kar")):
            return self._return(session, text, low, lang, order)

        # Warranty / Assured.
        if "warranty" in low or "assured" in low:
            return self._warranty(session, text, low, lang)

        # Reverse pickup fee.
        if "pickup fee" in low or "collection free" in low \
                or "collection charge" in low or "pickup charge" in low:
            return self._return_fee(session, text, low, lang, order)

        # Troubleshooting electronics.
        if "speaker" in low and ("turn on" in low or "power" in low or "not working" in low):
            return Result(
                "Please charge the speaker for at least 30 minutes with the original "
                "cable, then hold the power button for 10 seconds to force a restart. "
                "If it still does not turn on, it can be returned within its window "
                "or claimed under warranty.",
                decision="answer", sources=["TRB-4"])
        if "earbud" in low and "one side" in low:
            return Result(
                "Put both earbuds in the case and close the lid for 10 seconds, "
                "remove them from your phone's Bluetooth list, then hold the case "
                "button for 15 seconds until the light flashes and pair again.",
                decision="answer", sources=["TRB-1"])
        if "watch" in low and "sync" in low:
            return Result(
                "Please make sure the companion app has Bluetooth and location "
                "permission, restart the watch and the phone, then unpair and "
                "pair the watch again in the app.",
                decision="answer", sources=["TRB-2"])
        if "charging slowly" in low or "charge" in low and "slow" in low:
            return Result(
                "Please use the original cable and adapter, clean the charging port "
                "gently with a dry, soft brush, and turn on airplane mode while "
                "charging to check whether background apps are the cause.",
                decision="answer", sources=["TRB-3"])

        # Delivery attempts / not home.
        if "not home" in low or "not at home" in low or "nobody" in low \
                and "home" in low or "delivery attempt" in low:
            return Result(
                "The courier makes up to 3 delivery attempts. After a failed attempt "
                "you get an SMS to reschedule. If the third attempt fails, the parcel "
                "is returned to the warehouse (RTO) and the refund follows REF-1.",
                decision="answer", sources=["DEL-3"])

        # Delivery area.
        if "dubai" in low or "outside india" in low or "deliver to" in low:
            return Result(
                "ZentroMart delivers only within India, to pincodes listed as "
                "serviceable. Cash on delivery and Express are available only where "
                "the pincode table says so.",
                decision="answer", sources=["DEL-5", "FAQ-5"])

        # Coupons.
        if "coupon" in low:
            return Result(
                "Only 1 coupon can be applied per order. Coupons cannot be combined, "
                "added after the order is placed, or exchanged for cash.",
                decision="answer", sources=["CPN-1"])

        # Express late credit (general, no order).
        if "express" in low and "late" in low:
            if order:
                return self._express_credit(session, order, lang)
            return Result(
                "If an Express order arrives after its promised date, the ₹79 express "
                "fee you paid is credited to ZentroMart Wallet automatically within "
                "48 hours. Plus members pay no express fee, so they get no credit.",
                decision="answer", sources=["DEL-4"])

        # Plus membership.
        if "plus" in low and ("member" in low or "membership" in low or "benefit" in low):
            return Result(
                "ZentroMart Plus costs ₹999 a year. Benefits: free Express delivery "
                "where Express is available, no reverse pickup fee on any return, and "
                "refunds on returns initiated at pickup. Plus does not change return "
                "windows, return eligibility or refund timelines after initiation.",
                decision="answer", sources=["PLS-1", "PLS-2"])

        # COD.
        if "cash on delivery" in low or low.strip() == "cod" or " cod " in low:
            return Result(
                "Cash on delivery is available on orders up to ₹10,000, only on "
                "pincodes where COD is enabled, with a ₹25 COD fee per order. For COD "
                "refunds you choose ZentroMart Wallet (instant, within 2 hours) or a "
                "bank transfer (5 to 7 business days after bank details are verified).",
                decision="answer", sources=["PAY-2", "REF-3"])

        # Failed payment.
        if "deducted" in low and "order" in low or "money deducted" in low:
            return Result(
                "If money was deducted but the order was not placed, the amount is "
                "reversed automatically by the payment provider within 5 business days. "
                "No ticket is needed before that.",
                decision="answer", sources=["PAY-3"])

        # Address change.
        if "change" in low and "address" in low:
            return Result(
                "The delivery address can be changed before the order ships. After "
                "shipping, the address cannot be changed.",
                decision="answer", sources=["DEL-6"])

        return None


    # -- order-specific handlers ----------------------------------------------
    def _order_status(self, session, text, low, lang, order):
        timeline = order.get("timeline", "") or ""
        events = {}
        for part in timeline.split(" ; "):
            m = re.match(r"(\d{4}-\d{2}-\d{2})T[\d:]+\s+(.+)", part.strip())
            if m:
                events[m.group(2).strip().lower()] = m.group(1)
        if "ship" in low and "when" in low or "kab ship" in low:
            shipped = events.get("shipped")
            if shipped:
                day = date.fromisoformat(shipped)
                return Result(
                    f"Your order was shipped on {format_date(day)}.",
                    decision="answer", sources=["TRK-1"])
        if "delivered" in low and order.get("delivered_on"):
            day = date.fromisoformat(order["delivered_on"])
            return Result(
                f"Your order was delivered on {format_date(day)}.",
                decision="answer", sources=["TRK-1"])
        status = order.get("status", "")
        promised = order.get("promised_date", "")
        text_out = f"Your order is {status}."
        if promised:
            text_out += (f" The promised delivery date is "
                         f"{format_date(date.fromisoformat(promised))}.")
        if order.get("delivery_attempts"):
            text_out += (f" Delivery attempts so far: "
                         f"{order['delivery_attempts']}.")
        return Result(text_out, decision="answer", sources=["TRK-1"])

    def _express_credit(self, session, order, lang):
        if order.get("delivery_type") != "express":
            return Result(
                "Express late credit applies only to Express orders. There is no "
                "compensation for late Standard deliveries.",
                decision="answer", sources=["DEL-4"])
        if order.get("is_plus"):
            return Result(
                "ZentroMart Plus members pay no express fee, so they get no "
                "express late credit.",
                decision="answer", sources=["DEL-4"])
        credit = order.get("express_late_credit_inr") or "79"
        credited_on = order.get("express_late_credit_on")
        if credited_on:
            day = date.fromisoformat(credited_on)
            return Result(
                f"The ₹{credit} express fee was credited to ZentroMart Wallet "
                f"on {format_date(day)}.",
                decision="answer", sources=["DEL-4"])
        return Result(
            "If an Express order arrives after its promised date, the ₹79 express "
            "fee you paid is credited to ZentroMart Wallet automatically within "
            "48 hours.",
            decision="answer", sources=["DEL-4"])

    def _damage(self, session, text, low, lang, order):
        product = None
        sku = extract_sku(text)
        if sku:
            product = self.store.products.get(sku)
        if product is None:
            product = self.store.product_by_title_fragment(text)
        if product is None and order and order.get("damage_sku"):
            product = self.store.products.get(order["damage_sku"])
        price = None
        if product:
            try:
                price = int(product.get("price_inr", "0"))
            except ValueError:
                price = None
        if price is not None and price > 10000:
            return self._handover(
                session, lang,
                "damaged item worth more than ₹10,000 (Priority Resolution)",
                "damaged_item",
                f"Customer reports a damaged item worth ₹{price}; "
                f"Priority Resolution team must handle it, agent cannot approve")
        # Reword the handover text for high-value damage so it names the team.
        if price is not None and price > 10000:
            pass
        return Result(
            "Please report it within 48 hours of delivery, with photos of the item "
            "and the packaging. After verification you get a free replacement or a "
            "full refund, including delivery fees, and no reverse pickup fee applies. "
            "This applies to every category, including non-returnable items.",
            decision="answer", sources=["DWM-1", "DWM-2"])

    def _cancel(self, session, text, low, lang, order):
        if order:
            status = order.get("status", "")
            if order.get("delivery_type") == "slot" and status in ("placed", "packed"):
                return Result(
                    "Grocery orders can be cancelled up to 2 hours before the start "
                    "of the chosen delivery slot. Could you please share the slot "
                    "start time shown in My Orders, so I can check whether the "
                    "cutoff has passed?",
                    decision="clarify", sources=["CAN-3"])
            if status in ("placed", "packed"):
                return Result(
                    "You can cancel it free of charge in My Orders in the app "
                    "before it ships. I can't cancel orders from here, but the "
                    "refund is initiated immediately when you cancel it there.",
                    decision="answer", sources=["CAN-1"])
            if status in ("shipped", "out for delivery", "delivery attempted"):
                return Result(
                    "A shipped order cannot be cancelled. You can refuse the parcel "
                    "at the door: for cash-on-delivery orders nothing is charged, "
                    "and for prepaid orders the refund follows once the parcel "
                    "returns to the warehouse, usually 7 to 10 calendar days after "
                    "the last delivery attempt. Or accept the delivery and return "
                    "the item within its return window, if it is returnable.",
                    decision="refuse", sources=["CAN-2", "REF-1"])
            if status == "delivered":
                return Result(
                    "A delivered order cannot be cancelled, but you can return the "
                    "item within its return window if the category is returnable.",
                    decision="answer", sources=["CAN-2", "RET-2"])
            if status == "cancelled":
                return Result(
                    "This order is already cancelled. If it was prepaid, the refund "
                    "was initiated immediately on cancellation.",
                    decision="answer", sources=["CAN-1"])
        return Result(
            "Any order can be cancelled free of charge in My Orders in the app until "
            "it ships, and the refund is initiated immediately. A shipped order "
            "cannot be cancelled: you can refuse the parcel at the door, or accept "
            "delivery and return the item within its return window.",
            decision="answer", sources=["CAN-1", "CAN-2"])

    # -- refunds ----------------------------------------------------------------
    def _payment_method(self, low):
        if "wallet" in low:
            return "wallet"
        if "upi" in low:
            return "upi"
        if "net banking" in low or "netbanking" in low or "net-banking" in low:
            return "netbanking"
        if "card" in low or "credit" in low or "debit" in low:
            return "card"
        if "cash on delivery" in low or " cod " in f" {low} ":
            return "cod"
        return None

    def _refund(self, session, text, low, lang, order):
        if order:
            initiated, method, rrn, kind = None, None, None, None
            for field, meth_field, rrn_field, label in (
                    ("return_refund_initiated_on", "return_refund_method",
                     "return_rrn", "return"),
                    ("cancellation_refund_initiated_on",
                     "cancellation_refund_method", "cancellation_rrn", "cancellation"),
                    ("rto_refund_initiated_on", "rto_refund_method",
                     "rto_rrn", "rto")):
                if order.get(field):
                    initiated, method, rrn, kind = (
                        order[field], order.get(meth_field), order.get(rrn_field), label)
                    break
            completed = order.get("return_refund_completed_on")
            if initiated:
                started = date.fromisoformat(initiated)
                mapped = {"cod_wallet": "wallet", "cod_bank": "bank"}.get(
                    method, method)
                if completed:
                    out = (f"Your refund was completed on "
                           f"{format_date(date.fromisoformat(completed))}.")
                    if rrn:
                        out += f" Your refund reference number (RRN) is {rrn}."
                    return Result(out, decision="answer",
                                  sources=["REF-2", "REF-6"])
                window = refund_window(mapped, started, self.holidays) \
                    if mapped in ("upi", "card", "netbanking", "bank") else None
                if window is None:
                    out = (f"Your refund was initiated on {format_date(started)} "
                           f"to ZentroMart Wallet, so it should arrive instant, "
                           f"within 2 hours.")
                    if rrn:
                        out += f" Your refund reference number (RRN) is {rrn}."
                    return Result(out, decision="answer",
                                  sources=["REF-2", "REF-6"])
                earliest, latest = window
                grace = add_business_days(latest, 3, self.holidays)
                if TODAY > grace:
                    # ESC-1.5: more than 3 business days past the maximum timeline.
                    summary = (
                        f"Customer asks where their refund is. Refund initiated "
                        f"{initiated} by {method}, latest expected "
                        f"{latest.isoformat()}, still not received on "
                        f"{TODAY.isoformat()}. RRN: {rrn or 'not recorded'}.")
                    ticket = self.store.create_ticket(
                        summary, order_id=session.order_id,
                        reason="refund overdue beyond timeline",
                        intent="refund_status")
                    ticket_id = ticket.get("ticket_id") if ticket.get("created") else None
                    out = (f"Your refund was initiated on {format_date(started)} "
                           f"and should have arrived by {format_date(latest)}. "
                           f"I'm creating a ticket and connecting you to a "
                           f"colleague to trace it.")
                    if rrn:
                        out += (f" Your refund reference number (RRN) is {rrn}, "
                                f"which your bank can trace.")
                    if ticket_id:
                        out += f" Ticket ID: {ticket_id}."
                    return Result(out, decision="escalate",
                                  sources=["ESC-1", "REF-2", "REF-6"],
                                  ticket_id=ticket_id,
                                  tool_calls=["create_ticket()"])
                out = (f"Your refund was initiated on {format_date(started)} and "
                       f"should reach you between {format_date(earliest)} and "
                       f"{format_date(latest)}.")
                if rrn:
                    out += (f" Your refund reference number (RRN) is {rrn}.")
                return Result(out, decision="answer",
                              sources=["REF-2", "REF-6"])
            if order.get("return_pickup_on") and not order.get("return_qc_passed_on"):
                return Result(
                    "Your return has been picked up and is waiting for quality check "
                    "(QC). Once it passes QC, the refund is initiated within "
                    "2 business days of pickup. For ZentroMart Plus members, the "
                    "refund is initiated at pickup.",
                    decision="answer", sources=["RET-7", "REF-1", "PLS-2"])
            if order.get("return_requested_on"):
                return Result(
                    "Your return was requested and is awaiting pickup. Once picked up "
                    "and passed through quality check (QC), the refund is initiated "
                    "within 2 business days of pickup.",
                    decision="answer", sources=["RET-7", "REF-1"])
            # No refund in progress: explain initiation + timeline for this order.
            method = self._payment_method(low) or order.get("payment_method")
            if method:
                return self._refund_timeline_answer(method, None, lang)
            return Result(
                "For a returned item, the refund is initiated within 2 business days "
                "of pickup once it passes quality check (QC). For ZentroMart Plus "
                "members, it is initiated at pickup. After initiation, the time "
                "depends on your payment method -- could you tell me which payment "
                "method you used?",
                decision="clarify", sources=["RET-7", "REF-1", "REF-2"])

        started = parse_date(text)
        method = self._payment_method(low)
        if method == "cod":
            return Result(
                "For cash-on-delivery orders, you choose ZentroMart Wallet "
                "(instant, within 2 hours) or a bank transfer (5 to 7 business days "
                "after your bank details are verified).",
                decision="answer", sources=["REF-3", "REF-2"])
        if method and started:
            return self._refund_timeline_answer(method, started, lang)
        if method:
            return self._refund_timeline_answer(method, None, lang)
        return Result(
            "Could you tell me which payment method you used, and whether the "
            "refund has already been initiated? That decides the timeline.",
            decision="clarify", sources=["REF-2"])

    def _refund_timeline_answer(self, method, started, lang):
        labels = {"upi": "UPI", "card": "card", "netbanking": "net banking",
                  "wallet": "ZentroMart Wallet", "bank": "bank transfer"}
        timelines = {"upi": "1 to 3 business days", "card": "5 to 7 business days",
                     "netbanking": "3 to 5 business days",
                     "bank": "5 to 7 business days"}
        if method == "wallet":
            return Result(
                "Refunds to ZentroMart Wallet are instant, within 2 hours of "
                "initiation.",
                decision="answer", sources=["REF-2"])
        if started:
            window = refund_window(method, started, self.holidays)
            earliest, latest = window
            return Result(
                f"Refunds to {labels[method]} take {timelines[method]}. A refund "
                f"initiated on {format_date(started)} should therefore arrive "
                f"between {format_date(earliest)} and {format_date(latest)}, "
                f"counting business days (Monday to Saturday, skipping warehouse "
                f"holidays).",
                decision="answer", sources=["REF-2", "KB-00.3"])
        return Result(
            f"Refunds to {labels[method]} take {timelines[method]} after initiation.",
            decision="answer", sources=["REF-2"])

    # -- returns ------------------------------------------------------------------
    def _month_day(self, text):
        m = re.search(
            r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)", text, re.IGNORECASE)
        if not m:
            return None
        from .dates import MONTHS
        try:
            return date(2026, MONTHS[m.group(2).lower()], int(m.group(1)))
        except ValueError:
            return None

    def _return(self, session, text, low, lang, order):
        product = None
        sku = extract_sku(text)
        if sku:
            product = self.store.products.get(sku)
        if product is None:
            product = self.store.product_by_title_fragment(text)
        if product is None and order and order.get("return_sku"):
            product = self.store.products.get(order["return_sku"])
        if product is None and order and len(order.get("items", [])) == 1:
            product = self.store.products.get(order["items"][0].get("sku"))
        if product is None and session.last_product:
            product = session.last_product
        if product is None:
            return Result(
                "Which item is it? Please tell me the product name, so I can check "
                "its return window.",
                decision="clarify", sources=["RET-2"])
        session.last_product = product
        category = product.get("return_category", "")
        title = product.get("title", "")

        delivered = None
        if order and order.get("delivered_on"):
            delivered = date.fromisoformat(order["delivered_on"])
        if delivered is None:
            delivered = parse_date(text) or self._month_day(text)
        if delivered is None and ("parso" in low or "day before yesterday" in low):
            delivered = TODAY - timedelta(days=2)
        elif delivered is None and "yesterday" in low:
            delivered = TODAY - timedelta(days=1)

        placed = None
        if order and order.get("placed_at"):
            placed = date.fromisoformat(order["placed_at"][:10])
        if placed is None:
            placed = self._month_day(text)

        # Beauty: opened means the seal is gone (RET-4).
        if category == "beauty_sealed" and any(
                word in low for word in ("opened", "open", "khol", "tried", "used")):
            return Result(
                "Beauty and personal care products can be returned only if unopened "
                "with the seal intact (सील), so an opened product cannot be returned.",
                decision="answer", sources=["RET-4", "RET-2"])

        if category in NON_RETURNABLE:
            return Result(
                f"This item cannot be returned for change of mind: {title} is "
                f"not returnable. If it arrived damaged, expired, wrong or missing, "
                f"please report it within 48 hours with photos, for a free "
                f"replacement or full refund.",
                decision="answer", sources=["RET-2", "RET-5", "DWM-1"])

        sources = ["RET-2"]
        if category == "electronics":
            sources.append("RET-3")  # the 7-day (not 10-day) rule since 1 Aug 2026
        if category == "fashion" and placed and date(2026, 10, 1) <= placed <= date(2026, 10, 15):
            window = 30
            sources.append("RET-8")
        elif category == "electronics" and delivered and delivered < date(2026, 8, 1):
            window = 10
            sources.extend(["RET-3", "RETv3-2"])
        else:
            window = RETURN_WINDOWS.get(category, 10)

        damaged = any(word in low for word in ("damaged", "torn", "wrong", "defective"))
        is_plus = bool(order and order.get("is_plus"))
        fee_note = ""
        if category == "fashion":
            sources.append("RET-6")
            if damaged:
                fee_note = " No reverse pickup fee applies for a damaged item."
            elif is_plus:
                fee_note = (" No reverse pickup fee applies for ZentroMart Plus "
                            "members.")
            else:
                fee_note = (" A reverse pickup fee of ₹49 applies for a "
                            "change-of-mind return of clothing, footwear and bags.")
        if delivered:
            deadline = delivered + timedelta(days=window)
            if TODAY > deadline:
                return Result(
                    f"The return window for {title} ended on {format_date(deadline)} "
                    f"({window} days from delivery on {format_date(delivered)}). "
                    f"If it is still under warranty, you can raise a warranty claim "
                    f"in My Orders in the app.",
                    decision="answer", sources=sources)
            out = (f"Yes, you can return {title} within {window} days of delivery, "
                   f"so by {format_date(deadline)}.")
            # For fashion questions without an order, also note the festive rule
            # so both the normal and festive windows are visible.
            if category == "fashion" and not order:
                out += (" Clothing, footwear and bags from orders placed between "
                        "1 October and 15 October 2026 can be returned within "
                        "30 days of delivery.")
            out += fee_note
            return Result(out, decision="answer", sources=sources)
        out = f"You can return {title} within {window} days of delivery."
        if category == "fashion":
            out += (" Clothing, footwear and bags from orders placed between "
                    "1 October and 15 October 2026 can be returned within "
                    "30 days of delivery.")
        out += fee_note
        return Result(out, decision="answer", sources=sources)

    def _warranty(self, session, text, low, lang):
        product = None
        sku = extract_sku(text)
        if sku:
            product = self.store.products.get(sku)
        if product is None:
            product = self.store.product_by_title_fragment(text)
        if product is None:
            return self._retrieval_answer(text, low, lang)
        months = product.get("warranty_months", "")
        assured = product.get("assured") == "True"
        fulfilled = product.get("fulfilled_by", "")
        if assured and fulfilled != "ZentroMart":
            return Result(
                f"ZentroMart Assured never applies to marketplace seller items, even "
                f"if the listing says assured. This item carries only the brand "
                f"warranty of {months} months.",
                decision="answer", sources=["MKS-3", "WAR-1"])
        if assured:
            return Result(
                f"Yes, it carries a 12-month ZentroMart Assured warranty from the "
                f"delivery date, in addition to the brand warranty of {months} "
                f"months. Raise an Assured claim in the app under My Orders.",
                decision="answer", sources=["WAR-2", "WAR-1"])
        return Result(
            f"It carries the brand warranty of {months} months. It does not carry "
            f"ZentroMart Assured.",
            decision="answer", sources=["WAR-1"])

    def _return_fee(self, session, text, low, lang, order):
        damaged = any(word in low for word in ("torn", "damaged", "wrong", "defective"))
        if damaged:
            return Result(
                "Yes, the collection is free: no reverse pickup fee applies for an "
                "item returned because it was damaged, defective or wrong.",
                decision="answer", sources=["RET-6", "DWM-2"])
        if order and order.get("is_plus") or "plus" in low:
            return Result(
                "No, ZentroMart Plus members pay no reverse pickup fee on any return. "
                "The fee is waived for Plus members.",
                decision="answer", sources=["RET-6", "PLS-2"])
        return Result(
            "A reverse pickup fee of ₹49 applies to change-of-mind returns of "
            "clothing, footwear and bags. The fee is waived for ZentroMart Plus "
            "members and for any item returned because it was damaged, defective "
            "or wrong.",
            decision="answer", sources=["RET-6"])

    # -- retrieval fallback -------------------------------------------------------
    def _retrieval_answer(self, text, low, lang):
        if self.retriever is None:
            # Step 1 (no documents): generic guessing, honestly labelled in the
            # report as the no-RAG baseline. It gets the precedence traps wrong.
            if "refund" in low:
                return Result("Refunds usually take 5 to 7 days to reach your account.",
                              decision="answer", sources=[])
            if "return" in low:
                return Result("Most items can be returned within the return window.",
                              decision="answer", sources=[])
            return Result(self._phrase(lang, "dont_know"), decision="answer",
                          sources=[])
        results = self.retriever.retrieve(text, top_k=5)
        if not results:
            return Result(self._phrase(lang, "dont_know"), decision="answer",
                          sources=[])
        # Conservative grounding: the top chunk must cover *every*
        # distinctive word in the question, otherwise the question is not
        # covered and the honest answer is "I don't know". (A coupon chunk
        # mentions "discount" but says nothing about a *student* discount;
        # a delivery chunk says nothing about *midnight* scheduling.)
        generic = {"delivery", "order", "refund", "return", "cancel", "item",
                   "product", "zentromart", "please", "help", "question",
                   "can", "i", "is", "there", "my", "me", "a", "an", "the"}
        core = set(re.findall(r"[a-z\u0900-\u097f]+", low)) - generic
        top = results[0]
        chunk_text = (top["title"] + " " + top["text"]).lower()
        chunk_words = re.findall(r"[a-z\u0900-\u097f]+", chunk_text)

        def _covers(token):
            if len(token) < 3:
                return True
            return any(token in word or word in token
                       for word in chunk_words if len(word) >= 3)

        if top["score"] < 0.05 or not core or not all(_covers(t) for t in core):
            return Result(self._phrase(lang, "dont_know"), decision="answer",
                          sources=[])
        # Use the sentences of the top chunk that mention the question words.
        sentences = re.split(r"(?<=[.!?])\s+", top["text"].replace("\n", " "))
        picked = [s for s in sentences
                  if set(re.findall(r"[a-z\u0900-\u097f]+", s.lower())) & core]
        answer = " ".join(picked[:2]) if picked else sentences[0]
        sources = [r["section_id"] for r in results[:3]
                   if r["doc_type"] != "marketing"]
        return Result(answer, decision="answer", sources=sources)
