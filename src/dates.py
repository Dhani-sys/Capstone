"""Date rules for ZentroMart (dataset frozen on 20 October 2026).

Business days are Monday-Saturday, skipping the warehouse holidays in
``data/holidays.csv`` (KB-00.3). Return windows are calendar days (RET-2).
Refund timelines are business days counted *after* the initiation day (REF-2).

Worked example from the practice set: a card refund initiated on
17 October 2026 takes 5-7 business days. Counting after the 17th, skipping
Sunday 18 Oct and the 19 Oct warehouse holiday, gives 24 Oct (5th business
day) to 27 Oct (7th business day) -- exactly the expected answer.
"""
from __future__ import annotations

import csv
import os
from datetime import date, timedelta

# The dataset's reference "today" -- never the computer clock (KB-00.2).
TODAY = date(2026, 10, 20)

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

# Payment method -> (min, max) business days after initiation (REF-2).
# None means "instant" (Wallet).
REFUND_TIMELINES = {
    "upi": (1, 3),
    "card": (5, 7),
    "netbanking": (3, 5),
    "net banking": (3, 5),
    "wallet": None,
    "bank": (5, 7),  # bank transfer for COD orders
}

# Return window in calendar days by catalog return_category (RET-2).
RETURN_WINDOWS = {
    "electronics": 7,
    "fashion": 15,
    "home_kitchen": 10,
    "beauty_sealed": 7,
    "books_media": 10,
    "grocery": 0,
    "innerwear": 0,
    "digital": 0,
}

NON_RETURNABLE = {"grocery", "innerwear", "digital"}


def load_holidays(dataset_dir: str) -> set:
    path = os.path.join(dataset_dir, "data", "holidays.csv")
    holidays = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    holidays.add(date.fromisoformat(row["date"].strip()))
                except (KeyError, ValueError):
                    pass
    # Fallback to the holidays printed in KB-00.3 if the file is absent.
    if not holidays:
        holidays = {
            date(2026, 8, 15), date(2026, 10, 2),
            date(2026, 10, 19), date(2026, 11, 9),
        }
    return holidays


def is_business_day(day: date, holidays: set) -> bool:
    """Monday-Saturday, excluding holidays. Sunday (weekday 6) is never one."""
    return day.weekday() != 6 and day not in holidays


def add_business_days(start: date, count: int, holidays: set) -> date:
    """The date `count` business days after `start` (start day not counted)."""
    day = start
    remaining = count
    while remaining > 0:
        day += timedelta(days=1)
        if is_business_day(day, holidays):
            remaining -= 1
    return day


def add_calendar_days(start: date, count: int) -> date:
    return start + timedelta(days=count)


def format_date(day: date) -> str:
    """'27 October 2026' -- the format the style guide (STY-2) requires."""
    return f"{day.day} {day.strftime('%B')} {day.year}"


def parse_date(text: str):
    """Parse '17 October 2026' / '17 Oct 2026' / '2026-10-17' from free text."""
    import re

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.search(
        r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
        r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|"
        r"Sep|Sept|Oct|Nov|Dec)\s+(\d{4})",
        text, re.IGNORECASE,
    )
    if m:
        month_name = m.group(2).lower()
        month = MONTHS.get(month_name) or {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
            "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
        }.get(month_name)
        if month:
            try:
                return date(int(m.group(3)), month, int(m.group(1)))
            except ValueError:
                return None
    return None


def refund_window(method: str, initiated: date, holidays: set):
    """Return (earliest, latest) arrival dates, or None for instant Wallet."""
    timeline = REFUND_TIMELINES.get(method)
    if timeline is None:
        return None
    lo, hi = timeline
    return (
        add_business_days(initiated, lo, holidays),
        add_business_days(initiated, hi, holidays),
    )
