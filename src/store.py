"""Data layer: reads the ZentroMart CSVs and mirrors the supplied mock tools.

The mock tools in the dataset (``tools/tools.py``) gate order details behind
verification: ``lookup_order(order_id)`` alone returns no details, and a
4-digit phone suffix or 6-digit delivery pincode unlocks the record. This
module implements exactly that contract against the CSVs, so the agent
never sees order details for an unverified caller.

IDs, pincodes and phone numbers are always kept as strings. ``create_ticket``
appends to a runtime tickets file (never to the dataset itself) and a ticket
only counts as created when the returned dict says ``created=True`` (ESC-2).
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from uuid import uuid4


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Store:
    def __init__(self, dataset_dir: str, tickets_path: str = "tickets_created.csv"):
        self.dataset_dir = dataset_dir
        self.tickets_path = tickets_path
        data = os.path.join(dataset_dir, "data")
        self.customers = {
            r["customer_id"]: r for r in _read_csv(os.path.join(data, "customers.csv"))
        }
        self.orders = {
            r["order_id"]: r for r in _read_csv(os.path.join(data, "orders.csv"))
        }
        self.products = {
            r["sku"]: r for r in _read_csv(os.path.join(data, "products.csv"))
        }
        self.pincodes = {
            r["pincode"]: r for r in _read_csv(os.path.join(data, "pincodes.csv"))
        }
        self.items = {}
        for row in _read_csv(os.path.join(data, "order_items.csv")):
            self.items.setdefault(row["order_id"], []).append(row)

    # -- order lookup (mirrors tools.lookup_order) -------------------------
    def lookup_order(self, order_id: str, verification_value: str = None) -> dict:
        oid = "".join(ch for ch in str(order_id) if ch.isdigit())
        if verification_value is None or str(verification_value).strip() == "":
            return {
                "verification_required": True, "order_id": oid,
                "message": "Ask for the registered phone last 4 digits OR delivery pincode.",
            }
        proof = "".join(ch for ch in str(verification_value) if ch.isdigit())
        order = self.orders.get(oid)
        if order:
            customer = self.customers.get(order["customer_id"], {})
            phone = customer.get("phone", "")
            valid = (
                (len(proof) == 4 and phone.endswith(proof))
                or (len(proof) == 6 and proof == order.get("delivery_pincode"))
            )
            if valid:
                result = dict(order)
                result.update({
                    "found": True, "verified": True,
                    "items": self.items.get(oid, []),
                    "customer_first_name": customer.get("name", "").split()[0]
                    if customer.get("name") else "",
                    "is_plus": customer.get("is_plus") == "True",
                })
                return result
        return {
            "verified": False, "verification_required": True,
            "message": "Verification failed or order not found.",
        }

    # -- tickets (mirrors tools.create_ticket) ------------------------------
    def create_ticket(self, summary: str, order_id: str = None,
                      reason: str = None, intent: str = None) -> dict:
        if not summary or len(summary.strip()) < 20:
            return {"created": False,
                    "error": "summary too short: include issue, checks and reason"}
        tid = "TKT" + uuid4().hex[:12].upper()
        row = {
            "ticket_id": tid,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "order_id": order_id or "", "intent": intent or "",
            "reason": reason or "", "summary": summary,
        }
        fresh = not os.path.exists(self.tickets_path)
        directory = os.path.dirname(self.tickets_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.tickets_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if fresh:
                writer.writeheader()
            writer.writerow(row)
        return {"created": True, "ticket_id": tid}

    # -- catalog -------------------------------------------------------------
    def find_product(self, sku: str = None, title: str = None):
        if sku:
            product = self.products.get(sku.strip().upper())
            if product:
                return product
        if title:
            needle = title.strip().lower()
            for product in self.products.values():
                if product["title"].lower() == needle:
                    return product
            for product in self.products.values():
                if needle and needle in product["title"].lower():
                    return product
        return None

    def product_by_title_fragment(self, text: str):
        """Best catalog product whose title is (almost) fully named in `text`.

        Customers type a short form ("Kaafi Running Sneakers") while the
        catalog title adds a colour/pack suffix ("Kaafi Running Sneakers,
        Blue"), so exact-substring matching fails. Token overlap with at
        most two unmatched title tokens (colour, pack size) is the match.
        """
        import re as _re
        msg_tokens = set(_re.findall(r"[a-z0-9]+", text.lower()))
        best, best_overlap = None, 0
        for product in self.products.values():
            title = product["title"].lower()
            if title in text.lower():
                return product
            title_tokens = set(_re.findall(r"[a-z0-9]+", title))
            overlap = title_tokens & msg_tokens
            if len(overlap) >= 2 and len(title_tokens) - len(overlap) <= 2 \
                    and len(overlap) > best_overlap:
                best, best_overlap = product, len(overlap)
        return best
