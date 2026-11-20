"""Checks that run before anything is sent to the LLM."""

import re
from dataclasses import dataclass

MAX_PER_TRANSFER_INR = 499_999

_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b", re.I)
_SECRET_WORDS = re.compile(
    r"\b(otp|one[- ]time password|password|passcode|pin|cvv|cvc)\b\W{0,5}(is|:|=)?\W{0,5}\S*\d", re.I
)


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


@dataclass
class SensitiveCheck:
    found: bool
    kinds: list[str]
    redacted_text: str


def check_sensitive(text: str) -> SensitiveCheck:
    """Spot card numbers, IBANs, OTPs and passwords so we never forward them."""
    kinds = []
    redacted = text

    for match in _CARD_CANDIDATE.finditer(text):
        if _luhn_ok(match.group()):
            kinds.append("card number")
            redacted = redacted.replace(match.group(), "[card number removed]")

    for match in _IBAN.finditer(redacted):
        # skip things that are clearly not IBANs, like "UK 2024 trip"
        compact = match.group().replace(" ", "")
        if 15 <= len(compact) <= 34 and sum(c.isdigit() for c in compact) >= 8:
            kinds.append("IBAN")
            redacted = redacted.replace(match.group(), "[IBAN removed]")

    if _SECRET_WORDS.search(redacted):
        kinds.append("OTP / password")
        redacted = _SECRET_WORDS.sub("[secret removed]", redacted)

    return SensitiveCheck(found=bool(kinds), kinds=sorted(set(kinds)), redacted_text=redacted)


# --- Amount helper -----------------------------------------------------------

_LAKH_CRORE = re.compile(r"(\d+(?:\.\d+)?)\s*(lakh|lac|lakhs|crore|cr)\b", re.I)
_RUPEE_AMOUNT = re.compile(r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*(?:rupees|inr)\b", re.I)


def extract_inr_amount(text: str) -> int | None:
    """Pull a rupee amount out of things like '12 lakh', '₹7,50,000' or 'Rs 600000'."""
    m = _LAKH_CRORE.search(text)
    if m:
        value = float(m.group(1))
        unit = m.group(2).lower()
        return int(value * (10_000_000 if unit in ("crore", "cr") else 100_000))

    m = _RUPEE_AMOUNT.search(text)
    if m:
        raw = (m.group(1) or m.group(2)).replace(",", "")
        try:
            return int(float(raw))
        except ValueError:
            return None
    return None


def split_transfer(amount_inr: int) -> list[int]:
    """Split an amount into transfers that respect the ₹4,99,999 per-transfer cap."""
    parts = []
    remaining = amount_inr
    while remaining > 0:
        part = min(remaining, MAX_PER_TRANSFER_INR)
        parts.append(part)
        remaining -= part
    return parts


def format_inr(n: int) -> str:
    """Indian digit grouping: 1234567 -> 12,34,567"""
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail


def transfer_split_note(text: str) -> str | None:
    """If the user mentions an amount above the cap, pre-compute how to split it."""
    amount = extract_inr_amount(text)
    if not amount or amount <= MAX_PER_TRANSFER_INR:
        return None
    parts = split_transfer(amount)
    breakdown = " + ".join(f"₹{format_inr(p)}" for p in parts)
    return (
        f"Calculated: ₹{format_inr(amount)} is above the ₹4,99,999 per-transfer limit, "
        f"so it needs {len(parts)} separate transfers: {breakdown}."
    )
