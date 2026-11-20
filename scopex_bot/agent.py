"""The support flow: guardrails -> rewrite -> retrieve -> answer -> handoff check."""

import re
from dataclasses import dataclass, field
from typing import Protocol

from .guardrails import check_sensitive, transfer_split_note
from .knowledge import Article
from .llm import HANDOFF_TOKEN
from .retriever import Hit, HybridRetriever

CONTACT_URL = "https://scopex.com/contact-us/"


class LLM(Protocol):
    def rewrite_query(self, message: str, history: list[dict]) -> str: ...
    def answer(self, message: str, history: list[dict], hits: list[Hit], note: str | None) -> str: ...


@dataclass
class BotReply:
    text: str
    sources: list[Article] = field(default_factory=list)
    handoff: bool = False
    blocked: bool = False
    redacted_message: str | None = None
    search_query: str | None = None


def _cited_sources(text: str, hits: list[Hit]) -> list[Article]:
    cited = []
    for n in re.findall(r"\[(\d+)\]", text):
        i = int(n) - 1
        if 0 <= i < len(hits) and hits[i].article not in cited:
            cited.append(hits[i].article)
    return cited


class SupportAgent:
    def __init__(self, retriever: HybridRetriever, llm: LLM, top_k: int = 4):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def reply(self, message: str, history: list[dict]) -> BotReply:
        # 1. Never forward card numbers, IBANs or OTPs anywhere
        check = check_sensitive(message)
        if check.found:
            return BotReply(
                text=(
                    f"For your safety, please don't share your {' or '.join(check.kinds)} in chat. "
                    "I've removed it from your message. Scopex will never ask for your OTP, password "
                    "or full card details. What can I help you with?"
                ),
                blocked=True,
                redacted_message=check.redacted_text,
            )

        # 2. Turn follow-ups / Hinglish into a clean English search query
        try:
            query = self.llm.rewrite_query(message, history)
        except Exception:
            query = message

        # 3. Find the most relevant help articles
        hits = self.retriever.search(query, k=self.top_k)

        # 4. Pre-compute things the LLM is bad at (splitting big amounts)
        note = transfer_split_note(message) or transfer_split_note(query)

        # 5. Grounded answer
        raw = self.llm.answer(message, history, hits, note)
        handoff = HANDOFF_TOKEN in raw
        text = raw.replace(HANDOFF_TOKEN, "").strip()

        return BotReply(
            text=text,
            sources=_cited_sources(text, hits),
            handoff=handoff,
            search_query=query,
        )
