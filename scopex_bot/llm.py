"""LLM calls via Groq: query rewriting and grounded answering."""

import os

from groq import BadRequestError, Groq

from .retriever import Hit

ANSWER_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
REWRITE_MODEL = os.getenv("GROQ_REWRITE_MODEL", "openai/gpt-oss-20b")
HANDOFF_TOKEN = "[[HANDOFF]]"

REWRITE_PROMPT = """You turn a customer's latest chat message into a short search query \
for a money-transfer help center.
- Use the chat history to resolve words like "it", "that" or "the transfer".
- The customer may write in English, Hindi or Hinglish. Always output the query in English.
- Output only the query, nothing else."""

ANSWER_PROMPT = f"""You are Scopex Assistant, the support assistant for Scopex, a zero-fee app \
that NRIs use to send money from Europe and the US to India.

How to answer:
- Use ONLY the help center articles given below. Never make up fees, limits, timelines or policies.
- Cite the articles you used inline, like [1] or [2].
- Keep it short and friendly: 2-5 sentences, or short numbered steps for how-to questions.
- Reply in the same language and style the customer used (English, Hindi or Hinglish).

Limits:
- You cannot see anyone's account, KYC status or transactions. For questions about a specific \
transfer, explain what the articles say (status meanings, normal timelines, what to do next), then \
offer to connect them to the team.
- Never ask for card numbers, IBANs, OTPs, passwords or passkeys.
- Exchange rates change every few minutes. Never quote a live rate; say the app shows the current rate.
- No financial, tax or legal advice. For FIRC or tax documents, point to their bank as the articles say.

Handing off to a human:
If the articles don't answer the question, or the customer needs something only the support team \
can do (check a specific transfer, process a refund, fix a stuck KYC, account access problems), write \
one short helpful line and then put {HANDOFF_TOKEN} on its own line at the very end."""


def _format_context(hits: list[Hit], note: str | None) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        a = hit.article
        blocks.append(f"[{i}] {a.title}\nSource: {a.url}\n{a.content}")
    context = "\n\n".join(blocks)
    if note:
        context += f"\n\nExtra calculation for this question (already checked, safe to use):\n{note}"
    return context


class GroqLLM:
    def __init__(self, api_key: str | None = None):
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    def _chat(self, model: str, messages: list[dict], max_tokens: int) -> str:
        kwargs = dict(model=model, messages=messages, temperature=0.2, max_completion_tokens=max_tokens)
        try:
            resp = self.client.chat.completions.create(reasoning_effort="low", **kwargs)
        except BadRequestError:
            # model doesn't support reasoning_effort - try again without it
            resp = self.client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    def rewrite_query(self, message: str, history: list[dict]) -> str:
        recent = history[-6:]
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        user = f"Chat history:\n{convo or '(none)'}\n\nLatest message: {message}"
        query = self._chat(
            REWRITE_MODEL,
            [{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": user}],
            max_tokens=400,
        )
        return query.strip().strip('"') or message

    def answer(self, message: str, history: list[dict], hits: list[Hit], note: str | None) -> str:
        messages = [{"role": "system", "content": ANSWER_PROMPT}]
        messages += history[-6:]
        messages.append(
            {
                "role": "user",
                "content": f"Help center articles:\n\n{_format_context(hits, note)}\n\n"
                f"Customer message: {message}",
            }
        )
        return self._chat(ANSWER_MODEL, messages, max_tokens=1200)
