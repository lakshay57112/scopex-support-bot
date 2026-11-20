import pytest

from scopex_bot.agent import SupportAgent
from scopex_bot.guardrails import (
    check_sensitive,
    extract_inr_amount,
    format_inr,
    split_transfer,
    transfer_split_note,
)
from scopex_bot.knowledge import load_articles
from scopex_bot.llm import HANDOFF_TOKEN
from scopex_bot.retriever import HybridRetriever


# --- guardrails --------------------------------------------------------------

@pytest.mark.parametrize(
    "text, kind",
    [
        ("my card is 4111 1111 1111 1111 please check", "card number"),
        ("IBAN DE89 3704 0044 0532 0130 00", "IBAN"),
        ("my otp is 482913", "OTP / password"),
        ("password: hunter2024", "OTP / password"),
    ],
)
def test_sensitive_data_is_caught(text, kind):
    result = check_sensitive(text)
    assert result.found
    assert kind in result.kinds
    assert "removed" in result.redacted_text


@pytest.mark.parametrize(
    "text",
    [
        "I want to send 12 lakh to India",
        "my transfer id is 884213, where is it?",
        "how long does KYC take?",
        "I sent €2000 on 14 November 2026",
    ],
)
def test_normal_messages_pass(text):
    assert not check_sensitive(text).found


def test_amount_parsing():
    assert extract_inr_amount("send 12 lakh home") == 1_200_000
    assert extract_inr_amount("₹7,50,000 to my dad") == 750_000
    assert extract_inr_amount("Rs 600000") == 600_000
    assert extract_inr_amount("1.5 crore") == 15_000_000
    assert extract_inr_amount("send 500 euro") is None


def test_split_respects_cap():
    assert split_transfer(1_200_000) == [499_999, 499_999, 200_002]
    assert sum(split_transfer(15_000_000)) == 15_000_000
    assert all(p <= 499_999 for p in split_transfer(15_000_000))


def test_format_inr():
    assert format_inr(499_999) == "4,99,999"
    assert format_inr(12_00_000) == "12,00,000"
    assert format_inr(950) == "950"


def test_split_note_only_above_cap():
    assert transfer_split_note("send 2 lakh") is None
    note = transfer_split_note("I want to send 12 lakh")
    assert "3 separate transfers" in note


# --- agent flow with a fake LLM ------------------------------------------------

class FakeLLM:
    def __init__(self, answer_text):
        self.answer_text = answer_text
        self.calls = 0
        self.last_note = None

    def rewrite_query(self, message, history):
        return message

    def answer(self, message, history, hits, note):
        self.calls += 1
        self.last_note = note
        return self.answer_text


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever(load_articles(), use_embeddings=False)


def test_sensitive_message_never_reaches_llm(retriever):
    llm = FakeLLM("should not be called")
    reply = SupportAgent(retriever, llm).reply("card 4111111111111111, why failed?", [])
    assert reply.blocked
    assert llm.calls == 0
    assert "4111" not in reply.redacted_message


def test_citations_become_sources(retriever):
    llm = FakeLLM("KYC usually takes 2-3 minutes, up to an hour [1].")
    reply = SupportAgent(retriever, llm).reply("how long does kyc take", [])
    assert not reply.handoff
    assert [a.id for a in reply.sources] == ["kyc-time"]


def test_handoff_token_is_detected_and_stripped(retriever):
    llm = FakeLLM(f"I can't see your account, let me connect you to the team.\n{HANDOFF_TOKEN}")
    reply = SupportAgent(retriever, llm).reply("why was my account blocked?", [])
    assert reply.handoff
    assert HANDOFF_TOKEN not in reply.text


def test_big_amount_gets_split_note(retriever):
    llm = FakeLLM("You'll need 3 transfers [1].")
    SupportAgent(retriever, llm).reply("I want to send 12 lakh to India", [])
    assert llm.last_note and "₹4,99,999 + ₹4,99,999 + ₹2,00,002" in llm.last_note
