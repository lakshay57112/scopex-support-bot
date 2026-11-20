import os

import streamlit as st

from scopex_bot.agent import CONTACT_URL, SupportAgent
from scopex_bot.guardrails import check_sensitive
from scopex_bot.knowledge import load_articles
from scopex_bot.llm import GroqLLM
from scopex_bot.retriever import HybridRetriever

st.set_page_config(page_title="Scopex Assistant", page_icon="💸", layout="centered")

SUGGESTIONS = [
    "How long does KYC take?",
    "I want to send 12 lakh to India, how?",
    "Transfer shows completed but my mom didn't get the money",
    "Mera passkey kaam nahi kar raha, kya karu?",
    "Can I send to an NRE account?",
    "Can I cancel a transfer and get a refund?",
]


def get_api_key() -> str | None:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


@st.cache_resource(show_spinner="Loading help center articles...")
def get_retriever() -> HybridRetriever:
    return HybridRetriever(load_articles())


def render_reply_extras(sources, handoff):
    if sources:
        with st.expander("Sources"):
            for a in sources:
                st.markdown(f"- [{a.title}]({a.url})")
    if handoff:
        st.info("This one needs a person from the Scopex team.")
        st.link_button("Talk to a Scopex specialist", CONTACT_URL)


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("About this demo")
    st.write(
        "An AI support assistant for Scopex that answers questions using only the "
        "public [Scopex Help Center](https://help.scopex.com/en)."
    )
    st.markdown(
        "**How it works**\n"
        "1. Blocks card numbers, IBANs and OTPs before anything leaves the chat\n"
        "2. Rewrites follow-ups and Hinglish into a clean search query\n"
        "3. Hybrid search (keywords + embeddings) over the help articles\n"
        "4. Answers with citations, or hands off to a human when it isn't sure"
    )
    retriever = get_retriever()
    st.caption(f"{len(retriever.articles)} articles indexed")
    st.caption(f"Retrieval mode: {retriever.mode}")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(
        "Unofficial demo, not affiliated with Scopex. Built by Lakshay · "
        "[GitHub](https://github.com/lakshay57112) · "
        "[LinkedIn](https://www.linkedin.com/in/lakshay57112/)"
    )

# --- Main --------------------------------------------------------------------
st.title("💸 Scopex Assistant")
st.caption("Ask anything about sending money to India with Scopex.")

api_key = get_api_key()
if not api_key:
    st.error("Add GROQ_API_KEY to `.streamlit/secrets.toml` (or as an environment variable) to start.")
    st.stop()

if "agent" not in st.session_state:
    st.session_state.agent = SupportAgent(retriever, GroqLLM(api_key))
if "messages" not in st.session_state:
    st.session_state.messages = []

# replay the conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_reply_extras(msg.get("sources", []), msg.get("handoff", False))

# suggested questions on an empty chat
pending = None
if not st.session_state.messages:
    st.write("Try one of these:")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTIONS):
        if cols[i % 2].button(q, use_container_width=True):
            pending = q

prompt = st.chat_input("Type your question...") or pending

if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.chat_message("user"):
        st.markdown(check_sensitive(prompt).redacted_text)

    with st.chat_message("assistant"):
        with st.spinner("Checking the help center..."):
            try:
                reply = st.session_state.agent.reply(prompt, history)
            except Exception as err:
                st.error(f"Something went wrong talking to the model: {err}")
                st.stop()

    # store the redacted version if the user pasted something sensitive
    st.session_state.messages.append({"role": "user", "content": reply.redacted_message or prompt})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply.text,
            "sources": reply.sources,
            "handoff": reply.handoff,
        }
    )
    st.rerun()
