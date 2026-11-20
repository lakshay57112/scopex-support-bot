# Scopex Assistant

An AI support assistant for [Scopex](https://scopex.com), the zero-fee app NRIs use to send money from Europe and the US to India. It answers customer questions using only the public [Scopex Help Center](https://help.scopex.com/en), cites the article it used, and hands off to a human when it isn't sure.

**Live demo:** _add your Streamlit link here_

> Unofficial side project. Not affiliated with or endorsed by Scopex. All answers come from Scopex's public help content.

## Why I built this

Remittance support gets a lot of the same questions: how long KYC takes, why a transfer is stuck in "Verifying your deposited funds", how to fix passkey errors, how to send more than ₹4,99,999. Most of these are already answered in the help center, but customers ask in chat anyway, often in Hinglish and often at 2 AM. This bot handles those instantly and passes the rest to the team.

## What it does

- **Answers from the help center only.** No made-up fees, limits or timelines. Every answer cites its source article.
- **Hands off to a human** when the articles don't cover it, or when it needs account access (a specific transfer, a refund, a stuck KYC).
- **Blocks sensitive data.** Card numbers (Luhn-checked), IBANs, OTPs and passwords are caught and removed before anything is sent to the model.
- **Understands Hinglish and follow-ups.** "Mera passkey kaam nahi kar raha" or "what about for NRE?" get rewritten into a clean English search query first.
- **Does the maths.** Ask to send 12 lakh and it works out the split under the ₹4,99,999 per-transfer cap (₹4,99,999 + ₹4,99,999 + ₹2,00,002) in code, so the model never has to guess.

## How it works

```
user message
   │
   ├─ guardrails ── card / IBAN / OTP found? → warn, redact, stop
   │
   ├─ query rewrite (gpt-oss-20b) ── follow-ups + Hinglish → English search query
   │
   ├─ hybrid retrieval ── TF-IDF keywords + bge-small embeddings, merged with reciprocal rank fusion
   │
   ├─ amount helper ── splits large INR amounts under the per-transfer cap
   │
   └─ grounded answer (gpt-oss-120b) ── cites [1], [2] … or emits a handoff signal
```

**Stack:** Python, Streamlit, Groq (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`), fastembed (`BAAI/bge-small-en-v1.5`), scikit-learn.

## Retrieval evaluation

`eval/questions.json` has 33 realistic customer questions, each labelled with the article that should answer it.

| Mode | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| Keyword only (TF-IDF) | 85% | 94% | 0.895 |
| Hybrid (keyword + embeddings) | **94%** | **100%** | **0.965** |

Adding embeddings fixed most of the keyword misses, which were paraphrases like "I changed my phone and now can't approve transfers". With hybrid search, the right article is in the top 3 for every test question.

## Run it locally

```bash
git clone https://github.com/lakshay57112/scopex-support-bot.git
cd scopex-support-bot
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your free [Groq API key](https://console.groq.com/keys). Then:

```bash
streamlit run app.py          # the app
python -m pytest              # tests
python -m eval.run_eval       # retrieval metrics
```

The first run downloads the embedding model (~70 MB). If it can't, the app falls back to keyword search and keeps working.

## Project structure

```
app.py                    Streamlit chat UI
scopex_bot/
  agent.py                guardrails → rewrite → retrieve → answer → handoff
  retriever.py            hybrid TF-IDF + embedding search with RRF
  guardrails.py           sensitive-data checks, INR amount parsing and splitting
  llm.py                  Groq calls and prompts
  knowledge.py            loads the knowledge base
data/knowledge_base.json  29 articles from the Scopex Help Center
eval/                     retrieval evaluation
tests/                    unit tests
```

## What I'd add with access to Scopex's systems

- Plug into the existing support chat so handoffs land in the agent inbox with the conversation attached
- Look up a user's transfer status (with auth) so "where is my money" gets a real answer
- Log unanswered questions to find gaps in the help center
- Sync the knowledge base automatically when help articles change
