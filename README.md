# Real Estate Assistant — simplified

A small rewrite of [AleksNeStu/ai-real-estate-assistant](https://github.com/AleksNeStu/ai-real-estate-assistant).
It keeps the core idea and drops the platform around it. Ask in plain English and get matching
listings, mortgage estimates, or market stats:

> *"2-bedroom apartment in Krakow under 900k"* · *"mortgage on 750k with 10% down at 6% for 25 years"* · *"average price per m2 in Warsaw"*

## Run it

```bash
uv sync
```

```bash
uv run uvicorn realestate.api:app --reload
```

Open http://localhost:8000.

It works without any API key. In **offline mode**, answers come from templates. To have Claude
write the answers, set `ANTHROPIC_API_KEY` before starting:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

| Env var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Enables Claude-written answers |
| `REA_LLM` | `auto` | `on` / `off` / `auto` (on when credentials are present) |
| `REA_MODEL` | `claude-opus-5-5` | Claude model |
| `REA_DATA_CSV` | `data/properties.csv` | Use your own listings (same columns) |

Run the tests:

```bash
uv run pytest
```

## How it works

```
question ──► query.analyze() ──► intent + filters
                                    │
            ┌───────────────┬───────┴────────┬──────────────┐
         search          mortgage          market         general
     search.search()   finance.mortgage()  market.city_stats()
            └───────────────┴───────┬────────┴──────────────┘
                                    ▼
                    facts (listings / numbers / stats)
                                    ▼
          llm.complete()  ── Claude phrases the answer from the facts
          (fallback)      ── template answer when offline / on error
```

All facts are computed locally and deterministically. Claude only gets the facts as
context, so it phrases answers but never supplies listings or numbers itself.

| File | What it does |
|---|---|
| `realestate/query.py` | Rule-based intent classification and filter extraction (city, type, rooms, price ranges like `500k` / `between 3000 and 5000`, area, year, amenities). Infers rent vs. sale from the budget size. |
| `realestate/search.py` | Hard filters, then keyword-overlap ranking |
| `realestate/finance.py` | Fixed-rate mortgage amortization |
| `realestate/market.py` | Per-city medians and price per m² |
| `realestate/assistant.py` | Orchestration, LLM context, offline answers, follow-ups by listing ID (`tell me more about P012`) |
| `realestate/llm.py` | Claude call. Falls back to offline on errors and refusals. |
| `realestate/agent.py` | Agent mode: Claude calls the same tools itself, in a loop (see below) |
| `realestate/api.py` | FastAPI: `/api/chat`, `/api/search`, `/api/properties/{id}`, `/api/mortgage`, `/api/market` |
| `static/index.html` | Chat, listing cards with estimated monthly payment, and a live mortgage calculator |
| `data/properties.csv` | 150 synthetic listings across 5 Polish cities |

## Two modes: pipeline vs. agent

The chat box has a **Pipeline / Agent** switch. Both modes use the same tools (search,
mortgage, market stats) over the same data. They differ in **who decides which tool runs**.

| | Pipeline (`assistant.py`) | Agent (`agent.py`) |
|---|---|---|
| Who picks the tool | Our code (`query.analyze`) | Claude |
| Tools per question | Exactly one | As many as needed, each informed by the last result |
| API calls per question | 0–1 | One per loop iteration |
| Same input, same route? | Always | Not guaranteed |
| Works offline | Yes | No, needs an API key |

Try a question that needs two steps:

> *"Find 2-room flats for sale in Krakow under 900k and work out the mortgage on the cheapest."*

The pipeline picks one intent and stops. The agent searches, reads the results, then calls
the mortgage tool with the price it found. The UI lists each tool call under the answer.

The loop in `run_agent()` is the core of every tool-using agent:

1. Send the conversation and tool definitions to Claude.
2. Append Claude's reply to the conversation unchanged.
3. No tool calls? Its text is the answer, so stop.
4. Otherwise run each tool. Send all results back in one message, with errors marked
   `is_error` so Claude can correct itself. Go to 1.

`MAX_ITERATIONS` stops a confused model from looping forever. If the agent fails (API error,
refusal, or hitting that limit), the app answers in pipeline mode instead. The loop is written
by hand so it's easy to follow. In production you'd usually let the SDK run it
(`client.beta.messages.tool_runner`).

## What was simplified

| Original | Here |
|---|---|
| Next.js 16 + React 19 frontend, 9 languages | One vanilla HTML/JS page, English only |
| 13 LLM providers via LangChain, lazy provider factory | Anthropic SDK, plus an offline mode |
| ChromaDB + embeddings, hybrid retrieval, MMR reranker | Structured filters + keyword ranking over an in-memory list |
| Hybrid agent (RAG chain / tool agent / web research routing) | A fixed pipeline (analyze → compute facts → phrase the answer), plus an optional tool-calling agent |
| 1,400-line query analyzer with multi-intent scoring | ~200-line rule-based analyzer, 4 intents |
| PostgreSQL/SQLite, Alembic migrations | CSV file |
| Mortgage, TCO, ROI, rent-vs-buy, CMA, valuation, forecast tools | Mortgage calculator + market stats |
| Auth (JWT/OAuth/API keys), leads, agents CRM, e-signatures, webhooks, push, admin, MCP | Removed |
| Docker, k8s, Render/Railway/Vercel, security scanning CI | `uv run uvicorn …` |

Good next steps if you need more: swap `search.search()` for embeddings when listings have
rich free text, add more tools to `assistant.answer()`, or put a database behind
`data.load_properties()`.
