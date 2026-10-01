"""Agent mode: Claude decides which tools to call, in a loop.

Compare with assistant.py (pipeline mode):

- Pipeline: OUR code classifies the question, runs one tool, and Claude only words the answer.
- Agent:    CLAUDE reads the question, calls whichever tools it needs (possibly several, each
            informed by the last result), then answers.

The agent can handle multi-step questions the pipeline can't, e.g. "find a 2-room flat in
Krakow under 900k and work out the mortgage on the cheapest one". The cost: one API call per
loop iteration (slower, pricier), non-deterministic routing, and it needs an API key.

The loop is written out by hand so you can see every step. In production you'd usually let
the SDK run it for you (`client.beta.messages.tool_runner`), which does the same thing.
"""

import json
import logging

import anthropic

from . import llm
from .assistant import property_json
from .data import AMENITY_LABELS, Property
from .finance import mortgage
from .market import city_stats
from .query import CITY_VARIANTS, PROPERTY_TYPES
from .search import search

log = logging.getLogger(__name__)

MAX_ITERATIONS = 8  # safety stop: never let a confused model loop forever

SYSTEM_PROMPT = """You are a real estate assistant for listings in Polish cities. Prices are in \
PLN; rent prices are per month.

Use the tools to look up listings, mortgage figures and market statistics. Never invent \
listings, prices or features: every fact in your answer must come from a tool result. Refer \
to listings by their ID, like [P012]. Mortgage figures are estimates, not a lending offer. \
Keep answers short and scannable."""

# Tool definitions: what Claude sees. The descriptions matter as much as the code: they are
# how the model decides when to call each tool. `strict: True` guarantees inputs match the schema.
TOOLS = [
    {
        "name": "search_listings",
        "description": (
            "Search property listings. All filters are optional; omit any the user didn't ask "
            "for. Prices are in PLN: sale prices are totals, rent prices are per month."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "enum": list(CITY_VARIANTS)},
                "property_type": {"type": "string", "enum": list(PROPERTY_TYPES)},
                "listing_type": {"type": "string", "enum": ["sale", "rent"]},
                "rooms": {"type": "integer", "description": "Exact number of rooms"},
                "min_price": {"type": "number"},
                "max_price": {"type": "number"},
                "min_area": {"type": "number", "description": "Minimum size in m²"},
                "min_year_built": {"type": "integer"},
                "amenities": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(AMENITY_LABELS)},
                    "description": "Required amenities",
                },
                "keywords": {
                    "type": "string",
                    "description": "Free text used to rank results, e.g. 'quiet near park'",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_listing",
        "description": "Get full details of one listing by its ID, e.g. 'P012'.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "calculate_mortgage",
        "description": (
            "Calculate the monthly payment and total interest for a fixed-rate mortgage. "
            "Defaults: 20% down, 6.5% interest, 30 years."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "price": {"type": "number", "description": "Property price in PLN"},
                "down_payment_percent": {"type": "number"},
                "interest_rate": {"type": "number", "description": "Annual rate in percent"},
                "years": {"type": "integer"},
            },
            "required": ["price"],
            "additionalProperties": False,
        },
    },
    {
        "name": "market_stats",
        "description": "Median prices and average price per m² for one city, or all cities if omitted.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "enum": list(CITY_VARIANTS)}},
            "required": [],
            "additionalProperties": False,
        },
    },
]


class ToolRun:
    """Executes tool calls against local data and remembers what they returned, for the UI."""

    def __init__(self, properties: list[Property]):
        self.properties = properties
        self.listings: dict[str, Property] = {}  # every listing a tool returned, in order
        self.mortgage: dict | None = None
        self.market: list[dict] = []

    def execute(self, name: str, args: dict) -> dict:
        """Run one tool. Raises ValueError for bad input; the loop reports it back to Claude."""
        if name == "search_listings":
            filters = {k: v for k, v in args.items() if k not in ("amenities", "keywords")}
            filters.update({a: True for a in args.get("amenities", [])})
            results = search(self.properties, filters, args.get("keywords", ""), limit=8)
            self.listings.update((p.id, p) for p in results)
            return {"count": len(results), "listings": [p.summary() for p in results]}

        if name == "get_listing":
            for p in self.properties:
                if p.id.lower() == args["id"].lower():
                    self.listings[p.id] = p
                    return {"listing": p.summary(), "description": p.description}
            raise ValueError(f"No listing with ID {args['id']}")

        if name == "calculate_mortgage":
            self.mortgage = mortgage(**args).to_dict()
            return self.mortgage

        if name == "market_stats":
            self.market = city_stats(self.properties, args.get("city"))
            return {"cities": self.market}

        raise ValueError(f"Unknown tool {name}")


def run_agent(query: str, properties: list[Property], history: list[dict] | None = None) -> dict | None:
    """Answer `query` by letting Claude call tools. Returns None if the agent couldn't finish,
    so the caller can fall back to pipeline mode."""
    tools = ToolRun(properties)
    steps: list[dict] = []
    messages = llm.clean_history(history or []) + [{"role": "user", "content": query}]

    for _ in range(MAX_ITERATIONS):
        # 1. Ask Claude what to do next, given everything so far.
        try:
            response = llm.get_client().beta.messages.create(
                model=llm.MODEL,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
                output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError:
            log.warning("Claude rate limit hit during agent run")
            return None
        except anthropic.APIStatusError as e:
            log.warning("Claude API error %s during agent run", e.status_code)
            return None
        except anthropic.APIConnectionError:
            log.warning("Could not reach the Claude API during agent run")
            return None

        if response.stop_reason == "refusal":
            return None

        # 2. Record Claude's turn exactly as returned. It can include thinking blocks, which the
        #    API requires back unchanged, so never rebuild or edit earlier turns.
        messages.append({"role": "assistant", "content": response.content})

        # 3. No tool calls means Claude has finished and the text is the answer.
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_calls:
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if not text:
                return None
            return {
                "answer": text,
                "mode": "agent",
                "analysis": {"intent": "agent", "filters": {}, "mortgage": {}},
                "steps": steps,
                "properties": [property_json(p) for p in tools.listings.values()],
                "mortgage": tools.mortgage,
                "market": tools.market,
            }

        # 4. Run every requested tool and send all results back in ONE user message.
        #    Errors go back to Claude as is_error results so it can correct itself.
        results = []
        for call in tool_calls:
            try:
                output, is_error = tools.execute(call.name, call.input), False
            except (ValueError, TypeError) as e:
                output, is_error = {"error": str(e)}, True
            steps.append({"tool": call.name, "input": call.input, "error": is_error})
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(output),
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})

    log.warning("Agent stopped after %d iterations without an answer", MAX_ITERATIONS)
    return None
