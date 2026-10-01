"""Chat orchestration: analyze the query, gather facts with local tools, then answer.

Facts (listings, mortgage numbers, market stats) are always computed locally. Claude only
phrases the answer from those facts; without Claude, a template answer is used instead.
"""

import re

from . import llm
from .data import AMENITY_LABELS, Property
from .finance import mortgage
from .market import city_stats
from .query import QueryAnalysis, analyze
from .search import search

SYSTEM_PROMPT = """You are a friendly real estate assistant for listings in Polish cities \
(Krakow, Warsaw, Gdansk, Wroclaw, Poznan). Prices are in PLN; rent prices are per month.

Answer from the facts in the <context> block of the user's message. Do not invent listings, \
prices, or features. Refer to listings by their [ID]. If nothing matches, say so and suggest \
which filter to relax. Mortgage figures are estimates, not a lending offer. Keep answers short \
and scannable."""

RESULT_LIMIT = 6
PROPERTY_ID_RE = re.compile(r"\bP\d{3}\b", re.IGNORECASE)


def property_json(p: Property) -> dict:
    """Listing as JSON, with an indicative monthly mortgage payment for sale listings."""
    data = p.to_dict()
    if p.listing_type == "sale":
        data["monthly_payment_estimate"] = round(mortgage(p.price).monthly_payment)
    return data


def describe_filters(filters: dict) -> str:
    parts = []
    if "rooms" in filters:
        parts.append(f"{filters['rooms']}-room")
    parts.append(filters.get("property_type", "listings"))
    if "listing_type" in filters:
        parts.append("for " + filters["listing_type"])
    if "city" in filters:
        parts.append("in " + filters["city"])
    if "min_price" in filters:
        parts.append(f"from {filters['min_price']:,.0f} PLN")
    if "max_price" in filters:
        parts.append(f"up to {filters['max_price']:,.0f} PLN")
    if "min_area" in filters:
        parts.append(f"at least {filters['min_area']:g} m²")
    if "min_year_built" in filters:
        parts.append(f"built {filters['min_year_built']} or later")
    amenities = [AMENITY_LABELS[k] for k in AMENITY_LABELS if filters.get(k)]
    if amenities:
        parts.append("with " + ", ".join(amenities))
    return " ".join(parts)


def _fmt(n: float) -> str:
    return f"{n:,.0f}"


def _search_reply(analysis: QueryAnalysis, results: list[Property]) -> str:
    desc = describe_filters(analysis.filters)
    if not results:
        return f"No {desc} found. Try raising the budget or removing a filter."
    lines = [f"Found {len(results)} {'match' if len(results) == 1 else 'matches'} for {desc}:"]
    for p in results:
        unit = " /month" if p.listing_type == "rent" else ""
        lines.append(f"- [{p.id}] {p.title}, {p.city} — {_fmt(p.price)} PLN{unit}, {p.area_sqm:g} m²")
    return "\n".join(lines)


def _mortgage_reply(m: dict | None) -> str:
    if m is None:
        return "Tell me the property price, e.g. “mortgage on 800k at 6% over 25 years with 10% down”."
    return (
        f"For a {_fmt(m['price'])} PLN property with {_fmt(m['down_payment'])} PLN down "
        f"({100 * m['down_payment'] / m['price']:.0f}%), at {m['interest_rate']}% "
        f"over {m['years']} years, the monthly payment is about {_fmt(m['monthly_payment'])} PLN. "
        f"Total interest: {_fmt(m['total_interest'])} PLN. (Estimate, not a lending offer.)"
    )


def _market_reply(stats: list[dict]) -> str:
    lines = ["Market snapshot:"]
    for s in stats:
        sale = f"median sale {_fmt(s['median_sale_price'])} PLN ({_fmt(s['avg_sale_price_per_sqm'])} PLN/m²)" if s["median_sale_price"] else "no sales"
        rent = f"median rent {_fmt(s['median_rent'])} PLN/month" if s["median_rent"] else "no rentals"
        lines.append(f"- {s['city']}: {sale}; {rent}; {s['listings']} listings")
    return "\n".join(lines)


GENERAL_REPLY = (
    "I can search listings, estimate mortgage payments, and summarize market prices. Try:\n"
    "- “2-bedroom apartment in Krakow under 900k”\n"
    "- “mortgage on 750k with 10% down at 6% for 25 years”\n"
    "- “average price per m2 in Warsaw”"
)


def answer(query: str, properties: list[Property], history: list[dict] | None = None) -> dict:
    analysis = analyze(query)
    results: list[Property] = []
    mortgage_result: dict | None = None
    stats: list[dict] = []

    # Follow-ups like "tell me more about P012" pin specific listings.
    mentioned_ids = {m.upper() for m in PROPERTY_ID_RE.findall(query)}
    if mentioned_ids:
        results = [p for p in properties if p.id in mentioned_ids]

    if analysis.intent == "search" and not results:
        results = search(properties, analysis.filters, query, limit=RESULT_LIMIT)
        fallback = _search_reply(analysis, results)
    elif analysis.intent == "mortgage":
        params = dict(analysis.mortgage)
        if "price" not in params and len(results) == 1:
            params["price"] = results[0].price
        if "price" in params:
            mortgage_result = mortgage(**params).to_dict()
        fallback = _mortgage_reply(mortgage_result)
    elif analysis.intent == "market":
        stats = city_stats(properties, analysis.filters.get("city"))
        fallback = _market_reply(stats)
    elif results:
        fallback = "\n".join(p.summary() for p in results)
    else:
        fallback = GENERAL_REPLY

    context = [f"Detected intent: {analysis.intent}; filters: {analysis.filters or 'none'}"]
    if results:
        context.append("Matching listings:\n" + "\n".join(p.summary() for p in results))
    elif analysis.intent == "search":
        context.append("No listings matched these filters.")
    if mortgage_result:
        context.append(f"Mortgage calculation: {mortgage_result}")
    if stats:
        context.append(f"Market statistics: {stats}")

    reply = llm.complete(
        SYSTEM_PROMPT,
        f"<context>\n{chr(10).join(context)}\n</context>\n\n{query}",
        history,
    )

    return {
        "answer": reply or fallback,
        "mode": "llm" if reply else "offline",
        "analysis": analysis.to_dict(),
        "properties": [property_json(p) for p in results],
        "mortgage": mortgage_result,
        "market": stats,
    }
