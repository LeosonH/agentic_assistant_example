"""Natural-language query analysis: intent classification and filter extraction.

Rule-based on purpose: it's instant, free, deterministic, and easy to test.
"""

import re
from dataclasses import dataclass, field

INTENT_KEYWORDS = {
    "mortgage": ["mortgage", "loan", "monthly payment", "down payment", "interest rate", "afford"],
    "market": ["average", "median", "market", "trend", "statistics", "stats", "price per", "cheapest city"],
    "search": [
        "find", "show", "search", "looking for", "list", "apartment", "flat", "house", "studio",
        "loft", "townhouse", "bedroom", "room", "under", "below",
    ],  # rent/buy words are listing_type filters; alone they don't make a search ("how does buying work?")
}

CITY_VARIANTS = {
    "Krakow": ["krakow", "kraków", "cracow"],
    "Warsaw": ["warsaw", "warszawa"],
    "Gdansk": ["gdansk", "gdańsk"],
    "Wroclaw": ["wroclaw", "wrocław"],
    "Poznan": ["poznan", "poznań"],
}

PROPERTY_TYPES = {
    "studio": ["studio"],
    "loft": ["loft"],
    "townhouse": ["townhouse", "town house"],
    "house": ["house", "home with garden", "detached"],
    "apartment": ["apartment", "flat", "condo"],
}

AMENITY_KEYWORDS = {
    "has_parking": ["parking", "garage"],
    "has_garden": ["garden", "yard"],
    "has_pool": ["pool"],
    "is_furnished": ["furnished"],
    "has_elevator": ["elevator", "lift"],
    "has_balcony": ["balcony", "terrace"],
    "pets_allowed": ["pet", "dog", "cat"],
}

ROOMS_RE = re.compile(r"\b(\d+)\s*-?\s*(?:bed(?:room)?s?|rooms?|br)\b")
AREA_RE = re.compile(r"\b(\d+)\s*(?:m2|m²|sqm|sq\s?m|square met(?:er|re)s?)\b")
YEARS_RE = re.compile(r"\b(\d{1,2})\s*-?\s*(?:years?|yrs?)\b")
DOWN_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s*down\b")
YEAR_BUILT_RE = re.compile(r"\b(?:built|constructed|since|after|newer than)\D{0,10}((?:18|19|20)\d{2})\b")
RATE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*%")
MONEY_RE = re.compile(r"(?<![\w.])(?:[$€]|pln\s*)?(\d+(?:[.,]\d+)*)\s*(k|m|thousand|million)?(?![\w%])")
BETWEEN_RE = re.compile(r"\bbetween\b(.+?)\band\b(.+)")
RENT_CEILING = 50_000  # budgets at or above this are sale prices; below, monthly rents
MAX_WORDS = ("under", "below", "less than", "up to", "max", "at most", "cheaper than", "within")
MIN_WORDS = ("over", "above", "more than", "at least", "min", "from", "starting at")


@dataclass
class QueryAnalysis:
    query: str
    intent: str  # search | mortgage | market | general
    filters: dict = field(default_factory=dict)
    mortgage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"intent": self.intent, "filters": self.filters, "mortgage": self.mortgage}


def _has(text: str, phrase: str) -> bool:
    return re.search(r"\b" + re.escape(phrase), text) is not None


def _parse_money(number: str, suffix: str | None) -> float | None:
    # "1,200,000" / "1.200.000" are thousands separators; "1.5" / "1,5" are decimals.
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", number):
        value = float(re.sub(r"[.,]", "", number))
    else:
        value = float(number.replace(",", "."))
    if suffix in ("k", "thousand"):
        value *= 1_000
    elif suffix in ("m", "million"):
        value *= 1_000_000
    elif value < 1_000:  # bare small numbers are counts, floors, etc., not prices
        return None
    return value


def _find_prices(text: str) -> list[tuple[int, float]]:
    """Return (position, value) for every money-like number in text."""
    found = []
    for m in MONEY_RE.finditer(text):
        value = _parse_money(m.group(1), m.group(2))
        if value is not None:
            found.append((m.start(), value))
    return found


def _extract_price_range(text: str) -> dict:
    between = BETWEEN_RE.search(text)
    if between:
        low, high = _find_prices(between.group(1)), _find_prices(between.group(2))
        if low and high:
            return {"min_price": low[0][1], "max_price": high[0][1]}

    result = {}
    for pos, value in _find_prices(text):
        before = text[max(0, pos - 15) : pos]
        if any(w in before for w in MIN_WORDS):
            result["min_price"] = value
        else:  # "under 500k", or an unqualified "500k" — treated as a budget ceiling
            result["max_price"] = value
    return result


def extract_filters(query: str) -> dict:
    text = query.lower()
    filters: dict = {}

    for canonical, variants in CITY_VARIANTS.items():
        if any(_has(text, v) for v in variants):
            filters["city"] = canonical
            break

    for ptype, words in PROPERTY_TYPES.items():
        if any(_has(text, w) for w in words):
            filters["property_type"] = ptype
            break

    if any(_has(text, w) for w in ("rent", "rental", "to let", "per month", "/month", "monthly rent")):
        filters["listing_type"] = "rent"
    elif any(_has(text, w) for w in ("buy", "purchase", "for sale", "to own")):
        filters["listing_type"] = "sale"

    if m := ROOMS_RE.search(text):
        filters["rooms"] = int(m.group(1))
    if m := AREA_RE.search(text):
        filters["min_area"] = float(m.group(1))
    if m := YEAR_BUILT_RE.search(text):
        filters["min_year_built"] = int(m.group(1))

    # Strip rooms/area/year so "2-bedroom", "60 m2" or "built 2010" are never read as prices.
    price_text = YEAR_BUILT_RE.sub(" ", AREA_RE.sub(" ", ROOMS_RE.sub(" ", text)))
    filters.update(_extract_price_range(price_text))

    # A budget implies the listing type: nobody rents for 500k a month or buys for 3k.
    if "listing_type" not in filters:
        budget = filters.get("max_price") or filters.get("min_price")
        if budget:
            filters["listing_type"] = "sale" if budget >= RENT_CEILING else "rent"

    for amenity, words in AMENITY_KEYWORDS.items():
        if any(_has(text, w) for w in words):
            filters[amenity] = True

    return filters


def extract_mortgage_params(query: str) -> dict:
    text = query.lower()
    params: dict = {}
    if m := DOWN_RE.search(text):
        params["down_payment_percent"] = float(m.group(1))
        text = text.replace(m.group(0), " ")
    if m := RATE_RE.search(text):
        params["interest_rate"] = float(m.group(1))
    if m := YEARS_RE.search(text):
        params["years"] = int(m.group(1))
    prices = _find_prices(YEARS_RE.sub(" ", text))
    if prices:
        params["price"] = prices[0][1]
    return params


def classify_intent(query: str) -> str:
    text = query.lower()
    scores = {
        intent: sum(1 for kw in keywords if _has(text, kw))
        for intent, keywords in INTENT_KEYWORDS.items()
    }
    # More specific intents win ties over generic search.
    for intent in ("mortgage", "market", "search"):
        if scores[intent] and scores[intent] == max(scores.values()):
            return intent
    return "general"


def analyze(query: str) -> QueryAnalysis:
    intent = classify_intent(query)
    filters = extract_filters(query)
    if intent == "general" and set(filters) - {"listing_type"}:
        intent = "search"  # "Krakow under 500k" has no verb, but is clearly a search
    mortgage = extract_mortgage_params(query) if intent == "mortgage" else {}
    return QueryAnalysis(query=query, intent=intent, filters=filters, mortgage=mortgage)
