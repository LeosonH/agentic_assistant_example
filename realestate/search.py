"""Structured filtering plus lightweight keyword ranking over in-memory listings."""

import re

from .data import AMENITIES, Property

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "for", "with", "and", "or", "to", "of", "me", "i",
    "show", "find", "looking", "want", "need", "some", "any", "under", "over", "below", "above",
}


def matches(p: Property, filters: dict) -> bool:
    if "city" in filters and p.city != filters["city"]:
        return False
    if "property_type" in filters and p.property_type != filters["property_type"]:
        return False
    if "listing_type" in filters and p.listing_type != filters["listing_type"]:
        return False
    if "rooms" in filters and p.rooms != filters["rooms"]:
        return False
    if "min_area" in filters and p.area_sqm < filters["min_area"]:
        return False
    if "min_year_built" in filters and p.year_built < filters["min_year_built"]:
        return False
    if "min_price" in filters and p.price < filters["min_price"]:
        return False
    if "max_price" in filters and p.price > filters["max_price"]:
        return False
    return all(getattr(p, a) for a in AMENITIES if filters.get(a))


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-ząćęłńóśźż]+", text.lower()) if t not in STOPWORDS and len(t) > 2}


def search(
    properties: list[Property],
    filters: dict | None = None,
    text: str = "",
    limit: int = 10,
) -> list[Property]:
    """Hard-filter by structured criteria, then rank by keyword overlap with the query text."""
    candidates = [p for p in properties if matches(p, filters or {})]
    query_tokens = _tokens(text)

    def score(p: Property) -> tuple[int, float]:
        overlap = len(query_tokens & _tokens(f"{p.title} {p.description} {p.district}"))
        return (-overlap, p.price)  # best keyword match first, then cheapest

    return sorted(candidates, key=score)[:limit]
