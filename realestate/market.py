"""Per-city market statistics."""

from statistics import mean, median

from .data import Property


def city_stats(properties: list[Property], city: str | None = None) -> list[dict]:
    cities = sorted({p.city for p in properties if city is None or p.city == city})
    stats = []
    for c in cities:
        sale = [p for p in properties if p.city == c and p.listing_type == "sale"]
        rent = [p for p in properties if p.city == c and p.listing_type == "rent"]
        stats.append(
            {
                "city": c,
                "listings": len(sale) + len(rent),
                "for_sale": len(sale),
                "for_rent": len(rent),
                "median_sale_price": round(median(p.price for p in sale)) if sale else None,
                "avg_sale_price_per_sqm": round(mean(p.price_per_sqm for p in sale)) if sale else None,
                "median_rent": round(median(p.price for p in rent)) if rent else None,
                "avg_rent_per_sqm": round(mean(p.price_per_sqm for p in rent), 1) if rent else None,
            }
        )
    return stats
