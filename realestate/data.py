"""Property model and CSV loader."""

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "properties.csv"

AMENITY_LABELS = {
    "has_parking": "parking",
    "has_garden": "garden",
    "has_pool": "pool",
    "is_furnished": "furnished",
    "has_elevator": "elevator",
    "has_balcony": "balcony",
    "pets_allowed": "pets allowed",
}
AMENITIES = tuple(AMENITY_LABELS)


@dataclass(frozen=True)
class Property:
    id: str
    title: str
    city: str
    district: str
    property_type: str  # apartment | studio | house | loft | townhouse
    listing_type: str  # sale | rent
    rooms: int
    area_sqm: float
    price: float  # sale price, or monthly rent
    currency: str
    year_built: int
    energy_rating: str
    description: str
    has_parking: bool = False
    has_garden: bool = False
    has_pool: bool = False
    is_furnished: bool = False
    has_elevator: bool = False
    has_balcony: bool = False
    pets_allowed: bool = False

    @property
    def price_per_sqm(self) -> float:
        return round(self.price / self.area_sqm, 2) if self.area_sqm else 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "price_per_sqm": self.price_per_sqm}

    def summary(self) -> str:
        """One-line description used as LLM context."""
        unit = "/month" if self.listing_type == "rent" else ""
        extras = ", ".join(label for a, label in AMENITY_LABELS.items() if getattr(self, a))
        return (
            f"[{self.id}] {self.title} — {self.city}/{self.district}, {self.property_type} for "
            f"{self.listing_type}, {self.rooms} rooms, {self.area_sqm:g} m², "
            f"{self.price:,.0f} {self.currency}{unit}, built {self.year_built}, "
            f"energy {self.energy_rating}" + (f"; {extras}" if extras else "")
        )


def _row_to_property(row: dict) -> Property:
    return Property(
        id=row["id"],
        title=row["title"],
        city=row["city"],
        district=row["district"],
        property_type=row["property_type"],
        listing_type=row["listing_type"],
        rooms=int(row["rooms"]),
        area_sqm=float(row["area_sqm"]),
        price=float(row["price"]),
        currency=row.get("currency") or "PLN",
        year_built=int(row["year_built"]),
        energy_rating=row["energy_rating"],
        description=row["description"],
        **{a: row.get(a, "0").strip().lower() in ("1", "true", "yes") for a in AMENITIES},
    )


def load_properties(path: Path | str = DEFAULT_CSV) -> list[Property]:
    with open(path, newline="", encoding="utf-8") as f:
        return [_row_to_property(row) for row in csv.DictReader(f)]
