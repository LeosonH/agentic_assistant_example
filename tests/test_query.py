import pytest

from realestate.query import analyze, extract_filters, extract_mortgage_params


def test_readme_example():
    f = extract_filters("2-bedroom apartment in Kraków under 500k")
    assert f == {
        "city": "Krakow",
        "property_type": "apartment",
        "rooms": 2,
        "max_price": 500_000,
        "listing_type": "sale",  # inferred from a sale-sized budget
    }


def test_rent_range_and_amenity():
    f = extract_filters("flats for rent in Warsaw between 3000 and 5000 with balcony")
    assert f["listing_type"] == "rent"
    assert (f["min_price"], f["max_price"]) == (3000, 5000)
    assert f["has_balcony"] is True


@pytest.mark.parametrize(
    "query,expected",
    [
        ("house over 1.5m", {"min_price": 1_500_000}),
        ("under 1,200,000", {"max_price": 1_200_000}),
        ("budget 750 thousand", {"max_price": 750_000}),
        ("studio under 2,500 per month", {"max_price": 2_500}),
    ],
)
def test_price_formats(query, expected):
    f = extract_filters(query)
    assert {k: f[k] for k in expected} == expected


def test_small_numbers_are_not_prices():
    f = extract_filters("3 room loft at least 60 m2 built after 2010")
    assert f["rooms"] == 3
    assert f["min_area"] == 60
    assert f["min_year_built"] == 2010
    assert "max_price" not in f and "min_price" not in f


def test_budget_infers_rent():
    assert extract_filters("apartment in Gdansk under 4000")["listing_type"] == "rent"


@pytest.mark.parametrize(
    "query,intent",
    [
        ("Show me apartments in Poznan", "search"),
        ("Krakow under 900k", "search"),
        ("What's my monthly payment on a 600k loan?", "mortgage"),
        ("average price per m2 in Warsaw", "market"),
        ("hello there", "general"),
        ("how does buying property work?", "general"),
    ],
)
def test_intent(query, intent):
    assert analyze(query).intent == intent


def test_mortgage_params():
    p = extract_mortgage_params("mortgage on 750k with 10% down at 6.5% for 25 years")
    assert p == {"price": 750_000, "down_payment_percent": 10, "interest_rate": 6.5, "years": 25}
