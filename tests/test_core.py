import pytest

from realestate.assistant import answer
from realestate.data import load_properties
from realestate.finance import mortgage
from realestate.market import city_stats
from realestate.search import search

PROPERTIES = load_properties()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("REA_LLM", "off")


def test_dataset_loads():
    assert len(PROPERTIES) == 150
    assert {p.city for p in PROPERTIES} == {"Krakow", "Warsaw", "Gdansk", "Wroclaw", "Poznan"}


def test_search_applies_every_filter():
    filters = {"city": "Krakow", "listing_type": "sale", "max_price": 1_000_000, "has_balcony": True}
    results = search(PROPERTIES, filters, limit=100)
    assert results
    for p in results:
        assert p.city == "Krakow" and p.listing_type == "sale"
        assert p.price <= 1_000_000 and p.has_balcony


def test_search_ranks_keyword_matches_first():
    results = search(PROPERTIES, {"city": "Krakow"}, "near the river", limit=3)
    assert "river" in results[0].description


def test_mortgage_known_value():
    # 400k loan, 6% APR, 30y -> 2,398.20/month (standard amortization table value)
    m = mortgage(500_000, down_payment_percent=20, interest_rate=6, years=30)
    assert m.loan_amount == 400_000
    assert m.monthly_payment == pytest.approx(2398.20, abs=0.01)


def test_mortgage_zero_rate():
    assert mortgage(120_000, 0, 0, 10).monthly_payment == pytest.approx(1000)


def test_mortgage_rejects_bad_input():
    with pytest.raises(ValueError):
        mortgage(-1)


def test_city_stats():
    stats = city_stats(PROPERTIES, "Warsaw")
    assert len(stats) == 1 and stats[0]["city"] == "Warsaw"
    assert stats[0]["avg_sale_price_per_sqm"] > stats[0]["avg_rent_per_sqm"]


def test_answer_search_offline():
    r = answer("2-bedroom apartment in Krakow under 900k", PROPERTIES)
    assert r["mode"] == "offline"
    assert r["properties"]
    assert all(p["listing_type"] == "sale" and p["price"] <= 900_000 for p in r["properties"])
    assert all("monthly_payment_estimate" in p for p in r["properties"])


def test_answer_mortgage_offline():
    r = answer("mortgage on 750k with 10% down at 6% for 25 years", PROPERTIES)
    assert r["mortgage"]["loan_amount"] == 675_000
    assert "4,349" in r["answer"]


def test_answer_follow_up_by_id():
    r = answer("tell me more about P001", PROPERTIES)
    assert [p["id"] for p in r["properties"]] == ["P001"]
