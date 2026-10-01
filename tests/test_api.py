import pytest
from fastapi.testclient import TestClient

from realestate.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("REA_LLM", "off")


def test_health():
    assert client.get("/api/health").json() == {"status": "ok", "properties": 150, "llm": False}


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text


def test_chat():
    r = client.post(
        "/api/chat",
        json={"message": "flats for rent in Warsaw", "history": [{"role": "assistant", "content": "hi"}]},
    )
    body = r.json()
    assert body["analysis"]["intent"] == "search"
    assert body["properties"] and all(p["city"] == "Warsaw" for p in body["properties"])


def test_search_endpoint():
    body = client.get("/api/search", params={"q": "house in Gdansk with garden"}).json()
    assert body["filters"]["property_type"] == "house"
    assert all(p["has_garden"] for p in body["properties"])


def test_property_lookup():
    assert client.get("/api/properties/p001").json()["id"] == "P001"
    assert client.get("/api/properties/nope").status_code == 404


def test_mortgage_endpoint_validates():
    assert client.get("/api/mortgage", params={"price": 500000}).json()["loan_amount"] == 400000
    assert client.get("/api/mortgage", params={"price": -5}).status_code == 422


def test_market_endpoint():
    assert len(client.get("/api/market").json()["cities"]) == 5
