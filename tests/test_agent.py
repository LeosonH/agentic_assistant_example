import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from realestate import agent, llm
from realestate.api import app
from realestate.data import load_properties

PROPERTIES = load_properties()


def tool_use(id, name, input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def text(t):
    return SimpleNamespace(type="text", text=t)


def turn(*blocks, stop_reason=None):
    if stop_reason is None:
        stop_reason = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


class ScriptedClaude:
    """Replays a fixed list of responses and records every request."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def create(self, **kwargs):
        # Snapshot the messages: the agent keeps appending to the same list.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


@pytest.fixture
def install(monkeypatch):
    monkeypatch.setenv("REA_LLM", "on")

    def _install(scripted):
        client = SimpleNamespace(beta=SimpleNamespace(messages=scripted))
        monkeypatch.setattr(llm, "get_client", lambda: client)
        return scripted

    return _install


def test_multi_step_loop(install):
    first = turn(tool_use("t1", "search_listings", {"city": "Krakow", "rooms": 2, "listing_type": "sale"}))
    second = turn(tool_use("t2", "calculate_mortgage", {"price": 727000}))
    final = turn(text("The cheapest is [P129]; about 3,676 PLN/month."))
    claude = install(ScriptedClaude(first, second, final))

    result = agent.run_agent("2-room flat in Krakow to buy, and the mortgage on the cheapest", PROPERTIES)

    assert result["mode"] == "agent"
    assert result["answer"].startswith("The cheapest is [P129]")
    assert [s["tool"] for s in result["steps"]] == ["search_listings", "calculate_mortgage"]
    assert result["mortgage"]["loan_amount"] == 581_600
    assert all(p["city"] == "Krakow" and p["rooms"] == 2 for p in result["properties"])

    # Second request carries Claude's first turn unchanged, then the matching tool result.
    sent = claude.calls[1]["messages"]
    assert sent[-2] == {"role": "assistant", "content": first.content}
    tool_result = sent[-1]["content"][0]
    assert tool_result["tool_use_id"] == "t1" and tool_result["is_error"] is False
    assert json.loads(tool_result["content"])["count"] == len(result["properties"])


def test_tool_errors_go_back_to_claude(install):
    claude = install(
        ScriptedClaude(
            turn(tool_use("t1", "calculate_mortgage", {"price": -5}), tool_use("t2", "get_listing", {"id": "P999"})),
            turn(text("Sorry, I need a valid price.")),
        )
    )
    result = agent.run_agent("mortgage on minus five", PROPERTIES)

    assert result["answer"] == "Sorry, I need a valid price."
    results = claude.calls[1]["messages"][-1]["content"]
    assert len(results) == 2  # both results in one user message
    assert all(r["is_error"] for r in results)


def test_loop_has_a_safety_stop(install):
    endless = [turn(tool_use(f"t{i}", "market_stats", {})) for i in range(agent.MAX_ITERATIONS)]
    claude = install(ScriptedClaude(*endless))
    assert agent.run_agent("stats forever", PROPERTIES) is None
    assert len(claude.calls) == agent.MAX_ITERATIONS


def test_refusal_returns_none(install):
    install(ScriptedClaude(turn(text(""), stop_reason="refusal")))
    assert agent.run_agent("anything", PROPERTIES) is None


def test_api_agent_mode_needs_llm(monkeypatch):
    monkeypatch.setenv("REA_LLM", "off")
    r = TestClient(app).post("/api/chat", json={"message": "hi", "mode": "agent"})
    assert r.status_code == 400


def test_api_agent_failure_falls_back_to_pipeline(install, monkeypatch):
    install(ScriptedClaude(turn(text(""), stop_reason="refusal"), turn(text(""), stop_reason="refusal")))
    body = TestClient(app).post("/api/chat", json={"message": "flats in Warsaw", "mode": "agent"}).json()
    assert body["note"].startswith("Agent failed")
    assert body["analysis"]["intent"] == "search"
