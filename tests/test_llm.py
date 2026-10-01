from types import SimpleNamespace

import anthropic
import httpx
import pytest

from realestate import llm
from realestate.assistant import answer
from realestate.data import load_properties

PROPERTIES = load_properties()


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def install_fake(monkeypatch, messages: FakeMessages):
    monkeypatch.setenv("REA_LLM", "on")
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    monkeypatch.setattr(llm, "get_client", lambda: client)


def text_response(text, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


def test_llm_answer_is_used_and_grounded(monkeypatch):
    fake = FakeMessages(response=text_response("P129 looks great."))
    install_fake(monkeypatch, fake)

    r = answer("2-bedroom apartment in Krakow under 900k", PROPERTIES, [{"role": "assistant", "content": "hi"}])

    assert (r["mode"], r["answer"]) == ("llm", "P129 looks great.")
    sent = fake.calls[0]
    assert sent["model"] == "claude-opus-5-5"
    assert "[P129]" in sent["messages"][-1]["content"]  # listings are passed as context
    assert sent["messages"][0]["role"] == "user"  # leading assistant turn dropped


@pytest.mark.parametrize(
    "fake",
    [
        FakeMessages(error=anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))),
        FakeMessages(response=text_response("", stop_reason="refusal")),
    ],
)
def test_falls_back_to_offline_answer(monkeypatch, fake):
    install_fake(monkeypatch, fake)
    r = answer("average price per m2 in Warsaw", PROPERTIES)
    assert r["mode"] == "offline"
    assert r["answer"].startswith("Market snapshot")
