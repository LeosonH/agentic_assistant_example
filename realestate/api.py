"""FastAPI app: JSON endpoints plus the single-page UI."""

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import llm
from .agent import run_agent
from .assistant import answer, property_json
from .data import DEFAULT_CSV, load_properties
from .finance import mortgage
from .market import city_stats
from .query import analyze
from .search import search

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Real Estate Assistant (simplified)")
PROPERTIES = load_properties(os.environ.get("REA_DATA_CSV", DEFAULT_CSV))


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = []
    mode: Literal["pipeline", "agent"] = "pipeline"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "properties": len(PROPERTIES), "llm": llm.llm_enabled()}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    history = [m.model_dump() for m in req.history]
    if req.mode == "agent":
        if not llm.llm_enabled():
            raise HTTPException(400, "Agent mode needs Claude: set ANTHROPIC_API_KEY and restart.")
        result = run_agent(req.message, PROPERTIES, history)
        if result:
            return result
        return {**answer(req.message, PROPERTIES, history), "note": "Agent failed; answered in pipeline mode."}
    return answer(req.message, PROPERTIES, history)


@app.get("/api/search")
def search_properties(
    q: str = Query("", max_length=500, description="Natural-language query"),
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    filters = analyze(q).filters
    results = search(PROPERTIES, filters, q, limit=limit)
    return {"filters": filters, "count": len(results), "properties": [property_json(p) for p in results]}


@app.get("/api/properties/{property_id}")
def get_property(property_id: str) -> dict:
    for p in PROPERTIES:
        if p.id.lower() == property_id.lower():
            return property_json(p)
    raise HTTPException(404, "Property not found")


@app.get("/api/mortgage")
def mortgage_calc(
    price: float = Query(gt=0),
    down_payment_percent: float = Query(20.0, ge=0, le=100),
    interest_rate: float = Query(6.5, ge=0, le=30),
    years: int = Query(30, ge=1, le=50),
) -> dict:
    return mortgage(price, down_payment_percent, interest_rate, years).to_dict()


@app.get("/api/market")
def market(city: str | None = None) -> dict:
    return {"cities": city_stats(PROPERTIES, city)}
