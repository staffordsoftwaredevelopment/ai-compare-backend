import os
import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel


# ---------- Env ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-2-latest")
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar")

SHORTCUT_TOKEN = os.getenv("SHORTCUT_TOKEN", "")  # your shared secret token for iPhone Shortcut

SYNTH_MODEL = os.getenv("SYNTH_MODEL", OPENAI_MODEL)  # reuse OpenAI by default for synthesis

app = FastAPI(title="AI Compare Backend")


class CompareRequest(BaseModel):
    query: str
    system: Optional[str] = "Answer clearly. If you assume anything, say so."
    max_tokens: int = 500


class ProviderResult(BaseModel):
    provider: str
    model: str
    text: str = ""
    error: Optional[str] = None
    meta: Dict[str, Any] = {}


@app.get("/healthz")
def healthz():
    return {"ok": True}


def require_auth(auth_header: Optional[str]) -> None:
    # If SHORTCUT_TOKEN is set, require Bearer token
    if not SHORTCUT_TOKEN:
        return
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization")
    token = auth_header.split(" ", 1)[1].strip()
    if token != SHORTCUT_TOKEN:
        raise HTTPException(status_code=403, detail="Bad token")


# ----------------------------
# Provider callers
# ----------------------------
async def call_openai(query: str, system: str, max_tokens: int) -> ProviderResult:
    if not OPENAI_API_KEY:
        return ProviderResult(provider="openai", model=OPENAI_MODEL, error="Missing OPENAI_API_KEY")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": query}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            return ProviderResult(provider="openai", model=OPENAI_MODEL, text=text, meta={"id": data.get("id")})
    except Exception as e:
        return ProviderResult(provider="openai", model=OPENAI_MODEL, error=str(e))


async def call_anthropic(query: str, system: str, max_tokens: int) -> ProviderResult:
    if not ANTHROPIC_API_KEY:
        return ProviderResult(provider="anthropic", model=ANTHROPIC_MODEL, error="Missing ANTHROPIC_API_KEY")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            parts = []
            for block in data.get("content", []):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return ProviderResult(
                provider="anthropic",
                model=ANTHROPIC_MODEL,
                text="\n".join(parts).strip(),
                meta={"id": data.get("id")},
            )
    except Exception as e:
        return ProviderResult(provider="anthropic", model=ANTHROPIC_MODEL, error=str(e))


async def call_xai(query: str, system: str, max_tokens: int) -> ProviderResult:
    if not XAI_API_KEY:
        return ProviderResult(provider="xai", model=XAI_MODEL, error="Missing XAI_API_KEY")

    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}"}
    payload = {
        "model": XAI_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": query}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            return ProviderResult(provider="xai", model=XAI_MODEL, text=text, meta={"id": data.get("id")})
    except Exception as e:
        return ProviderResult(provider="xai", model=XAI_MODEL, error=str(e))


async def call_perplexity(query: str, system: str, max_tokens: int) -> ProviderResult:
    if not PERPLEXITY_API_KEY:
        return ProviderResult(provider="perplexity", model=PERPLEXITY_MODEL, error="Missing PERPLEXITY_API_KEY")

    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}"}
    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": query}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            return ProviderResult(provider="perplexity", model=PERPLEXITY_MODEL, text=text, meta={"id": data.get("id")})
    except Exception as e:
        return ProviderResult(provider="perplexity", model=PERPLEXITY_MODEL, error=str(e))


# ----------------------------
# Synthesis (JSON enforced + retry + time sensitivity + confidence per claim)
# ----------------------------
async def synthesize_consensus(query: str, results: List[ProviderResult], max_tokens: int = 900) -> Dict[str, Any]:
    server_utc = datetime.now(timezone.utc).isoformat()

    def safe_text(r: ProviderResult) -> str:
        if r.error:
            return f"[ERROR] {r.error}"
        return (r.text or "").strip()

    bundle = "\n".join([f"### {r.provider.upper()} ({r.model})\n{safe_text(r)}\n" for r in results])

    system_prompt = (
        "You are a neutral synthesis engine.\n\n"
        "Rules:\n"
        "- Use ONLY the provided model outputs. Do NOT add external knowledge.\n"
        "- Ignore errors/empty outputs.\n"
        "- Extract atomic factual claims for consensus.\n"
        "- Detect time sensitivity:\n"
        "  HIGH: markets day performance, live sports, breaking news, weather, schedules, prices, or 'today/yesterday/Friday'\n"
        "  MEDIUM: roles, laws, availability, product pricing/specs\n"
        "  LOW: definitions/history/math/stable facts\n"
        "- Confidence scoring per claim (0..1):\n"
        "  support = (# providers that clearly state the claim) / (total non-error providers)\n"
        "  penalties:\n"
        "    -0.15 if numbers/dates differ materially across providers for this claim\n"
        "    -0.10 if claim is hedged ('about/around/roughly') but is a precise value\n"
        "    -0.10 if time-sensitive and no as-of timestamp/date is stated\n"
        "  confidence = clamp(support - penalties, 0, 1)\n"
        "- Disagreements must be factual only (numbers/dates/contradictions), not style.\n"
        "- Output STRICT JSON ONLY. No markdown. No commentary.\n"
        "- If you cannot comply, output {}.\n"
    )

    user_prompt = (
        f"USER QUESTION:\n{query}\n\n"
        f"AS_OF_SERVER_TIME_UTC:\n{server_utc}\n\n"
        f"MODEL OUTPUTS:\n{bundle}\n\n"
        "Return JSON EXACTLY matching this structure:\n"
        "{\n"
        '  "time_sensitivity": {"label":"low|medium|high","reason":"...","as_of":"..."},\n'
        '  "consensus": {\n'
        '    "answer":"...",\n'
        '    "claims":[\n'
        '      {"id":"c1","claim":"...","confidence":0.0,\n'
        '       "evidence_by_provider":{"openai":"","anthropic":"","xai":"","perplexity":""},\n'
        '       "notes":""}\n'
        "    ]\n"
        "  },\n"
        '  "disagreements":[\n'
        '    {"topic":"...","conflicting_claims":[{"claim":"...","providers":["openai"],"confidence":0.0}],"why_disagrees":"..."}\n'
        "  ],\n"
        '  "notable_missing":["..."]\n'
        "}\n"
    )

    async def call_and_parse() -> Dict[str, Any]:
        if not OPENAI_API_KEY:
            return {
                "time_sensitivity": {"label": "unknown", "reason": "Missing OPENAI_API_KEY for synthesis", "as_of": server_utc},
                "consensus": {"answer": "", "claims": []},
                "disagreements": [],
                "notable_missing": ["Synthesis unavailable: missing OPENAI_API_KEY"],
            }

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = {
            "model": SYNTH_MODEL,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            return json.loads(content)

    try:
        return await call_and_parse()
    except Exception:
        try:
            return await call_and_parse()
        except Exception:
            return {
                "time_sensitivity": {"label": "unknown", "reason": "Synthesis failed after retry", "as_of": server_utc},
                "consensus": {"answer": "", "claims": []},
                "disagreements": [],
                "notable_missing": ["Synthesis failed: model returned malformed JSON twice"],
            }


# ----------------------------
# API
# ----------------------------
@app.post("/compare")
async def compare(req: CompareRequest, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)

    tasks = [
        call_openai(req.query, req.system or "", req.max_tokens),
        call_anthropic(req.query, req.system or "", req.max_tokens),
        call_xai(req.query, req.system or "", req.max_tokens),
        call_perplexity(req.query, req.system or "", req.max_tokens),
    ]
    results = await asyncio.gather(*tasks)

    synthesis = await synthesize_consensus(req.query, results)

    return {
        "query": req.query,
        "time_sensitivity": synthesis.get("time_sensitivity", {}),
        "consensus": synthesis.get("consensus", {}),
        "disagreements": synthesis.get("disagreements", []),
        "notable_missing": synthesis.get("notable_missing", []),
        "results": [r.model_dump() for r in results],
    }
