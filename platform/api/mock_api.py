"""
Mock API for the leaderboard dashboard.

Serves hardcoded data with the shape the real API will have once the repository
layer is wired up. Everything is in-memory; no database, no persistence, no
authentication beyond a per-team bearer token that the lecturer would have
handed out through the admin SPA.

Run (from the repo root):
    uvicorn --app-dir platform api.mock_api:app --reload --port 8000

Then open http://127.0.0.1:8000/ for the dashboard SPA.

Note: `--app-dir platform` is required because the directory name `platform`
collides with the Python stdlib module of the same name. We deliberately do
*not* make `platform/` a package for that reason.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Hardcoded mock data
# ---------------------------------------------------------------------------
#
# Shape mirrors what the repository interface will return once wired up:
#   teams   -> team_id, display_name, token (bearer credential)
#   models  -> model_id, owner_team_id, endpoint_url, ready_for_events
#   probes  -> model_id, timestamp (iso8601 UTC), reachable, correct
#
# There is no separate `model.display_name`. The endpoint URL is the model's
# identity on the leaderboard (rendered small under the team name).

_TEAMS = [
    {"team_id": "team-01", "display_name": "Team Alpha",   "token": "tok-alpha-a1b2"},
    {"team_id": "team-02", "display_name": "Team Bravo",   "token": "tok-bravo-c3d4"},
    {"team_id": "team-03", "display_name": "Team Charlie", "token": "tok-charlie-e5f6"},
    {"team_id": "team-04", "display_name": "Team Delta",   "token": "tok-delta-g7h8"},
    {"team_id": "team-05", "display_name": "Team Echo",    "token": "tok-echo-i9j0"},
    # Foxtrot exists in the registry but has not registered a model yet — used
    # to demo the register-your-model flow at the bottom of the dashboard.
    {"team_id": "team-06", "display_name": "Team Foxtrot", "token": "tok-foxtrot-k1l2"},
]

_MODELS = [
    {"model_id": "mdl-alpha-01",   "owner_team_id": "team-01", "endpoint_url": "https://alpha.students.example/infer",   "ready_for_events": True},
    {"model_id": "mdl-bravo-01",   "owner_team_id": "team-02", "endpoint_url": "https://bravo.students.example/infer",   "ready_for_events": True},
    {"model_id": "mdl-charlie-01", "owner_team_id": "team-03", "endpoint_url": "https://charlie.students.example/infer", "ready_for_events": True},
    {"model_id": "mdl-delta-01",   "owner_team_id": "team-04", "endpoint_url": "https://delta.students.example/infer",   "ready_for_events": True},
    {"model_id": "mdl-echo-01",   "owner_team_id": "team-05", "endpoint_url": "https://echo.students.example/infer",   "ready_for_events": False},
]


def _synth_probes(model_id: str, seed: int, n: int = 90) -> list[dict]:
    """Deterministic hand-tuned probe series per model.

    Uses a tiny LCG so this file has no random dependency and screenshots are
    reproducible. Different seeds produce different failure patterns.
    """
    probes: list[dict] = []
    now = datetime.now(timezone.utc).replace(microsecond=0)
    state = seed or 1
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        r = state % 100

        if seed == 5:  # echo — mostly red; not deployed yet
            reachable = r > 70
            correct = None if not reachable else r > 90
        elif seed == 3:  # charlie — clean run then a drift dip and recovery
            in_dip = 30 <= i <= 42
            reachable = r > 3
            if not reachable:
                correct = None
            elif in_dip:
                correct = r > 55
            else:
                correct = r > 8
        elif seed == 4:  # delta — occasional flakes
            reachable = r > 12
            correct = None if not reachable else r > 18
        else:  # alpha, bravo — mostly healthy
            reachable = r > 4
            correct = None if not reachable else r > 6

        ts = now - timedelta(minutes=(n - i) * 5)
        probes.append(
            {
                "model_id": model_id,
                "timestamp": ts.isoformat(),
                "reachable": bool(reachable),
                "correct": (None if correct is None else bool(correct)),
            }
        )
    return probes


_PROBES: dict[str, list[dict]] = {
    "mdl-alpha-01":   _synth_probes("mdl-alpha-01",   seed=1),
    "mdl-bravo-01":   _synth_probes("mdl-bravo-01",   seed=2),
    "mdl-charlie-01": _synth_probes("mdl-charlie-01", seed=3),
    "mdl-delta-01":   _synth_probes("mdl-delta-01",   seed=4),
    "mdl-echo-01":    _synth_probes("mdl-echo-01",    seed=5),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _team_by_token(token: str) -> Optional[dict]:
    token = (token or "").strip()
    if not token:
        return None
    for t in _TEAMS:
        if t["token"] == token:
            return t
    return None


def _model_owned_by(team_id: str) -> Optional[dict]:
    for m in _MODELS:
        if m["owner_team_id"] == team_id:
            return m
    return None


def _aggregate(probes: list[dict]) -> dict:
    if not probes:
        return {
            "correctness_pct": None,
            "uptime_pct": None,
            "current_status": "pending",
            "last_probe_at": None,
        }
    reachable_ct = sum(1 for p in probes if p["reachable"])
    correct_ct = sum(1 for p in probes if p["correct"] is True)
    latest = probes[-1]
    if not latest["reachable"]:
        current = "unreachable"
    elif latest["correct"] is False:
        current = "degraded"
    else:
        current = "operational"
    return {
        "correctness_pct": round(100.0 * correct_ct / len(probes), 2),
        "uptime_pct":      round(100.0 * reachable_ct / len(probes), 2),
        "current_status":  current,
        "last_probe_at":   latest["timestamp"],
    }


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

app = FastAPI(title="ITU SDSE v2 · lecturer-tooling · mock API")


@app.get("/api/dashboard")
def dashboard() -> dict:
    """One-shot payload the SPA renders. Composes teams + models + probes."""
    teams_by_id = {t["team_id"]: t for t in _TEAMS}
    rows = []
    for m in _MODELS:
        probes = _PROBES.get(m["model_id"], []) if m["ready_for_events"] else []
        rows.append(
            {
                "model_id":         m["model_id"],
                "team_name":        teams_by_id[m["owner_team_id"]]["display_name"],
                "endpoint_url":     m["endpoint_url"],
                "ready_for_events": m["ready_for_events"],
                "probes":           [
                    {"reachable": p["reachable"], "correct": p["correct"]}
                    for p in probes
                ],
                **_aggregate(probes),
            }
        )
    return {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "probe_window":  90,
        "models":        rows,
    }


@app.post("/api/models", status_code=201)
def register_model(payload: dict = Body(...)) -> dict:
    """Register a team's inference endpoint.

    Body: {"token": "...", "endpoint_url": "https://..."}
    One model per team in v1 — teams edit their existing model rather than
    registering multiple.
    """
    team = _team_by_token(payload.get("token", ""))
    if not team:
        raise HTTPException(status_code=401, detail="unknown token")
    endpoint_url = (payload.get("endpoint_url") or "").strip()
    if not endpoint_url:
        raise HTTPException(status_code=400, detail="endpoint_url required")
    if _model_owned_by(team["team_id"]):
        raise HTTPException(
            status_code=409,
            detail=f"{team['display_name']} already has a model — use edit instead",
        )
    new_id = f"mdl-{team['team_id'].split('-')[-1]}-01"
    _MODELS.append({
        "model_id":         new_id,
        "owner_team_id":    team["team_id"],
        "endpoint_url":     endpoint_url,
        "ready_for_events": False,  # engine picks it up once ready toggled true
    })
    _PROBES[new_id] = []
    return {"model_id": new_id, "team_name": team["display_name"]}


@app.patch("/api/models/{model_id}")
def edit_model(model_id: str, payload: dict = Body(...)) -> dict:
    """Edit a team's model row.

    Body: {"token": "...", "team_name"?: "...", "endpoint_url"?: "..."}
    Token must belong to the model's owning team.
    """
    team = _team_by_token(payload.get("token", ""))
    if not team:
        raise HTTPException(status_code=401, detail="unknown token")
    model = next((m for m in _MODELS if m["model_id"] == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="model not found")
    if model["owner_team_id"] != team["team_id"]:
        raise HTTPException(status_code=403, detail="not your model")

    new_url = (payload.get("endpoint_url") or "").strip()
    if new_url:
        model["endpoint_url"] = new_url

    new_team_name = (payload.get("team_name") or "").strip()
    if new_team_name:
        team["display_name"] = new_team_name

    return {"ok": True}


# ---------------------------------------------------------------------------
# Static SPA
# ---------------------------------------------------------------------------

_DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_DASHBOARD_DIR / "index.html")


app.mount("/static", StaticFiles(directory=_DASHBOARD_DIR), name="static")
