# ITU SDSE Project — v2 Lecturer Tooling

## What this repo is

The **student-facing** artifact lives on `main` and in `README.md`, `docs/`, `notebooks/`, `.github/workflows/`, `action.yml`. Students fork it and turn the messy `notebooks/main.ipynb` monolith into a modular Python project with a Dagger workflow, a GitHub Actions training pipeline, and a model artifact validated by the [`model-validator`](https://github.com/lasselundstenjensen/itu-sdse-project-model-validator) action.

**Do not restructure the student inputs.** They are inputs to an exam and must stay recognisable.

## What v2 adds (this branch)

Students are now also expected to:
1. **Deploy** the trained model as an inference endpoint on Digital Ocean.
2. **Monitor** the deployed model for drift and **recover** (retrain + redeploy a new version) when drift is detected.

To grade and stress-test that, **we as lecturers** are building several tooling components in this repo, alongside the existing student material:

<important if="you are working on the event engine (platform/event-engine/**) or designing probe/health/drift-injection behavior">
### 1. Event engine (probe & stress)
- Runs on a schedule (every few minutes).
- For each registered model marked "ready for events", sends a **health inference request** to the student's Digital Ocean endpoint.
- Records **two independent indicators** per probe:
  - **Reachability** — did the endpoint respond at all (TCP/HTTP)?
  - **Correctness** — was the returned inference the expected answer for the known input?
- Periodically injects drift-inducing traffic patterns to test that students' monitoring + retrain-and-redeploy loop actually fires and recovers.
- Writes every probe result to the **status database** (see below).

**Student endpoint contract.** Each student's deployed model must expose a fixed HTTP contract that the event engine can rely on. This contract is part of the exam — students are graded on implementing it. The three required endpoints:

- `GET /health` — liveness. Returns `200 {"status": "ok"}` when the model is loaded and ready to serve. Used for the **reachability** indicator.
- `POST /infer` — inference. Request body: `{"input": <known-fixture>}`. Response: `{"prediction": <value>, "model_version": "<string>"}`. The event engine compares `prediction` against the expected answer for that known fixture to compute the **correctness** indicator.
- `GET /version` — model version. Returns `{"model_version": "<string>", "trained_at": "<iso8601>"}`. The event engine watches this to detect that a student has redeployed a new model after drift — a change in `model_version` following a run of failing probes is what triggers a **recovery event**.

Schemas live under `platform/registry/schemas/` (JSON Schema) so both the event engine and any future SDK can validate them. Keep them minimal and stable — students will code against these shapes.
</important>

<important if="you are working on the leaderboard dashboard SPA (platform/dashboard/**) or designing its metrics/visual layout">
### 2. Leaderboard dashboard (public SPA)
- Read-only, viewable by everyone (students, lecturers, guests).
- Lists all registered models with, at minimum: team name, model name, endpoint URL, ready-for-events toggle state, latest reachability + correctness status, recent history.
- Inspired by the Claude Status page — a per-model horizontal timeline of coloured bars (green/yellow/red) over the last N probes, plus an aggregate uptime/correctness percentage.
- **Metrics displayed:** current status, uptime % over configurable window, correctness % over configurable window, timestamp of last probe, timestamp of last failure, count of successful recovery events (drift detected → new model version deployed and healthy again).
- Single-page application. No auth needed to view; auth only required to mutate (see access control).

**Visual reference — replicate this.** See [`docs/dashboard-reference.png`](docs/dashboard-reference.png) (Anthropic status page). Per-model row is:

- **Left:** bold model title (team name · model display_name).
- **Right:** status label in green when healthy — `Operational` on the reference — set from the most recent probe.
- **Middle (full-width bar strip):** ~90 thin vertical bars, one per recent probe, oldest on the left, newest on the right. Bar colour is the probe outcome: green = reachable & correct, amber = reachable but incorrect (drift-suspected), red = unreachable. Small gap between bars; strip fills the row.
- **Under the strip:** three-column footer — left `N probes ago`, centre `<pct> % uptime` (or correctness — see below) between two thin horizontal rules, right `Today`.
- **Page header (top-right, once per page):** small muted line `Uptime over the last N probes.` No historical link needed in v1.

**Palette (light theme, paper background).** Bars: green `#7CB342`, amber `#F2B01E`, red `#E5484D` on a warm off-white page (`#FAFAF7`). Row separators are thin `#E5E5E0` rules. Title is near-black; the "Operational" label uses the same green as healthy bars. Font: system UI stack (matches the reference).

**Reachability + correctness — decision.** Use a **single strip per model, with three-colour semantics** (green / amber / red as above) rather than two tracks. This matches the reference visually and keeps rows compact. The two indicators are still stored independently in the Status DB; the SPA collapses them at render time.

**Aggregate metric.** Display **correctness %** as the centre-footer number (not raw uptime) — that's what the exam actually grades. Label it `<pct> % correctness` to be honest about which of the two indicators it is. Uptime %, last-probe time, last-failure time, and recovery count live in a hover tooltip or an expanded detail view, not in the compact row.

**Compact-row layout — no more than this.** Nothing per-row beyond title, strip, status label, and the three-column footer. Endpoint URL, ready-for-events toggle, and any admin controls belong on a per-model detail page or dialog reached by clicking the row — the leaderboard grid stays scan-friendly.
</important>

<important if="you are working on team auth, the admin SPA (platform/admin/**), or per-model write permissions">
### 3. Admin SPA (lecturer-only)
- Separate single-page application from the leaderboard.
- Lets lecturers **assign, enable, and disable teams** and their credentials (team id + password).
- Lets lecturers see all teams and force-toggle a team's ready-for-events flag if needed.

### 4. Team access control (per-model mutation)
- Everyone can **read** all model data.
- Only the team that owns a model can **update** their model's endpoint URL and their `ready-for-events` toggle.
- Auth mechanism should be intentionally simple: **team id + password** verified against the registry DB. A signed token / session cookie is fine; do not over-engineer. Bcrypt or scrypt for password hashing.
</important>

## Data stores

Two logically separate databases. Both start as **file-based** for local dev; the abstraction layer must let us swap backends later (e.g. SQLite → Postgres) without changing the SPAs or event engine.

**Registry DB** (teams + models) — written by admin SPA (teams) and by team members via the leaderboard SPA (their own model rows); read by both SPAs and the event engine.

**Status DB** (probe history) — append-only time-series of probe results plus recovery events; written by the event engine; read by the leaderboard SPA.

Kept separate so probe history can grow large without bloating the registry, and so we can rotate/prune it independently.

<important if="you are modifying the registry/status data model or the repository interface (platform/registry/**)">
Registry DB schema:
- Teams: `team_id`, `password_hash`, `enabled`, `created_at`.
- Models: `model_id`, `owner_team_id`, `display_name`, `endpoint_url`, `ready_for_events`, `registered_at`, `updated_at`.

Status DB schema:
- Probes: `probe_id`, `model_id`, `timestamp`, `reachable` (bool), `correct` (bool | null if unreachable), `latency_ms`, `response_snippet`, `probe_kind` (health | drift-injection | …).
- Recoveries: `model_id`, `detected_at`, `redeployed_at`, `new_model_version`.
</important>

## Abstraction layer

The two SPAs and the event engine all go through a shared **repository interface** — never file paths or DB drivers directly. Backend starts file-based; must be swappable to SQLite/Postgres without touching consumers.

<important if="you are modifying the registry/status data model or the repository interface (platform/registry/**)">
Minimum operations:
- `teams`: create, list, get, set_enabled, verify_password, delete.
- `models`: create, list, get, list_by_team, update_endpoint, set_ready, delete.
- `probes`: append, list_by_model(since, limit), aggregate(model_id, window) → uptime_pct, correctness_pct, last_seen.
- `recoveries`: append, list_by_model.
</important>

## Non-goals (v1 of this tooling)

- Cloud hosting / production deploy.
- Real multi-region monitoring.
- Alerting (email, Slack, PagerDuty).
- Historical rollups older than the retention window (start with "keep everything").
- Fancy auth (OAuth, SSO). Team id + password is enough.

## Working conventions

- **Absolute simplicity is the design brief.** This is a grading harness for a course, not a product. Prefer the boring choice every time: plain functions over classes, one file over three, standard library over a dependency, a single process over two, direct calls over indirection. Don't add abstractions for future flexibility that isn't asked for. Don't build config knobs for cases we don't have. Don't design for scale we won't hit. If a piece of the design isn't earning its keep against the four tooling components as they exist today, cut it. The repository interface exists because `CLAUDE.md` names it as a swap point — that is the *only* pre-emptive abstraction; everywhere else, write the shortest thing that works.
- Student-facing files (`README.md`, `notebooks/**`, `action.yml`, `.github/workflows/**` that ship to students, `docs/project-architecture.png`) are read-only for v2 tooling work unless the change is explicitly a student-facing improvement.
- v2 lecturer tooling lives under `platform/`:
  - `platform/registry/` — shared repository interface + file-backed implementation.
  - `platform/event-engine/` — the scheduled probe worker.
  - `platform/dashboard/` — leaderboard SPA.
  - `platform/admin/` — admin SPA.
  - `platform/data/` — gitignored local DB files (`.gitkeep` and a seed script committed).
- **Stack:** Python everywhere on the backend (matches the course). SPAs are vanilla HTML + JS, no build step; Alpine.js is allowed as a sprinkle if reactivity gets painful. **FastAPI** wraps the shared repository module and serves both SPAs — the event engine imports the module in-process, so there's one interface with two callers.
- Follow the global `verify-` prefix convention for any skill/workflow whose job is to check another step's output.

