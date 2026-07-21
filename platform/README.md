# platform/ — v2 lecturer tooling

Grading harness for the v2 exam extension. See top-level `CLAUDE.md` for the full plan and design decisions.

## What lives here today

```
platform/
├── api/           # FastAPI mock — hardcoded data, no DB
│   └── mock_api.py
├── dashboard/     # public leaderboard SPA (vanilla HTML+JS)
│   └── index.html
└── requirements.txt
```

The full target layout adds `registry/`, `event-engine/`, `admin/`, and `data/`. Those are not built yet.

## Running the mock dashboard

```bash
pip install -r platform/requirements.txt
uvicorn --app-dir platform api.mock_api:app --reload --port 8000
```

`--app-dir platform` matters: the directory name `platform` collides with the Python stdlib module of the same name, so we don't turn it into a package. Uvicorn imports `api.mock_api` from inside it instead.

Open <http://127.0.0.1:8000/>.

The mock returns hardcoded probe series for five teams — one mostly-healthy, one baseline, one with a drift dip and recovery, one flaky, and one pending its first probe. Reload the page to redraw; the payload is fixed per process start.

## API surface (mock)

- `GET /api/dashboard` — one-shot payload the SPA renders. Composes teams + models + a fixed-length probe series per model. This is the *only* endpoint the mock ships; when the real repository lands, this endpoint will call `probes.list_by_model(…)` + `probes.aggregate(…)` under the hood, and per-team mutation routes (`PATCH /api/models/{id}`, etc.) will be added alongside it.

## Design notes worth re-reading

- The dashboard row layout, palette, three-colour bar semantics, and the choice of correctness % as the headline metric are pinned in `CLAUDE.md` → §2 with `docs/dashboard-reference.png` as the visual anchor. Don't drift from those without editing `CLAUDE.md` first.
- Simplicity rule (`CLAUDE.md` → *Working conventions*): the repository interface is the *only* pre-emptive abstraction. Everywhere else, shortest thing that works.
