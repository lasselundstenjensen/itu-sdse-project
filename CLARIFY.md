# CLARIFY — open questions before building admin + dashboard v1

Scope: what stops us from writing the first version of `platform/admin/` and `platform/dashboard/` against a file-based repository. Grouped roughly in the order the gaps bite.

## 1. File-DB shape — JSONL vs SQLite

Both are "file-based" but they behave differently:

- **Concurrent writes.** Event engine appends probes while the dashboard reads. SQLite handles this via WAL out of the box. JSONL needs an explicit `fcntl` / advisory-lock pattern in the repo module, or a single-writer discipline.
- **Aggregates.** `probes.aggregate(model_id, window)` — over JSONL it's a full-file scan (fine at course scale, painful once tuned). Over SQLite it's an indexed query.
- **Migration story.** SQLite → Postgres is a one-hop schema copy. JSONL → Postgres is a small ETL.

The repo interface can be sketched without picking, but running the SPAs against real data needs a decision.

## 2. Process model — one process or two?

`CLAUDE.md` says the event engine "imports the module in-process." That's ambiguous:

- **(a)** Engine runs as a background task inside the FastAPI process (APScheduler / asyncio task / `BackgroundTasks`).
- **(b)** Engine is a separate Python process that just imports the same repo module.

Both are legitimate; they lead to different file-locking, restart, and `make dev` shapes.

## 3. Admin/lecturer auth is undefined

Team auth is specified (team_id + bcrypt password against Registry). Nothing says how a lecturer authenticates to the admin SPA:

- Are lecturers a special row in `Teams` (e.g. `is_admin` flag)?
- A separate `Admins` table?
- A single shared password from env?
- How is the *first* admin credential seeded on a fresh checkout?

Without this, the admin SPA can't be built at all.

## 4. Session mechanism

"Signed token or cookie is fine, don't over-engineer" — but pick one:

- Cookie session (needs a signing-key source + CSRF story for mutation calls)
- Bearer JWT (needs signing key + expiry policy)
- HTTP Basic per-request (simplest; ugly UX)

## 5. Concrete HTTP API surface

The plan names "FastAPI wraps the repo module" but not the routes. Both SPAs need a fixed contract before either can be built:

- `GET /api/models`, `GET /api/models/{id}`, `PATCH /api/models/{id}` (team-authed)
- `GET /api/probes?model_id=…&since=…&limit=…`
- `GET /api/models/{id}/aggregate?window=…`
- Admin: `POST /api/teams`, `PATCH /api/teams/{id}` (enable/disable, reset password), `PATCH /api/models/{id}/ready` (force-toggle)
- `POST /api/auth/login`, `POST /api/auth/logout`
- Error envelope shape.

## 6. Model creation flow

Registry schema has `owner_team_id`, but the flow isn't stated:

- Does the admin create model rows and assign them to teams at setup, or do teams self-service create models after login?
- Can a team own more than one model? (Schema allows it; UX doesn't say.)
- What fields does a team supply at model creation vs after?

## 7. Endpoint-URL & ready-for-events policy

- Is `endpoint_url` validated on submit — must it be HTTPS, must it resolve, must `/health` return 200 before accepting?
- Can a team flip `ready_for_events = true` before any successful probe, or is that gated on ≥1 green probe?

## 8. Dashboard visual/semantic decisions

Mostly **resolved** in `CLAUDE.md` against `docs/dashboard-reference.png` (Anthropic status page style):

- **Single strip, three-colour semantics** (green = reachable & correct, amber = reachable but incorrect, red = unreachable) — chose over two overlaid tracks.
- **Bar count N = ~90** probes per row (matches the reference).
- **Centre-footer metric is correctness %** (not uptime); uptime lives in the tooltip / detail view.
- **Compact row only** carries title · strip · status label · footer. Endpoint URL, ready-for-events toggle, and admin controls belong on a per-model detail page.

Still open:

- **Empty-state** — what does a just-registered model with zero probes look like on the leaderboard? (Blank strip with a `Pending first probe` status label is the obvious answer; confirm.)
- **Aggregate window switcher** — is the last-N-probes count fixed at 90, or user-switchable (1 h / 24 h / 7 d)? Fixed is simpler; simpler wins per the new working convention.

## 9. Recovery-event definition

The dashboard shows "count of successful recovery events" but the plan doesn't nail down what one is:

- Any `correct=false → correct=true` transition, or ≥K consecutive failures followed by recovery?
- Must the recovered probe come with a *new* `model_version` (via `GET /version`) to count, or is behavioural recovery enough?
- Does an out-of-band version bump with no prior failure count as anything?

This blocks the dashboard because the counter has to display *something*.

## 10. Seed data

A fresh checkout with an empty Status DB shows nothing on the leaderboard — hard to develop against and hard to demo. `CLAUDE.md` mentions a seed script but not what it populates (how many teams, how much probe history, any recovery events).

---

## Not blockers for admin/dashboard v1

Flagging so we don't confuse scopes:

- The `/infer` known-fixture source and the drift-injection schedule — live entirely inside the event engine.
- Retention / rotation policy for the Status DB — `CLAUDE.md` defers this to "keep everything," which is fine for v1.

## Highest-leverage decisions

If forced to pick two, these produce the biggest downstream rework when left implicit:

1. **File-DB choice (§1)** — shapes the repo interface's concurrency + query guarantees.
2. **Admin auth model (§3)** — shapes the admin SPA's entire surface.
