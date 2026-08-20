# Boardroom AI — Backend

The API behind [Boardroom AI](https://boardroom-ai-frontend.vercel.app/): four AI executives (CEO, CFO, CTO, CMO) debate a business problem in real time, then converge on a structured action plan and a timeline chart.

Built with **FastAPI**, **AutoGen** (multi-agent debate), and **LangGraph** (plan + chart synthesis), running on **Groq** for inference and **Supabase** for auth, storage, and encrypted API-key management.

## How it works

1. **Debate** — `AssistantAgent`s for each executive role run through a bounded `RoundRobinGroupChat` (AutoGen), streamed to the client turn-by-turn as they're generated.
2. **Synthesis** — Once the debate ends, a LangGraph pipeline takes over:
   - `extract_plan` compresses the debate into a 4-step action plan.
   - `generate_chart_data` asks the LLM for structured timeline data (`{step, duration_days}`) as JSON — **never code**.
   - `render_chart` draws the Gantt chart itself, from a fixed matplotlib function. The model's output is validated and clamped before it ever touches the chart, and it has no path to execute anything.
3. **Delivery** — Everything streams to the frontend over Server-Sent Events: debate turns, status updates, and a final `complete` event with the plan and chart (inline base64 PNG — no storage bucket required).

## Persistence

Supabase tables:
- `profiles` — one row per user; `groq_key_secret_id` points into Supabase Vault.
- `plans` — one row per session (title, action plan markdown, chart as base64).
- `chats` — one row per debate turn, linked to a `plan` (cascades on delete).

Groq API keys for signed-in users are stored **encrypted via Supabase Vault**, accessed only through `get_groq_key()` / `save_groq_key()` `SECURITY DEFINER` RPCs scoped to `auth.uid()` — never in plaintext, never in `user_metadata`. Guest mode has no persistence; the key lives only in the browser for that session.

## API

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/stream` | Optional (Bearer JWT) | Runs the debate + plan pipeline, streams results via SSE. Guests get an ephemeral session; authenticated users get it saved to `plans`/`chats`. |
| `GET` | `/api/history` | Required | Returns the authenticated user's past sessions (plan + debate turns), most recent first. |
| `HEAD` | `/health` | None | Liveness check — also used by the frontend to detect and message through cold starts on free-tier hosting. |

## Getting started

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
uvicorn main:app --reload
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `SUPABASE_URL` | Yes | Your Supabase project URL. |
| `SUPABASE_SERVICE_KEY` | Yes | Service-role key — used server-side only, bypasses RLS (queries are manually scoped by `user_id`). |
| `FRONTEND_URL` | Yes | Comma-separated list of exact production origin(s) allowed by CORS, e.g. `https://boardroom-ai-frontend.vercel.app`. Trailing slashes are stripped automatically. |
| `PREVIEW_ORIGIN_REGEX` | No | Regex scoped to your own preview deployments (e.g. `https://boardroom-ai-frontend-.*\.vercel\.app`). Leave unset to only trust `FRONTEND_URL` + `localhost`. |

## Deployment notes

- Deployed on Render's free tier, which spins the service down after inactivity. The frontend pings `/health` and shows a "waking up" message on cold starts — expect the first request after idle time to take up to ~50s.
- CORS is origin-allowlisted, not wildcarded — update `FRONTEND_URL`/`PREVIEW_ORIGIN_REGEX` if you change domains, and redeploy (env var changes require a restart to take effect).

## Security notes

- The chart-generation step deliberately never executes model-generated code — only structured, validated JSON. This closed off what was originally a remote-code-execution path (LLM-authored Python being run directly on the server).
- Row Level Security is enabled on all Supabase tables (`plans`, `chats`, `profiles`), scoped to `auth.uid()`.
