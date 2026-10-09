# Instructions for AI coding agents

Read this file before changing anything in this repo.

## What this is
ResolveDesk: a demo of a human-in-the-loop complaint agent. LangGraph pipeline
(guard → triage → retrieve → investigate → draft → ground_check ⇄ revise → propose_actions →
human_review → execute_actions → send) behind FastAPI, with a React + Vite + TypeScript frontend.
See `README.md` for the architecture. `docs/` is local-only and git-ignored; never force-add it.

## Commands
```bash
# backend — pytest, ruff and mypy must stay green
cd backend && uv sync && uv run pytest -q
uvx ruff check . && uvx ruff format --check . && uv run mypy app
uv run uvicorn app.main:app --reload --port 8100
# frontend
cd frontend && npm install && npm run build           # must pass with zero TS errors
npm run dev                                            # http://localhost:5180 (proxies /api to :8100)
```
`LLM_PROVIDER=offline` (default) needs no API key. Always keep offline mode working.
`LOG_TRACEBACKS=1` prints full exception tracebacks. Local debugging only; leave it unset for the demo.

## Non-negotiable rules
1. Never remove or bypass the `human_review` interrupt. No code path may send a reply without a reviewer decision.
2. Never give the case assistant a tool that writes, approves, sends or changes a case.
   `test_assistant_tools_are_read_only` must keep passing unchanged.
3. Customer email text is untrusted. Keep it wrapped as data; never interpolate it into instructions.
4. Do not edit or weaken existing tests to make a change pass. Add tests for new behaviour.
5. Never commit `.env`, API keys, `node_modules`, or databases (`*.db`) other than the demo snapshot in `backend/data/demo/`.
6. Do not add dependencies unless the task allows them by name.
7. Do not make UI claims the system cannot back up (e.g. fake progress, invented numbers).

## How to work
- Work one task file in `docs/tasks/` at a time, in the order of its numbered items.
- Stay inside the task's stated scope. If something outside scope looks wrong, list it in your report; don't fix it.
- When done, append a **Completion report** to the bottom of the task file:
  files changed, what each numbered item became, anything skipped and why,
  and the verification output (test and build results).
