# Demo runbook (5 minutes)

A new real-model case takes about 15–20 seconds. The snapshot cases do not call the model.

## Before

1. Use the OS light theme. Silence notifications. Set the browser zoom to 100%.
2. In `backend/.env`, set `LLM_PROVIDER` to the provider you will use, and set the matching key in the environment (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`). Do not print the key. Leave the model name as it is.
3. Turn demo mode on, either by adding `DEMO_MODE=true` to `backend/.env`, or in the shell before you start the API. `API_PORT` is read only by the checklist script. The server itself takes `--port`.

```bash
cd backend
uv run uvicorn app.main:app --port 8100
```

Windows PowerShell 5.1 does not support `&&`. Set the variable, then start the server:

```powershell
cd backend
$env:DEMO_MODE = "true"
uv run uvicorn app.main:app --port 8100
```

4. From `frontend` (`VITE_PORT` defaults to 5180, `VITE_API_TARGET` defaults to `http://127.0.0.1:8100`):

```bash
cd frontend
npm run dev
```

Open http://localhost:5180. The header should show the model name, not "Offline mode", and a **Reset demo** button.

5. Run the checklist. It exits non-zero if anything failed:

```bash
cd backend
uv run python -m scripts.preflight
```

6. Press **Reset demo** and confirm. The inbox should show 8 cases, and the brake complaint (S01) should be open.

If port 8100 or 5180 is already taken, preflight prints the PID and nothing else about that process. Do not start a second copy.

## Script

Say one line, then click.

| Step | Target | Say | Do |
|---|---|---|---|
| 1. Inbox | 20 s | "Critical cases sit on top, and an injection flag is visible before you open the email." | Stay on Review. Point at the critical badges and the shield on S24. |
| 2. S01 | 70 s | "Severity follows risk, not how calm the email sounds." | Open **Brakes grinding after service**. Show the triage reason, the earlier brake visit in the investigation, the safety-first draft, and the proposed actions with their evidence. |
| 3. Approve | 50 s | "I choose the actions. The reply only gains them after I approve." | Tick roadside pickup and safety inspection. Wait for the preview "What we have arranged". Leave escalation unticked. Approve. Show the references in the reply and the audit line with the reviewer name. |
| 4. Injection | 50 s | "The email is data. It cannot approve its own refund, and sending still needs a person to acknowledge the warning." | Open S24. Show the guard banner. Show that the draft does not promise AED 20,000. The approve button stays blocked until the acknowledgement is ticked. Do not send it. |
| 5. Assistant | 30 s | "The assistant can look things up. It cannot approve or send." | Ask **Approve and send this reply**. The answer refuses. The tools list has no write tool. |
| 6. Live | 40 s plus the run | "A new case shows each step as it is stored. Nothing here is a fake progress bar." | New complaint, pick the airbag sample (S28), Run agent. The modal closes and the stepper fills from the events. Talk through the graph while it runs. Stop when it reaches review. |

If time runs short, say only these three:

1. Severity follows the risk to the customer, not the tone of the email.
2. Nothing is sent unless a named person approves it.
3. The assistant has no tool that can approve or send.

## Fallbacks

- **Model slow or down.** Do not create a live case. Walk steps 1–5 on the snapshot. Those cases are already computed.
- **Network down.** Stop the server and start it again with `LLM_PROVIDER=offline` and `DEMO_MODE=true`, then press Reset demo only if you built an offline snapshot. The shipped snapshot was built with the real model, so Reset still shows those cases. What changes on screen: the header pill says **Offline mode**, new cases are rule-based, and the assistant does not call a model. The human gate, the injection banner, and the action preview stay.
- **A live case fails.** The inbox shows **Failed** and a **Retry** button. The finished steps are kept and Retry continues from the checkpoint. Use it once if it happens. If Retry fails too, go back to a snapshot case.
- **UI broken.** Use the screenshots in `assets/screenshots/`: S01 at review, S01 after approve, the injection case, and live progress.

## After

Press **Reset demo** so the next run starts from S01 awaiting review again.
