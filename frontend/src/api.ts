export type Severity = "low" | "medium" | "high" | "critical";
export type Status = "processing" | "awaiting_review" | "sent" | "rejected" | "closed_no_reply" | "failed";

export interface TimelineEvent {
  at: string;
  step: string;
  actor: string;
  detail: string;
  latency_ms?: number;
  input_tokens?: number;
  output_tokens?: number;
  failed_step?: string;
}

export function modelUsage(event: TimelineEvent): string | null {
  if (event.step !== "triage" && event.step !== "draft" && event.step !== "investigate" && event.step !== "revise" && event.step !== "propose_actions") return null;
  const parts: string[] = [];
  if (typeof event.latency_ms === "number") parts.push(`${event.latency_ms} ms`);
  if (typeof event.input_tokens === "number" || typeof event.output_tokens === "number") {
    const total = (event.input_tokens ?? 0) + (event.output_tokens ?? 0);
    parts.push(`${total} tokens`);
  }
  return parts.length > 0 ? parts.join(" · ") : null;
}

export interface Snippet {
  id: string;
  title: string;
  text: string;
  score: number;
}

export interface Fact {
  text: string;
  source_tool: string;
}

export interface Findings {
  summary: string;
  facts: Fact[];
  nothing_found: boolean;
}

export interface ToolCall {
  tool: string;
  args: Record<string, string | number | null>;
  result: string;
  latency_ms?: number;
  input_tokens?: number;
  output_tokens?: number;
}

export interface DraftVersion {
  version: number;
  body: string;
  cited_policy_ids: string[];
  issues: string[];
}

export interface Proposal {
  id: string;
  type: string;
  args: Record<string, string>;
  rationale: string;
  evidence: string[];
  ref?: string;
}

export interface ActionResult {
  action_id: string;
  type: string;
  status: "executed" | "not_approved";
  ref: string | null;
  args: Record<string, string>;
}

export interface CaseState {
  case_id: string;
  email: { sender: string; subject: string; body: string };
  guard?: { injection_suspected: boolean; reasons: string[] };
  triage?: { category: string; severity: Severity; language: "en" | "ar"; summary: string; rationale: string };
  snippets?: Snippet[];
  findings?: Findings;
  tool_calls?: ToolCall[];
  draft?: { body: string; cited_policy_ids: string[] };
  draft_history?: DraftVersion[];
  revision_count?: number;
  grounding?: { ok: boolean; issues: string[] };
  proposals?: Proposal[];
  action_results?: ActionResult[];
  review?: {
    decision: string;
    reviewer: string;
    body?: string | null;
    note?: string | null;
    approved_action_ids?: string[];
    injection_acknowledged?: boolean;
  };
  final_body?: string | null;
  status: Status;
  model?: string;
  events: TimelineEvent[];
}

export interface CaseSummary {
  id: string;
  created_at: string;
  updated_at: string;
  sender: string;
  subject: string;
  status: Status;
  category: string | null;
  severity: Severity | null;
  injection_suspected: boolean;
  review_decision: string | null;
}

export interface CaseDetail extends CaseSummary {
  state: CaseState;
  next: string[];
}

export interface Sample {
  id: string;
  sender: string;
  subject: string;
  body: string;
}

export interface Metrics {
  total: number;
  by_status: Record<string, number>;
  review_decisions: Record<string, number>;
  approved_unchanged_rate: number | null;
  injection_flags: number;
  avg_latency_ms: { triage: number | null; draft: number | null; investigate: number | null };
  total_tokens: number | null;
  actions?: {
    proposed: Record<string, number>;
    approved: Record<string, number>;
    dropped: Record<string, number>;
  };
}

const API_UNREACHABLE = "Can’t reach the ResolveDesk API. Check that the backend is running on port 8100.";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch (err) {
    if (err instanceof TypeError) throw new Error(API_UNREACHABLE);
    throw err;
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep status text */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export interface Health {
  ok: boolean;
  model: string;
  prompt_version: string;
  investigate_prompt_version: string;
  demo_mode: boolean;
  snapshot_cases: number;
  s01_case_id: string | null;
}

export const api = {
  health: () => call<Health>("/health"),
  metrics: () => call<Metrics>("/metrics"),
  samples: () => call<Sample[]>("/samples"),
  cases: () => call<CaseSummary[]>("/cases"),
  getCase: (id: string) => call<CaseDetail>(`/cases/${id}`),
  createCase: (email: { sender: string; subject: string; body: string }) =>
    call<CaseDetail>("/cases", { method: "POST", body: JSON.stringify(email) }),
  createCaseAsync: (email: { sender: string; subject: string; body: string }) =>
    call<{ id: string }>("/cases/async", { method: "POST", body: JSON.stringify(email) }),
  retryCase: (id: string) => call<{ id: string }>(`/cases/${id}/retry`, { method: "POST" }),
  resetDemo: () => call<{ cases: number }>("/demo/reset", { method: "POST" }),
  review: (
    id: string,
    body: {
      decision: "approve" | "edit" | "reject";
      reviewer: string;
      body?: string;
      note?: string;
      approved_action_ids?: string[];
      injection_acknowledged?: boolean;
    },
  ) => call<CaseDetail>(`/cases/${id}/review`, { method: "POST", body: JSON.stringify(body) }),
  preview: (id: string, body: { approved_action_ids: string[]; body?: string }) =>
    call<{ final_body: string; arranged_section: string }>(`/cases/${id}/preview`, { method: "POST", body: JSON.stringify(body) }),
  ask: (id: string, question: string) =>
    call<{ answer: string; tools_used: string[] }>(`/cases/${id}/assistant`, {
      method: "POST",
      body: JSON.stringify({ question }),
    }),
  graph: () => call<{ mermaid: string }>("/graph"),
};
