import { BadgeCheck, ListChecks, LoaderCircle, Mail, PenLine, ScanSearch, Search, Send, Shield, Tags, UserRound, type LucideIcon } from "lucide-react";
import { modelUsage, type CaseDetail, type TimelineEvent } from "../api";
import { formatAbsolute, formatRelative, useNow } from "../time";
import Tooltip from "./Tooltip";

const STEPS: { key: string; label: string; icon: LucideIcon }[] = [
  { key: "intake", label: "Intake", icon: Mail },
  { key: "guard", label: "Guard", icon: Shield },
  { key: "triage", label: "Triage", icon: Tags },
  { key: "retrieve", label: "Retrieve", icon: Search },
  { key: "investigate", label: "Investigate", icon: ScanSearch },
  { key: "draft", label: "Draft", icon: PenLine },
  { key: "ground_check", label: "Ground check", icon: BadgeCheck },
  { key: "propose_actions", label: "Actions", icon: ListChecks },
  { key: "human_review", label: "Review", icon: UserRound },
  { key: "send", label: "Send", icon: Send },
];

const SPAM_SKIP = new Set(["retrieve", "investigate", "draft", "ground_check", "propose_actions", "human_review", "send"]);

type StepState = "done" | "current" | "pending" | "skipped";

export function agentRunning(c: CaseDetail): boolean {
  return c.status === "processing";
}

function failedStep(c: CaseDetail): string | null {
  if (c.status !== "failed") return null;
  const marker = [...c.state.events].reverse().find((item) => item.step === "failed");
  if (!marker) return null;
  if (marker.failed_step && STEPS.some((step) => step.key === marker.failed_step)) return marker.failed_step;
  const name = marker.detail.split(" ")[0];
  return STEPS.some((step) => step.key === name) ? name : null;
}

function stepState(key: string, c: CaseDetail, done: Set<string>, currentKey: string | null): StepState {
  const waiting = c.next.includes("human_review");
  if (c.status === "closed_no_reply" && SPAM_SKIP.has(key)) return "skipped";
  if (c.status === "rejected" && key === "send") return "skipped";
  if (key === "human_review" && waiting && !agentRunning(c)) return "current";
  if (key === currentKey) return "current";
  if (done.has(key) && key !== failedStep(c)) return "done";
  return "pending";
}

function latest(events: TimelineEvent[], key: string): TimelineEvent | undefined {
  const matches = events.filter((event) => event.step === key);
  return matches[matches.length - 1];
}

export default function Pipeline({ c }: { c: CaseDetail }) {
  const now = useNow();
  const done = new Set(c.state.events.map((event) => event.step));
  const closeEvent = latest(c.state.events, "close");
  const running = agentRunning(c);
  const broken = failedStep(c);
  const currentKey = broken ?? (running ? (STEPS.find((step) => !done.has(step.key) && !(c.status === "closed_no_reply" && SPAM_SKIP.has(step.key)))?.key ?? null) : null);

  return (
    <ol className="stepper">
      {STEPS.map((step) => {
        const state = stepState(step.key, c, done, currentKey);
        const own = latest(c.state.events, step.key);
        const event = own ?? (state === "skipped" ? closeEvent : undefined);
        const Icon = step.icon;
        const usage = event ? modelUsage(event) : null;
        const tip = event ? (
          <>
            <span dir="auto">{event.detail}</span>
            {usage && <span className="tip-meta">{usage}</span>}
            <span className="tip-meta">
              {event.actor} · {formatAbsolute(event.at)} ({formatRelative(event.at, now)})
            </span>
          </>
        ) : state === "current" ? (
          "Waiting for a reviewer."
        ) : (
          "Not reached yet."
        );
        return (
          <li key={step.key} className={`step step-${state}`}>
            <Tooltip text={tip}>
              <button type="button" className="step-btn" aria-current={state === "current" ? "step" : undefined}>
                <span className="step-mark">
                  {state === "current" && running ? <LoaderCircle className="spin" aria-hidden /> : <Icon aria-hidden />}
                </span>
                <span className="step-label">{step.label}</span>
                {step.key === "ground_check" && (c.state.revision_count ?? 0) > 0 && (
                  <span className="step-badge">revised {c.state.revision_count}×</span>
                )}
              </button>
            </Tooltip>
          </li>
        );
      })}
    </ol>
  );
}
