import { Copy, ShieldAlert, Sparkles } from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { api, modelUsage, type CaseDetail } from "../api";
import { categoryLabel } from "../labels";
import { formatAbsolute, formatRelative, useNow } from "../time";
import { useToast } from "../toast";
import AssistantPanel, { readAssistantOpen, storeAssistantOpen } from "./AssistantPanel";
import { SeverityBadge, StatusBadge } from "./Badges";
import Collapse from "./Collapse";
import Pipeline, { agentRunning } from "./Pipeline";
import ReviewPanel from "./ReviewPanel";
import Tooltip from "./Tooltip";

function PolicyText({ text, open, onToggle }: { text: string; open: boolean; onToggle: () => void }) {
  const ref = useRef<HTMLParagraphElement>(null);
  const [overflows, setOverflows] = useState(false);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const style = getComputedStyle(el);
      const fontSize = Number.parseFloat(style.fontSize);
      const lineHeight = style.lineHeight.endsWith("px") ? Number.parseFloat(style.lineHeight) : fontSize * 1.45;
      const twoLines = lineHeight * 2 + 1;
      const clone = el.cloneNode(true) as HTMLParagraphElement;
      clone.className = "policy-text";
      clone.style.position = "absolute";
      clone.style.visibility = "hidden";
      clone.style.pointerEvents = "none";
      clone.style.height = "auto";
      clone.style.display = "block";
      clone.style.width = `${el.clientWidth}px`;
      el.parentElement?.appendChild(clone);
      const full = clone.scrollHeight;
      clone.remove();
      setOverflows(full > twoLines);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    window.addEventListener("resize", measure);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [text, open]);

  return (
    <>
      <p ref={ref} className={open ? "policy-text" : "policy-text clamp"}>
        {text}
      </p>
      {overflows && (
        <button type="button" className="link" aria-expanded={open} onClick={onToggle}>
          {open ? "Show less" : "Show more"}
        </button>
      )}
    </>
  );
}

function SummaryText({ text }: { text: string }) {
  const ref = useRef<HTMLParagraphElement>(null);
  const [open, setOpen] = useState(false);
  const [overflows, setOverflows] = useState(false);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || open) return;
    setOverflows(el.scrollHeight > el.clientHeight + 1);
  }, [text, open]);

  return (
    <>
      <p ref={ref} className={open ? "" : "clamp-4"} dir="auto">
        {text}
      </p>
      {overflows && (
        <button type="button" className="link" aria-expanded={open} onClick={() => setOpen((current) => !current)}>
          {open ? "Show less" : "Show more"}
        </button>
      )}
    </>
  );
}

export default function CaseView({ caseId, onChanged }: { caseId: string; onChanged: () => void }) {
  const [c, setC] = useState<CaseDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [assistantOpen, setAssistantOpen] = useState(readAssistantOpen);
  const [policiesOpen, setPoliciesOpen] = useState(true);
  const [auditOpen, setAuditOpen] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [highlight, setHighlight] = useState<string | null>(null);
  const now = useNow();
  const [clock, setClock] = useState(() => Date.now());
  const toast = useToast();
  const running = c != null && agentRunning(c);
  const changedRef = useRef(onChanged);
  changedRef.current = onChanged;

  const load = useCallback(() => {
    setError(null);
    api
      .getCase(caseId)
      .then(setC)
      .catch((err: unknown) => setError((err as Error).message));
  }, [caseId]);

  useEffect(load, [load]);

  useEffect(() => {
    if (!running) return;
    let cancelled = false;
    let seq = 0;
    let timer = 0;
    const tick = () => {
      const current = ++seq;
      setClock(Date.now());
      api
        .getCase(caseId)
        .then((next) => {
          if (cancelled || current !== seq) return;
          setC(next);
          if (!agentRunning(next)) changedRef.current();
        })
        .catch(() => {
          // A missed poll keeps the last case on screen.
        })
        .finally(() => {
          if (!cancelled) timer = window.setTimeout(tick, 1000);
        });
    };
    timer = window.setTimeout(tick, 1000);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [running, caseId]);

  const toggleAssistant = () => {
    const next = !assistantOpen;
    storeAssistantOpen(next);
    setAssistantOpen(next);
  };

  const openPolicy = (id: string) => {
    setPoliciesOpen(true);
    setExpanded((current) => new Set(current).add(id));
    setHighlight(id);
    window.setTimeout(() => {
      document.getElementById(`policy-${id}`)?.scrollIntoView({ block: "nearest" });
    }, 0);
  };

  if (error) {
    return (
      <div className="banner error">
        {error}{" "}
        <button type="button" className="btn" onClick={load}>
          Retry
        </button>
      </div>
    );
  }
  if (!c) {
    return (
      <div className="caseview" aria-busy="true" aria-label="Loading case">
        <div className="skel skel-title" />
        <div className="skel skel-stepper" />
        <div className="skel-grid">
          <div className="skel skel-card" />
          <div className="skel skel-card" />
        </div>
      </div>
    );
  }

  const s = c.state;
  const t = s.triage;
  const awaiting = c.next.includes("human_review");
  const cited = new Set(s.draft?.cited_policy_ids ?? []);
  const policies = [...(s.snippets ?? [])].sort((a, b) => {
    const aCited = cited.has(a.id) ? 0 : 1;
    const bCited = cited.has(b.id) ? 0 : 1;
    if (aCited !== bCited) return aCited - bCited;
    return b.score - a.score;
  });

  const focusDraft = () => {
    document.getElementById("draft-card")?.scrollIntoView({ block: "nearest" });
    document.getElementById("draft-editor")?.focus();
  };

  const retry = async () => {
    try {
      await api.retryCase(c.id);
      setC({ ...c, status: "processing" });
      onChanged();
    } catch (err) {
      toast((err as Error).message);
    }
  };

  const timing = (() => {
    const events = s.events;
    if (events.length === 0) return null;
    const start = Date.parse(events[0].at);
    if (Number.isNaN(start)) return null;
    const end = running ? clock : Date.parse(events[events.length - 1].at);
    if (Number.isNaN(end)) return null;
    const seconds = Math.max(0, (end - start) / 1000).toFixed(1);
    if (running) return `Elapsed ${seconds} s`;
    if (c.status === "failed") return `Stopped after ${seconds} s`;
    return `Agent finished in ${seconds} s`;
  })();

  const copyId = async () => {
    try {
      await navigator.clipboard.writeText(c.id);
      toast("Case id copied");
    } catch {
      toast("Could not copy the case id");
    }
  };

  return (
    <div className="case-stage">
      <div className="caseview">
        <div className="fold">
          <header className="case-head">
            <div className="case-head-main">
              <h2 dir="auto">{s.email.subject}</h2>
              <p className="meta-line">
                <span>{s.email.sender}</span>
                <span aria-hidden>·</span>
                <Tooltip text={formatRelative(c.created_at, now)}>
                  <span className="icon-focus" tabIndex={0}>
                    {formatAbsolute(c.created_at)}
                  </span>
                </Tooltip>
                <span aria-hidden>·</span>
                <span className="id-chip">{c.id.slice(0, 8)}</span>
                {timing && <span className="muted small">{timing}</span>}
                <button type="button" className="icon-btn" aria-label="Copy case id" onClick={() => void copyId()}>
                  <Copy aria-hidden />
                </button>
              </p>
            </div>
            <div className="case-head-side">
              <StatusBadge status={c.status} />
              {s.guard?.injection_suspected && (
                <span className="badge injection">
                  <ShieldAlert aria-hidden /> Injection suspected
                </span>
              )}
              {t && <SeverityBadge severity={t.severity} />}
              {t && <span className="tag">{categoryLabel(t.category)}</span>}
              {awaiting && (
                <button type="button" className="btn primary" onClick={focusDraft}>
                  Review draft
                </button>
              )}
              {c.status === "failed" && (
                <button type="button" className="btn primary" onClick={() => void retry()}>
                  Retry
                </button>
              )}
              <button
                type="button"
                className={assistantOpen ? "btn on" : "btn"}
                aria-pressed={assistantOpen}
                onClick={toggleAssistant}
              >
                <Sparkles aria-hidden /> Ask assistant
              </button>
            </div>
          </header>
          <Pipeline c={c} />
          {s.guard?.injection_suspected && (
            <div className="banner warn">
              <ShieldAlert aria-hidden />
              <span>
                <b>Suspected prompt injection.</b> {s.guard.reasons.join("; ")}. The email was treated as data only;
                check the draft carefully.
              </span>
            </div>
          )}
          <div className="decision-row">
            <div className="context-col">
              <section className="card email-card">
                <h3>Customer email</h3>
                <p className="email" dir="auto">
                  {s.email.body}
                </p>
              </section>
              {t && (
                <section className="card">
                  <h3>Triage</h3>
                  <div className="row gap wrap">
                    <span className="tag strong">{categoryLabel(t.category)}</span>
                    <SeverityBadge severity={t.severity} />
                    <span className="tag">{t.language === "ar" ? "Arabic" : "English"}</span>
                  </div>
                  <p>{t.summary}</p>
                  <p className="muted small">
                    <b>Why:</b> {t.rationale}
                  </p>
                </section>
              )}
              {s.findings && (
                <section className="card">
                  <h3>Investigation</h3>
                  <p className="muted small">
                    <b>Agent summary</b>
                  </p>
                  {s.findings.summary ? <SummaryText text={s.findings.summary} /> : null}
                  {s.findings.nothing_found ? <p className="muted">Nothing relevant found.</p> : null}
                  {s.findings.facts.length > 0 && (
                    <details>
                      <summary>
                        {s.findings.facts.length} {s.findings.facts.length === 1 ? "fact" : "facts"} from{" "}
                        {(s.tool_calls ?? []).length} {(s.tool_calls ?? []).length === 1 ? "lookup" : "lookups"}
                      </summary>
                      <ul className="facts">
                        {s.findings.facts.map((fact) => (
                          <li key={`${fact.source_tool}-${fact.text}`}>
                            <span dir="auto">{fact.text}</span>
                            <span className="muted small">{fact.source_tool}</span>
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                  {(s.tool_calls ?? []).length > 0 && (
                    <details>
                      <summary>Tool calls ({s.tool_calls?.length})</summary>
                      <ul className="tool-calls">
                        {s.tool_calls?.map((call, index) => (
                          <li key={`${call.tool}-${index}`}>
                            <b>{call.tool}</b>
                            <span className="muted small">
                              {Object.keys(call.args).length > 0 ? ` (${JSON.stringify(call.args)})` : " ()"}
                              {typeof call.latency_ms === "number" ? ` · ${call.latency_ms} ms` : ""}
                            </span>
                            <span className="small" dir="auto">
                              {call.result}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                </section>
              )}
            </div>
            <div className="review-col">
              <ReviewPanel
                key={c.id + (s.review?.decision ?? "open")}
                c={c}
                onOpenPolicy={openPolicy}
                onDone={(updated) => {
                  setC(updated);
                  onChanged();
                  const decision = updated.state.review?.decision;
                  const to = updated.state.email.sender;
                  if (decision === "approve") toast(`Reply approved and queued to ${to}`);
                  else if (decision === "edit") toast(`Edited reply queued to ${to}`);
                  else if (decision === "reject") toast("Draft rejected");
                }}
              />
            </div>
          </div>
        </div>

        {policies.length > 0 && (
          <Collapse title="Policies used" open={policiesOpen} onToggle={() => setPoliciesOpen((open) => !open)}>
            <ul className="policies">
              {policies.map((policy) => {
                const isOpen = expanded.has(policy.id);
                return (
                  <li
                    key={policy.id}
                    id={`policy-${policy.id}`}
                    className={[
                      "policy",
                      cited.has(policy.id) ? "cited" : "",
                      highlight === policy.id ? "highlight" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    <div className="row between wrap">
                      <b>
                        [{policy.id}] {policy.title}
                      </b>
                      <span className="muted small">
                        {cited.has(policy.id) ? "cited · " : ""}score {policy.score}
                      </span>
                    </div>
                    <PolicyText
                      text={policy.text}
                      open={isOpen}
                      onToggle={() =>
                        setExpanded((current) => {
                          const next = new Set(current);
                          if (next.has(policy.id)) next.delete(policy.id);
                          else next.add(policy.id);
                          return next;
                        })
                      }
                    />
                  </li>
                );
              })}
            </ul>
          </Collapse>
        )}

        <Collapse title="Audit trail" open={auditOpen} onToggle={() => setAuditOpen((open) => !open)}>
          <ol className="timeline">
            {s.events.map((event, index) => {
              const usage = modelUsage(event);
              return (
                <li key={`${event.step}-${index}`}>
                  <Tooltip text={formatAbsolute(event.at)}>
                    <span className="icon-focus" tabIndex={0}>
                      {formatRelative(event.at, now)}
                    </span>
                  </Tooltip>
                  <span className="step-name">{event.step.replace(/_/g, " ")}</span>
                  <span className="actor">{event.actor}</span>
                  <span className="small" dir="auto">
                    {event.detail}
                    {usage && <span className="muted"> · {usage}</span>}
                  </span>
                </li>
              );
            })}
          </ol>
          {s.model && <p className="muted small">Model: {s.model}</p>}
        </Collapse>
      </div>
      {assistantOpen && <AssistantPanel caseId={c.id} onClose={toggleAssistant} />}
    </div>
  );
}
