import { Check, ClipboardList, Inbox, ShieldAlert, Workflow } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api, type CaseSummary, type Health, type Metrics } from "./api";
import CaseList, { type InboxFilter } from "./components/CaseList";
import CaseView from "./components/CaseView";
import Modal from "./components/Modal";
import NewCase from "./components/NewCase";
import Tooltip from "./components/Tooltip";
import { useToast } from "./toast";

function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(() => window.matchMedia("(max-width: 1023px)").matches);
  useEffect(() => {
    const media = window.matchMedia("(max-width: 1023px)");
    const onChange = () => setNarrow(media.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);
  return narrow;
}

export default function App() {
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [composing, setComposing] = useState(false);
  const [graphOpen, setGraphOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<InboxFilter>("needs_review");
  const [inboxOpen, setInboxOpen] = useState(true);
  const narrow = useNarrow();
  const toast = useToast();

  const refresh = useCallback(async () => {
    try {
      const [nextCases, nextMetrics, nextHealth] = await Promise.all([api.cases(), api.metrics(), api.health()]);
      setCases(nextCases);
      setMetrics(nextMetrics);
      setHealth(nextHealth);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!narrow) setInboxOpen(true);
    else setInboxOpen(!selected);
  }, [narrow, selected]);

  const awaiting = metrics?.by_status?.awaiting_review ?? 0;
  const rate = metrics?.approved_unchanged_rate;
  const offline = health?.model.startsWith("offline") ?? false;

  const onStarted = (id: string) => {
    setComposing(false);
    setSelected(id);
    setFilter("all");
    toast("Case started");
    void refresh();
  };

  const resetDemo = async () => {
    const count = health?.snapshot_cases ?? 0;
    if (!window.confirm(`Restore the ${count} demo cases? Cases created since will be removed.`)) return;
    try {
      const result = await api.resetDemo();
      const next = await api.cases();
      setCases(next);
      const opened = health?.s01_case_id ? next.find((item) => item.id === health.s01_case_id) : undefined;
      setSelected(opened?.id ?? null);
      setFilter("needs_review");
      toast(`Demo restored, ${result.cases} cases`);
      void refresh();
    } catch (err) {
      toast((err as Error).message);
    }
  };

  const retryCase = async (id: string) => {
    try {
      await api.retryCase(id);
      setSelected(id);
      setFilter("all");
      toast("Retrying");
      void refresh();
    } catch (err) {
      toast((err as Error).message);
    }
  };

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>
            R
          </span>
          <span className="brand-name">ResolveDesk</span>
          <span className="sub">Meridian Motors</span>
        </div>
        {health && (
          <Tooltip
            text={
              offline ? (
                <>
                  Rule-based triage, no LLM
                  <span className="tip-meta">Prompt {health.prompt_version}</span>
                </>
              ) : (
                <>Prompt {health.prompt_version}</>
              )
            }
          >
            <span className={offline ? "pill warn" : "pill"} tabIndex={0}>
              {offline ? "Offline mode" : health.model}
            </span>
          </Tooltip>
        )}
        <div className="kpis">
          <div className="kpi">
            <Inbox aria-hidden />
            <span className="kpi-value">{metrics?.total ?? 0}</span>
            <span className="kpi-label">Cases</span>
          </div>
          <button
            type="button"
            className={filter === "needs_review" ? "kpi button on" : "kpi button"}
            onClick={() => setFilter("needs_review")}
            aria-pressed={filter === "needs_review"}
          >
            <ClipboardList aria-hidden />
            <span className="kpi-value">{awaiting}</span>
            <span className="kpi-label">Needs review</span>
          </button>
          <div className="kpi">
            <Check aria-hidden />
            <span className="kpi-value">{rate == null ? "–" : `${Math.round(rate * 100)}%`}</span>
            <span className="kpi-label">Approved unchanged</span>
          </div>
          <div className="kpi">
            <ShieldAlert aria-hidden />
            <span className="kpi-value">{metrics?.injection_flags ?? 0}</span>
            <span className="kpi-label">Injection flags</span>
          </div>
        </div>
        {health?.demo_mode && (
          <button type="button" className="btn" onClick={() => void resetDemo()}>
            Reset demo
          </button>
        )}
        <button type="button" className="btn" onClick={() => setGraphOpen(true)}>
          <Workflow aria-hidden /> Pipeline
        </button>
      </header>

      {error && (
        <div className="banner error">
          <span>Something went wrong: {error}</span>
          <button type="button" className="btn" onClick={() => void refresh()}>
            Retry
          </button>
        </div>
      )}

      <div className="body">
        <aside className={inboxOpen ? "sidebar" : "sidebar collapsed"}>
          <div className="side-actions">
            <button
              type="button"
              className="btn primary block"
              onClick={() => {
                setComposing(true);
              }}
            >
              + New complaint
            </button>
            {narrow && (
              <button type="button" className="btn block" onClick={() => setInboxOpen((open) => !open)} aria-expanded={inboxOpen}>
                {inboxOpen ? "Hide inbox" : "Show inbox"}
              </button>
            )}
          </div>
          <CaseList
            cases={cases}
            loading={loading}
            filter={filter}
            onFilter={setFilter}
            selected={selected}
            onSelect={(id) => {
              setSelected(id);
              setComposing(false);
            }}
            onRetry={(id) => void retryCase(id)}
          />
        </aside>
        <main className="main">
          {selected ? (
            <CaseView key={selected} caseId={selected} onChanged={() => void refresh()} />
          ) : (
            <div className="empty">
              <h2>Complaints, triaged and drafted. Sent only by a person.</h2>
              <p>Start with New complaint and pick a sample email, or open a case from the inbox.</p>
            </div>
          )}
        </main>
      </div>
      {composing && <NewCase onStarted={onStarted} onClose={() => setComposing(false)} />}
      {graphOpen && <GraphModal onClose={() => setGraphOpen(false)} />}
    </div>
  );
}

function GraphModal({ onClose }: { onClose: () => void }) {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api
      .graph()
      .then((graph) => setText(graph.mermaid))
      .catch((err: unknown) => setError((err as Error).message));
  }, []);
  return (
    <Modal title="Pipeline" onClose={onClose}>
      {error && <div className="banner error">{error}</div>}
      {text == null && !error && <p className="muted">Loading graph…</p>}
      {text && (
        <pre className="code">
          <code>{text}</code>
        </pre>
      )}
    </Modal>
  );
}
