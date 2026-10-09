import { LoaderCircle, Lock } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api";

interface Turn {
  q: string;
  a?: string;
  tools?: string[];
  error?: string;
}

const SUGGESTIONS = [
  "Summarise this case in two lines",
  "Are there other open cases like this one?",
  "Which policy applies here?",
  "Approve and send this reply",
];

export function readAssistantOpen(): boolean {
  try {
    const stored = localStorage.getItem("assistant-open");
    if (stored === "open") return true;
    if (stored === "closed") return false;
  } catch {
    /* storage unavailable */
  }
  return window.matchMedia("(min-width: 1600px)").matches;
}

export function storeAssistantOpen(open: boolean): void {
  try {
    localStorage.setItem("assistant-open", open ? "open" : "closed");
  } catch {
    /* storage unavailable */
  }
}

export default function AssistantPanel({ caseId, onClose }: { caseId: string; onClose: () => void }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [turns, busy]);

  const ask = async (question: string) => {
    const text = question.trim();
    if (!text || busy) return;
    setBusy(true);
    setQ("");
    setTurns((current) => [...current, { q: text }]);
    try {
      const result = await api.ask(caseId, text);
      setTurns((current) =>
        current.map((turn, index) => (index === current.length - 1 ? { ...turn, a: result.answer, tools: result.tools_used } : turn)),
      );
    } catch (err) {
      setTurns((current) =>
        current.map((turn, index) => (index === current.length - 1 ? { ...turn, error: (err as Error).message } : turn)),
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <aside className="drawer" aria-label="Case assistant">
      <div className="drawer-head">
        <div>
          <div className="row gap">
            <Lock aria-hidden />
            <b>Case assistant</b>
          </div>
          <p className="muted small">Read-only. It can look things up but cannot change a case.</p>
        </div>
        <div className="row gap">
          <button type="button" className="link" onClick={() => setTurns([])} disabled={turns.length === 0 || busy}>
            Clear
          </button>
          <button type="button" className="btn" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
      <div className="drawer-body">
        {turns.length === 0 && (
          <div className="suggestions">
            {SUGGESTIONS.map((suggestion) => (
              <button key={suggestion} type="button" className="suggest" onClick={() => void ask(suggestion)} disabled={busy}>
                {suggestion}
              </button>
            ))}
          </div>
        )}
        {turns.map((turn, index) => (
          <div key={index} className="turn">
            <div className="bubble user">{turn.q}</div>
            {turn.a && (
              <div className="bubble bot">
                <div className="pre" dir="auto">
                  {turn.a}
                </div>
                <div className="tool-row">
                  {turn.tools && turn.tools.length > 0 ? (
                    turn.tools.map((tool) => (
                      <span key={tool} className="tag">
                        {tool}
                      </span>
                    ))
                  ) : (
                    <span className="muted small">no tools used</span>
                  )}
                </div>
              </div>
            )}
            {turn.error && <div className="bubble bot error">{turn.error}</div>}
            {!turn.a && !turn.error && (
              <div className="bubble bot typing">
                <LoaderCircle className="spin" aria-hidden /> Looking that up…
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <form
        className="drawer-input"
        onSubmit={(event) => {
          event.preventDefault();
          void ask(q);
        }}
      >
        <textarea
          value={q}
          onChange={(event) => setQ(event.target.value)}
          placeholder="Ask about this case…"
          disabled={busy}
          rows={2}
          aria-label="Question for the case assistant"
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void ask(q);
            }
          }}
        />
        <button type="submit" className="btn primary" disabled={busy || !q.trim()}>
          Ask
        </button>
      </form>
    </aside>
  );
}
