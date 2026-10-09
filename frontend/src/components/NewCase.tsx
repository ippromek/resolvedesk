import { useEffect, useState, type FormEvent } from "react";
import { api, type Sample } from "../api";
import { sampleTag } from "../labels";
import Modal from "./Modal";

export default function NewCase({ onStarted, onClose }: { onStarted: (id: string) => void; onClose: () => void }) {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [picked, setPicked] = useState<string | null>(null);
  const [sender, setSender] = useState("");
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fieldErrors, setFieldErrors] = useState<{ sender?: string; subject?: string; body?: string }>({});

  useEffect(() => {
    api.samples().then(setSamples).catch(() => setSamples([]));
  }, []);

  const pick = (sample: Sample) => {
    setPicked(sample.id);
    setSender(sample.sender);
    setSubject(sample.subject);
    setBody(sample.body);
    setFieldErrors({});
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const next: { sender?: string; subject?: string; body?: string } = {};
    if (sender.trim().length < 3) next.sender = "Enter the sender address (at least 3 characters).";
    if (!subject.trim()) next.subject = "Enter a subject.";
    if (!body.trim()) next.body = "Enter the email body.";
    setFieldErrors(next);
    if (Object.keys(next).length > 0) return;
    setBusy(true);
    setError(null);
    try {
      const created = await api.createCaseAsync({ sender: sender.trim(), subject: subject.trim(), body });
      onStarted(created.id);
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  };

  return (
    <Modal title="New complaint" onClose={onClose} busy={busy}>
      <form onSubmit={(event) => void submit(event)}>
        <p className="muted">Paste an inbound email, or load one of the synthetic samples.</p>
        <div className="field">
          <span>Sample email</span>
          <div className="samples">
            {samples.map((sample) => (
              <button
                key={sample.id}
                type="button"
                className={picked === sample.id ? "sample on" : "sample"}
                onClick={() => pick(sample)}
                disabled={busy}
              >
                <span className="tag">{sampleTag(sample.id, sample.subject)}</span>
                <span className="sample-subject" dir="auto">
                  {sample.subject}
                </span>
              </button>
            ))}
          </div>
        </div>
        <label className="field">
          <span>From</span>
          <input
            value={sender}
            onChange={(event) => setSender(event.target.value)}
            disabled={busy}
            aria-invalid={Boolean(fieldErrors.sender)}
            dir="auto"
          />
          {fieldErrors.sender && <span className="field-error">{fieldErrors.sender}</span>}
        </label>
        <label className="field">
          <span>Subject</span>
          <input
            value={subject}
            onChange={(event) => setSubject(event.target.value)}
            disabled={busy}
            aria-invalid={Boolean(fieldErrors.subject)}
            dir="auto"
          />
          {fieldErrors.subject && <span className="field-error">{fieldErrors.subject}</span>}
        </label>
        <label className="field">
          <span>Body</span>
          <textarea
            value={body}
            onChange={(event) => setBody(event.target.value)}
            rows={6}
            disabled={busy}
            aria-invalid={Boolean(fieldErrors.body)}
            dir="auto"
          />
          {fieldErrors.body && <span className="field-error">{fieldErrors.body}</span>}
        </label>
        {error && <div className="banner error">{error}</div>}
        <div className="row end">
          <button type="submit" className="btn primary" disabled={busy}>
            Run agent
          </button>
        </div>
      </form>
    </Modal>
  );
}
