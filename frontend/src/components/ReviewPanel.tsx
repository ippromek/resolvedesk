import { CircleCheck, FileText, LoaderCircle, TriangleAlert } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, type CaseDetail, type Proposal } from "../api";
import { formatRelative, useNow } from "../time";

function isPolicyId(text: string): boolean {
  return /^[A-Z]{2,4}-\d+$/.test(text.trim());
}

function storedReviewer(): string {
  try {
    return localStorage.getItem("reviewer") ?? "";
  } catch {
    return "";
  }
}

function draftRows(text: string): number {
  const wrapped = text.split("\n").reduce((total, line) => total + Math.max(1, Math.ceil(line.length / 68)), 0);
  return Math.min(16, Math.max(8, wrapped));
}

function actionLabel(proposal: Proposal): string {
  const branch = proposal.args.branch;
  const place = branch ? `${branch} branch` : "";
  if (proposal.type === "schedule_roadside_pickup") return place ? `Schedule roadside pickup — ${place}` : "Schedule roadside pickup";
  if (proposal.type === "book_safety_inspection") return place ? `Book safety inspection — ${place}` : "Book safety inspection";
  if (proposal.type === "register_recall_repair") return `Register recall repair — ${proposal.args.recall_id ?? ""}`.replace(/ — $/, "");
  if (proposal.type === "open_refund_review") return "Open a refund review";
  if (proposal.type === "escalate_to_branch_manager") return "Escalate to the branch manager";
  if (proposal.type === "offer_courtesy_car") return place ? `Offer a courtesy car — ${place}` : "Offer a courtesy car";
  if (proposal.type === "forward_to_data_protection") return `Forward to data protection — ${proposal.args.request_type ?? ""}`.replace(/ — $/, "");
  return proposal.type;
}

function wordCount(text: string): number {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  return trimmed.split(/\s+/).length;
}

export default function ReviewPanel({
  c,
  onDone,
  onOpenPolicy,
}: {
  c: CaseDetail;
  onDone: (c: CaseDetail) => void;
  onOpenPolicy: (id: string) => void;
}) {
  const s = c.state;
  const original = s.draft?.body ?? "";
  const [text, setText] = useState(original);
  const [reviewer, setReviewer] = useState(storedReviewer);

  useEffect(() => {
    if (storedReviewer()) return;
    api
      .health()
      .then((health) => {
        if (health.demo_mode) setReviewer("Demo");
      })
      .catch(() => undefined);
  }, []);
  const [note, setNote] = useState("");
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");
  const [checked, setChecked] = useState(false);
  const [picked, setPicked] = useState<string[]>([]);
  const [versionsOpen, setVersionsOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState("");
  const previewSeq = useRef(0);
  const now = useNow();
  const awaiting = c.next.includes("human_review");
  const edited = text.trim() !== original.trim();
  const pickedKey = picked.join(",");

  useEffect(() => {
    if (!awaiting || picked.length === 0) {
      setPreview("");
      return;
    }
    const seq = previewSeq.current + 1;
    previewSeq.current = seq;
    const handle = window.setTimeout(() => {
      void api
        .preview(c.id, {
          approved_action_ids: picked,
          body: edited ? text : undefined,
        })
        .then((result) => {
          if (seq === previewSeq.current) setPreview(result.arranged_section);
        })
        .catch(() => {
          if (seq === previewSeq.current) setPreview("");
        });
    }, 300);
    return () => window.clearTimeout(handle);
  }, [awaiting, c.id, edited, pickedKey, text]);
  const cited = s.draft?.cited_policy_ids ?? [];

  const injection = s.guard?.injection_suspected === true;
  const primaryBlocked = injection && !checked;

  const decide = async (decision: "approve" | "edit" | "reject") => {
    if (!reviewer.trim()) {
      setError("Enter your name: every decision is recorded against a person.");
      return;
    }
    if (decision === "edit" && !text.trim()) {
      setError("The reply cannot be empty.");
      return;
    }
    if (decision === "reject" && !reason.trim()) {
      setError("A reason is required to reject a draft.");
      return;
    }
    if (decision !== "reject" && primaryBlocked) return;
    try {
      localStorage.setItem("reviewer", reviewer.trim());
    } catch {
      /* storage unavailable */
    }
    setBusy(true);
    setError(null);
    const noteText = decision === "reject" ? reason.trim() : note.trim() || undefined;
    try {
      const updated = await api.review(c.id, {
        decision,
        reviewer: reviewer.trim(),
        body: decision === "edit" ? text : undefined,
        note: noteText,
        approved_action_ids: decision === "reject" ? [] : picked,
        injection_acknowledged: decision !== "reject" && injection ? checked : false,
      });
      onDone(updated);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const onPrimary = () => {
    void decide(edited ? "edit" : "approve");
  };

  if (!s.draft) {
    let message = "No reply was drafted. This case was closed without one.";
    if (c.status === "processing") message = "The reply has not been drafted yet.";
    else if (c.status === "failed") message = "The reply was not drafted. Retry continues from the failed step.";
    return (
      <section className="card review" id="draft-card">
        <h3>Draft reply</h3>
        <p className="muted">{message}</p>
      </section>
    );
  }

  if (!awaiting) {
    const reviewEvent = [...s.events].reverse().find((event) => event.step === "human_review");
    const when = reviewEvent ? formatRelative(reviewEvent.at, now) : "";
    const who = s.review?.reviewer ?? "a reviewer";
    const decision = s.review?.decision;
    let heading = "Reply sent";
    if (decision === "reject") heading = "Draft rejected";
    return (
      <section className="card review" id="draft-card">
        <h3>{heading}</h3>
        <p className="decision-line">
          {decision === "approve" ? "Approved" : decision === "edit" ? "Edited" : "Rejected"} by {who}
          {when ? ` · ${when}` : ""}
        </p>
        {decision === "reject" && s.review?.note && (
          <p className="muted" dir="auto">
            Reason: {s.review.note}
          </p>
        )}
        {decision === "reject" ? (
          <>
            <p className="muted small">Draft (not sent)</p>
            <pre className="reply unsent" dir="auto">
              {original}
            </pre>
          </>
        ) : (
          <pre className="reply" dir="auto">
            {s.final_body}
          </pre>
        )}
        {(s.action_results ?? []).length > 0 && (
          <ul className="action-results">
            {s.action_results?.map((result) => (
              <li key={result.action_id} className={result.status === "not_approved" ? "muted" : ""}>
                {result.status === "executed"
                  ? `${actionLabel({ id: result.action_id, type: result.type, args: result.args, rationale: "", evidence: [] })} · ${result.ref}`
                  : `${actionLabel({ id: result.action_id, type: result.type, args: result.args, rationale: "", evidence: [] })} · not approved`}
              </li>
            ))}
          </ul>
        )}
      </section>
    );
  }

  const issues = s.grounding?.issues ?? [];
  const grounded = s.grounding?.ok === true;
  const revisions = s.revision_count ?? 0;
  const earlier = (s.draft_history ?? []).slice(0, -1);
  const proposals = s.proposals ?? [];
  const allPicked = proposals.length > 0 && proposals.every((proposal) => picked.includes(proposal.id));
  const actionCount = picked.length;
  const actionWord = actionCount === 1 ? "action" : "actions";
  const primaryLabel = edited ? `Send edited reply · ${actionCount} ${actionWord}` : `Approve & send · ${actionCount} ${actionWord}`;

  return (
    <section
      className="card review"
      id="draft-card"
      onKeyDown={(event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !busy && !rejecting && !primaryBlocked) {
          event.preventDefault();
          onPrimary();
        }
      }}
    >
      <div className="review-scroll">
        <div className="row between wrap">
          <h3>Draft reply</h3>
          {edited && <span className="tag strong">Edited</span>}
        </div>
        {s.grounding &&
          (grounded ? (
            <p className="ground ok">
              <CircleCheck aria-hidden /> Grounded in cited policy
            </p>
          ) : (
            <div className="ground bad">
              <p>
                <TriangleAlert aria-hidden /> {issues.length} grounding {issues.length === 1 ? "issue" : "issues"}
              </p>
              <ul className="issues">
                {issues.map((issue) => (
                  <li key={issue}>{issue}</li>
                ))}
              </ul>
            </div>
          ))}
        {revisions > 0 && (
          <div>
            <button type="button" className="link" aria-expanded={versionsOpen} onClick={() => setVersionsOpen((open) => !open)}>
              Agent revised this draft {revisions}× before review
            </button>
            {versionsOpen && (
              <ol className="versions">
                {earlier.map((version) => (
                  <li key={version.version}>
                    <p className="muted small">Version {version.version}</p>
                    <pre className="reply unsent" dir="auto">
                      {version.body}
                    </pre>
                    {version.issues.length > 0 && (
                      <ul className="issues">
                        {version.issues.map((issue) => (
                          <li key={issue}>{issue}</li>
                        ))}
                      </ul>
                    )}
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}
        {cited.length > 0 && (
          <div className="chips">
            {cited.map((id) => (
              <button key={id} type="button" className="chip-btn" onClick={() => onOpenPolicy(id)}>
                {id}
              </button>
            ))}
          </div>
        )}
        <textarea
          id="draft-editor"
          className="reply-edit"
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={draftRows(text)}
          dir="auto"
          disabled={busy}
          aria-label="Draft reply"
        />
        <div className="row between">
          <span className="muted small">
            {wordCount(text)} {wordCount(text) === 1 ? "word" : "words"}
          </span>
          {edited && (
            <button type="button" className="link" onClick={() => setText(original)} disabled={busy}>
              Reset to draft
            </button>
          )}
        </div>
        <div className="row gap wrap fields">
          <label className="field inline">
            <span>Reviewer</span>
            <input value={reviewer} onChange={(event) => setReviewer(event.target.value)} placeholder="Your name" disabled={busy} />
          </label>
          <label className="field inline grow">
            <span>Note (optional)</span>
            <input value={note} onChange={(event) => setNote(event.target.value)} maxLength={1000} disabled={busy || rejecting} />
          </label>
        </div>
        {rejecting && (
          <label className="field">
            <span>Reason for rejecting</span>
            <textarea
              value={reason}
              onChange={(event) => setReason(event.target.value)}
              rows={3}
              maxLength={1000}
              disabled={busy}
              aria-label="Reason for rejecting"
            />
          </label>
        )}
        {injection && !rejecting && (
          <label className="check">
            <input
              type="checkbox"
              checked={checked}
              onChange={(event) => setChecked(event.target.checked)}
              disabled={busy}
            />
            <span>I have checked this reply against the injection warning</span>
          </label>
        )}
        {proposals.length > 0 && (
          <fieldset className="actions" disabled={busy || rejecting}>
            <legend>Proposed actions</legend>
            <label className="action">
              <input
                type="checkbox"
                checked={allPicked}
                onChange={() => setPicked(allPicked ? [] : proposals.map((proposal) => proposal.id))}
              />
              <span>
                <b>Select all</b>
              </span>
            </label>
            {proposals.map((proposal) => (
              <label key={proposal.id} className="action">
                <input
                  type="checkbox"
                  checked={picked.includes(proposal.id)}
                  onChange={(event) =>
                    setPicked((current) =>
                      event.target.checked ? [...current, proposal.id] : current.filter((id) => id !== proposal.id),
                    )
                  }
                />
                <span>
                  <b>{actionLabel(proposal)}</b>
                  <span className="muted small" dir="auto">
                    {proposal.rationale}
                  </span>
                  <span className="evidence">
                    {proposal.evidence.map((item) =>
                      isPolicyId(item) ? (
                        <span key={item} className="chip">
                          {item}
                        </span>
                      ) : (
                        <span key={item} className="fact-line" dir="auto">
                          <FileText aria-hidden />
                          {item}
                        </span>
                      ),
                    )}
                  </span>
                </span>
              </label>
            ))}
          </fieldset>
        )}
        {preview && (
          <p className="arranged" dir="auto">
            {preview}
          </p>
        )}
      </div>
      {error && <div className="banner error">{error}</div>}
      <div className="decision-bar">
        <span className="muted small">
          {actionCount} {actionWord}
        </span>
        <div className="row end gap">
          {rejecting ? (
            <>
              <button type="button" className="btn" disabled={busy} onClick={() => setRejecting(false)}>
                Cancel
              </button>
              <button type="button" className="btn danger" disabled={busy} onClick={() => void decide("reject")}>
                {busy && <LoaderCircle className="spin" aria-hidden />}
                Confirm reject
              </button>
            </>
          ) : (
            <>
              <button type="button" className="btn" disabled={busy} onClick={() => setRejecting(true)}>
                Reject
              </button>
              <button type="button" className="btn primary" disabled={busy || primaryBlocked} onClick={onPrimary} aria-keyshortcuts="Control+Enter">
                {busy && <LoaderCircle className="spin" aria-hidden />}
                {primaryLabel}
              </button>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
