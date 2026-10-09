import { Search, ShieldAlert } from "lucide-react";
import { useMemo, useState } from "react";
import type { CaseSummary, Severity } from "../api";
import { SEVERITY_RANK, categoryLabel } from "../labels";
import { formatRelative, useNow } from "../time";
import { SeverityBadge } from "./Badges";
import Tooltip from "./Tooltip";

export type InboxFilter = "needs_review" | "all" | "sent" | "closed";

const TABS: { id: InboxFilter; label: string }[] = [
  { id: "needs_review", label: "Review" },
  { id: "all", label: "All" },
  { id: "sent", label: "Sent" },
  { id: "closed", label: "Closed" },
];

function inFilter(filter: InboxFilter, item: CaseSummary): boolean {
  switch (filter) {
    case "needs_review":
      return item.status === "awaiting_review";
    case "all":
      return true;
    case "sent":
      return item.status === "sent";
    case "closed":
      return item.status === "rejected" || item.status === "closed_no_reply";
    default: {
      const neverFilter: never = filter;
      return neverFilter;
    }
  }
}

function shown(filter: InboxFilter, item: CaseSummary): boolean {
  return item.status === "processing" || item.status === "failed" || inFilter(filter, item);
}

function listRank(status: CaseSummary["status"]): number {
  switch (status) {
    case "processing":
    case "failed":
      return 0;
    case "awaiting_review":
      return 1;
    case "sent":
    case "rejected":
    case "closed_no_reply":
      return 2;
    default: {
      const neverStatus: never = status;
      return neverStatus;
    }
  }
}

function severityRank(severity: Severity | null): number {
  if (!severity) return 4;
  return SEVERITY_RANK[severity];
}

interface Props {
  cases: CaseSummary[];
  loading: boolean;
  filter: InboxFilter;
  onFilter: (filter: InboxFilter) => void;
  selected: string | null;
  onSelect: (id: string) => void;
  onRetry?: (id: string) => void;
}

export default function CaseList({ cases, loading, filter, onFilter, selected, onSelect, onRetry }: Props) {
  const [query, setQuery] = useState("");
  const now = useNow();
  const needle = query.trim().toLowerCase();

  const counts = useMemo(() => {
    const base = needle
      ? cases.filter((item) => `${item.subject} ${item.sender}`.toLowerCase().includes(needle))
      : cases;
    return {
      needs_review: base.filter((item) => shown("needs_review", item)).length,
      all: base.filter((item) => shown("all", item)).length,
      sent: base.filter((item) => shown("sent", item)).length,
      closed: base.filter((item) => shown("closed", item)).length,
    };
  }, [cases, needle]);

  const visible = useMemo(() => {
    return cases
      .filter((item) => shown(filter, item))
      .filter((item) => !needle || `${item.subject} ${item.sender}`.toLowerCase().includes(needle))
      .sort((a, b) => {
        const byRank = listRank(a.status) - listRank(b.status);
        if (byRank !== 0) return byRank;
        const bySeverity = severityRank(a.severity) - severityRank(b.severity);
        if (bySeverity !== 0) return bySeverity;
        return b.created_at.localeCompare(a.created_at);
      });
  }, [cases, filter, needle]);

  return (
    <div className="inbox-body">
      <label className="search">
        <span className="sr">Search subject and sender</span>
        <Search aria-hidden />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search subject or sender"
          type="search"
        />
      </label>
      <div className="tabs" role="group" aria-label="Inbox">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            aria-pressed={filter === tab.id}
            className={filter === tab.id ? "tab on" : "tab"}
            onClick={() => onFilter(tab.id)}
          >
            {tab.label}
            <span className="count">{counts[tab.id]}</span>
          </button>
        ))}
      </div>
      {loading ? (
        <div className="skel-list" aria-busy="true" aria-label="Loading cases">
          {Array.from({ length: 6 }, (_, index) => (
            <div key={index} className="skel skel-row" />
          ))}
        </div>
      ) : visible.length === 0 ? (
        <p className="empty-list">{needle ? "No matches" : "No cases"}</p>
      ) : (
        <ul className="caselist">
          {visible.map((item) => (
            <li key={item.id} className="case-row">
              <button
                type="button"
                className={selected === item.id ? "caseitem active" : "caseitem"}
                data-severity={item.severity ?? "none"}
                aria-current={selected === item.id ? "true" : undefined}
                onClick={() => onSelect(item.id)}
              >
                <span className="subject" dir="auto">
                  {item.subject}
                </span>
                <span className="meta-line">
                  <span className="sender">{item.sender}</span>
                  <span aria-hidden>·</span>
                  <span>{formatRelative(item.created_at, now)}</span>
                </span>
                <span className="tag-row">
                  {(item.status === "failed" || item.status === "processing") && (
                    <span className={`badge status-${item.status}`}>{item.status === "failed" ? "Failed" : "Processing"}</span>
                  )}
                  {item.severity && <SeverityBadge severity={item.severity} />}
                  {item.category && (
                    <span className="tag clip" title={categoryLabel(item.category)}>
                      {categoryLabel(item.category)}
                    </span>
                  )}
                </span>
              </button>
              {item.status === "failed" && onRetry && (
                <button
                  type="button"
                  className="btn"
                  onClick={(event) => {
                    event.stopPropagation();
                    onRetry(item.id);
                  }}
                >
                  Retry
                </button>
              )}
              {item.injection_suspected && (
                <span className="inject-flag">
                  <Tooltip text="Suspected prompt injection">
                    <span className="icon-focus" tabIndex={0} aria-label="Suspected prompt injection">
                      <ShieldAlert aria-hidden />
                    </span>
                  </Tooltip>
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
