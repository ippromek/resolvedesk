import type { Severity, Status } from "../api";
import { SEVERITY_LABEL, STATUS_LABEL } from "../labels";

export function StatusBadge({ status }: { status: Status }) {
  return <span className={`badge status-${status}`}>{STATUS_LABEL[status]}</span>;
}

export function SeverityBadge({ severity }: { severity: Severity }) {
  return <span className={`badge sev-${severity}`}>{SEVERITY_LABEL[severity]}</span>;
}
