import { ChevronDown } from "lucide-react";
import type { ReactNode } from "react";

export default function Collapse({
  title,
  open,
  onToggle,
  children,
  meta,
}: {
  title: string;
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
  meta?: ReactNode;
}) {
  return (
    <section className="card">
      <button type="button" className="collapse-btn" aria-expanded={open} onClick={onToggle}>
        <ChevronDown className={open ? "chev open" : "chev"} aria-hidden />
        <span>{title}</span>
        {meta}
      </button>
      {open && <div className="collapse-body">{children}</div>}
    </section>
  );
}
