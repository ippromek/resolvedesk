import type { ReactNode } from "react";

export default function Tooltip({
  text,
  children,
}: {
  text: ReactNode;
  children: ReactNode;
}) {
  return (
    <span className="tip">
      {children}
      <span className="tip-pop" role="tooltip">
        {text}
      </span>
    </span>
  );
}
