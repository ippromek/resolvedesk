import type { Severity, Status } from "./api";

export const STATUS_LABEL: Record<Status, string> = {
  processing: "Processing",
  awaiting_review: "Awaiting review",
  sent: "Sent",
  rejected: "Rejected",
  closed_no_reply: "Closed, no reply",
  failed: "Failed",
};

export const SEVERITY_LABEL: Record<Severity, string> = {
  low: "Low",
  medium: "Medium",
  high: "High",
  critical: "Critical",
};

export const SEVERITY_RANK: Record<Severity, number> = {
  critical: 0,
  high: 1,
  medium: 2,
  low: 3,
};

const CATEGORY_LABEL: Record<string, string> = {
  vehicle_fault: "Vehicle fault",
  service_experience: "Service experience",
  warranty: "Warranty",
  billing: "Billing",
  sales: "Sales",
  delivery_delay: "Delivery delay",
  data_privacy: "Data privacy",
  spam: "Spam",
  other: "Other",
};

export function categoryLabel(category: string): string {
  return CATEGORY_LABEL[category] ?? category.replace(/_/g, " ");
}

/** A short tag for the sample picker, derived only from the sample id and subject. */
export function sampleTag(id: string, subject: string): string {
  const haystack = `${id} ${subject}`.toLowerCase();
  if (haystack.includes("inject")) return "Injection test";
  if (id.toLowerCase().includes("-ar-") || /[؀-ۿ]/.test(subject)) return "Arabic";
  if (haystack.includes("spam") || haystack.includes("seo") || haystack.includes("winner") || haystack.includes("crypto")) {
    return "Spam";
  }
  if (/brake|airbag|engine|power|recall|فرامل/.test(haystack)) return "Safety";
  if (/charge|invoice|refund|billing|فاتورة/.test(haystack)) return "Billing";
  if (haystack.includes("warrant")) return "Warranty";
  if (/data|marketing|privacy|delete/.test(haystack)) return "Privacy";
  if (haystack.includes("deliver")) return "Delivery";
  if (/sales|finance|trade|test-drive|test drive/.test(haystack)) return "Sales";
  return "Service";
}
