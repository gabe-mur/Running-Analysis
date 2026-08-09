// Value formatting and label lookups. No DOM, no fetch, no app state, so any
// module can import from here without pulling in a dependency chain.

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;").replaceAll("'", "&#039;");

const pace = (value) => {
  if (!Number.isFinite(value)) return "—";
  let minutes = Math.floor(value);
  let seconds = Math.round((value - minutes) * 60);
  if (seconds === 60) { minutes += 1; seconds = 0; }
  return `${minutes}:${String(seconds).padStart(2, "0")}/mi`;
};
const number = (value, digits = 1) => Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
const dateLabel = (value) => value ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)) : "Unknown date";
const calendarDateLabel = (value) => value ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(new Date(`${value}T12:00:00`)) : "Unknown date";
const daypartLabel = (value) => {
  if (!value) return null;
  const hour = new Date(value).getHours();
  return hour < 12 ? "Morning" : hour < 17 ? "Afternoon" : "Evening";
};
const titleCase = (value) => String(value ?? "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
const trendLabel = (value) => ({
  improving: "Improving", declining: "Declining", stable: "About the same",
  uncertain: "No clear change", insufficient_data: "Not enough data",
}[value] ?? "Not enough data");
const evidenceLabel = (value) => ({
  high: "Strong evidence", moderate: "Some evidence", low: "Limited evidence",
  unavailable: "Not enough data",
}[value] ?? "Not enough data");
const readinessLabel = (value) => ({ ready: "As planned", caution: "Ease if needed", not_ready: "Rest" }[value] ?? titleCase(value));
const dataQualityLabel = (value) => ({
  good: "Complete", partial: "Mostly complete", poor: "Limited", unavailable: "Unavailable",
}[value] ?? titleCase(value));
const gpsQualityLabel = (value) => ({
  gps_complete: "GPS complete", gps_partial: "Some GPS missing", gps_missing: "No GPS",
}[value] ?? titleCase(value));
const durationLabel = (seconds) => {
  if (!Number.isFinite(seconds)) return "—";
  const hours = Math.floor(seconds / 3600); const minutes = Math.floor((seconds % 3600) / 60); const secs = seconds % 60;
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}` : `${minutes}:${String(secs).padStart(2, "0")}`;
};
const durationSeconds = (value) => {
  if (!value) return null;
  const parts = String(value).split(":").map(Number);
  if (parts.some((part) => !Number.isFinite(part)) || parts.length < 2 || parts.length > 3) return null;
  return parts.length === 2 ? parts[0] * 60 + parts[1] : parts[0] * 3600 + parts[1] * 60 + parts[2];
};
const datetimeLocalValue = (value) => {
  const date = new Date(value);
  const pad = (part) => String(part).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
};

export const fitnessWindowLabel = (days) => ({
  14: "2 weeks", 28: "4 weeks", 42: "6 weeks", 56: "8 weeks",
  90: "3 months", 180: "6 months", 365: "1 year",
}[days] ?? `${days} days`);

export {
  escapeHtml, pace, number, dateLabel, calendarDateLabel, daypartLabel, titleCase,
  trendLabel, evidenceLabel, readinessLabel, dataQualityLabel, gpsQualityLabel,
  durationLabel, durationSeconds, datetimeLocalValue,
};
