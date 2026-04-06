const apiBaseRaw = import.meta.env.VITE_API_BASE;
const apiBase =
  apiBaseRaw != null && String(apiBaseRaw).trim() !== ""
    ? String(apiBaseRaw).replace(/\/$/, "")
    : "";

const apiTextBaseRaw = import.meta.env.VITE_API_TEXT_BASE;
const apiTextBase =
  apiTextBaseRaw != null && String(apiTextBaseRaw).trim() !== ""
    ? String(apiTextBaseRaw).replace(/\/$/, "")
    : null;

/** Hint when the dev server returns HTML instead of JSON (wrong origin or API down). */
const JSON_BACKEND_HINT =
  "Open the Vite dev URL (5173) and run: python demo-api/app.py (port 5000).";

export function apiImage(path) {
  return `${apiBase}${path}`;
}

/** Text routes live under `/api/text/*` on the same host as image routes unless `VITE_API_TEXT_BASE` is set. */
export function apiText(path) {
  const tail = path.replace(/^\/api/, "") || "/";
  const norm = tail.startsWith("/") ? tail : `/${tail}`;
  if (apiTextBase) return `${apiTextBase}${norm}`;
  return `${apiBase}/api/text${norm}`;
}

/** Multimodal routes live under `/api/mm/*`. */
export function apiMultimodal(path) {
  const tail = path.replace(/^\/api/, "") || "/";
  const norm = tail.startsWith("/") ? tail : `/${tail}`;
  return `${apiBase}/api/mm${norm}`;
}

export function modelDisplayName(m) {
  if (m?.arch) return m.arch;
  return String(m?.id ?? "")
    .replace(/_best$/i, "")
    .replace(/_/g, " ");
}

export function prettyClassName(name) {
  return String(name ?? "").replace(/_/g, " ");
}

function isLikelyApiRequest(url) {
  const s = String(url);
  return s.startsWith("/api");
}

export async function fetchJson(url) {
  const r = await fetch(url);
  const text = await r.text();
  const ct = r.headers.get("content-type") || "";
  if (!ct.includes("application/json")) {
    const looksHtml = /^\s*</.test(text);
    const hint = looksHtml
      ? ` Response was not JSON. ${JSON_BACKEND_HINT} (requested: ${url})`
      : ` (requested: ${url})`;
    throw new Error(`HTTP ${r.status}: expected JSON.${hint}`);
  }
  return JSON.parse(text);
}

export async function fetchJsonOk(url, options) {
  const r = await fetch(url, options);
  const text = await r.text();
  const ct = r.headers.get("content-type") || "";
  if (!ct.includes("application/json")) {
    const looksHtml = /^\s*</.test(text);
    const hint = looksHtml
      ? ` ${JSON_BACKEND_HINT} (requested: ${url})`
      : ` (requested: ${url})`;
    throw new Error(`HTTP ${r.status}: expected JSON.${hint}`);
  }
  const data = JSON.parse(text);
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}
