// Shared camera + capture helpers for all kiosk pages.
export async function getConfig() {
  return (await fetch("/api/config")).json();
}

export async function startCamera(video, width = 960) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: width }, height: { ideal: Math.round(width * 0.75) }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  await new Promise((r) => (video.videoWidth ? r() : video.addEventListener("loadedmetadata", r, { once: true })));
  return stream;
}

const sleepUntil = (t) => new Promise((r) => setTimeout(r, Math.max(0, t - performance.now())));

/** Capture n JPEG frames spread evenly over durationMs. onProgress(0..1) for UI. */
export async function captureBurst(video, n, durationMs, quality = 0.92, onProgress) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  const step = durationMs / Math.max(1, n - 1);
  const t0 = performance.now();
  const pending = [];
  for (let i = 0; i < n; i++) {
    await sleepUntil(t0 + i * step);
    ctx.drawImage(video, 0, 0);
    // toBlob snapshots the canvas now and encodes in the background
    pending.push(new Promise((r) => canvas.toBlob(r, "image/jpeg", quality)));
    onProgress?.((i + 1) / n);
  }
  const blobs = await Promise.all(pending);
  return { blobs, captureMs: performance.now() - t0 };
}

export function framesForm(blobs, fields = {}) {
  const fd = new FormData();
  blobs.forEach((b, i) => fd.append("frames", b, `f${i}.jpg`));
  for (const [k, v] of Object.entries(fields)) if (v !== undefined && v !== null) fd.append(k, v);
  return fd;
}

export async function postForm(url, fd) {
  const r = await fetch(url, { method: "POST", body: fd });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
  return body;
}

export const LABEL_TEXT = {
  ALLOW: "ALLOW",
  DENY: "DENY",
  REQUEST_VERIFICATION: "VERIFY",
};

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

export function bar(v, color) {
  if (v === null || v === undefined) return '<span class="muted">n/a</span>';
  const c = color || (v >= 0.65 ? "var(--allow)" : v >= 0.35 ? "var(--verify)" : "var(--deny)");
  return `<div class="row" style="gap:8px;flex-wrap:nowrap"><div class="bar" style="flex:1"><span style="width:${Math.round(v * 100)}%;background:${c}"></span></div><b>${v.toFixed(2)}</b></div>`;
}

/** Debug table for a /api/verify result. */
export function renderDebug(res) {
  if (!res) return "";
  const rows = [];
  const live = res.liveness;
  if (live) {
    const detail = {
      minifas: (c) => `per-frame ${c.per_frame?.join(", ") ?? ""}`,
      flatness: (c) => (c.ratio !== undefined ? `ratio ${c.ratio}, motion ${c.max_motion}, ${c.pairs} pairs` : c.reason || ""),
      moire: (c) => (c.peak !== undefined ? `peak ${c.peak}` : c.reason || ""),
      glare: (c) => (c.saturated_frac !== undefined ? `saturated ${c.saturated_frac}` : ""),
    };
    const names = { minifas: "Anti-spoof CNN (MiniFASNet x2)", flatness: "3-D motion (flatness)", moire: "Screen pattern (moire)", glare: "Specular glare" };
    for (const k of ["minifas", "flatness", "moire", "glare"]) {
      const c = live.checks[k] || {};
      rows.push(`<tr><td>${names[k]}</td><td>${bar(c.score)}</td><td class="muted small">${esc(detail[k](c))}</td></tr>`);
    }
    rows.push(`<tr><td>Blink seen</td><td>${live.checks.blink?.blinked ? "yes" : "no"}</td><td class="muted small">max ${live.checks.blink?.max ?? "-"}</td></tr>`);
    rows.push(`<tr><td><b>Fused liveness</b></td><td>${bar(live.score)}</td><td class="muted small">${esc(live.vetoes.join("; "))}</td></tr>`);
  }
  if (res.identity) {
    const id = res.identity;
    rows.push(`<tr><td><b>Identity (${esc(id.mode)})</b></td><td>${bar(id.similarity)}</td><td class="muted small">${esc(id.name || "no match")} ${esc(JSON.stringify(id.scores || {}))}</td></tr>`);
  }
  if (res.challenge) {
    rows.push(`<tr><td>Challenge: ${esc(res.challenge.action)}</td><td>${res.challenge.passed ? "passed" : "failed"}</td><td class="muted small">${esc(JSON.stringify(res.challenge))}</td></tr>`);
  }
  const q = res.quality || {};
  rows.push(`<tr><td>Quality</td><td>${q.good_frames}/${q.frames} good</td><td class="muted small">face ${q.face_px}px, blur ${q.blur}, light ${q.brightness} ${esc(JSON.stringify(q.issues || {}))}</td></tr>`);
  const t = res.timing_ms || {};
  rows.push(`<tr><td>Server timing</td><td>${t.processing} ms</td><td class="muted small">decode ${t.decode} · detect/landmarks ${t.detect_landmarks_quality} · liveness ${t.liveness} · identity ${t.identity}</td></tr>`);
  return `<table><tr><th>Check</th><th style="width:38%">Score</th><th>Details</th></tr>${rows.join("")}</table>`;
}
