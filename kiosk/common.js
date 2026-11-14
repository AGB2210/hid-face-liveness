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

/** Like captureBurst, but uploads frames 0..n-2 to the server *while* capturing, so the
 *  network time overlaps with the capture. The last frame is returned for the verify call. */
export async function captureStreamed(video, n, durationMs, quality = 0.92, onProgress) {
  const burstId = crypto.randomUUID();
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  const step = durationMs / Math.max(1, n - 1);
  const t0 = performance.now();
  const uploads = [];
  let last = null;
  for (let i = 0; i < n; i++) {
    await sleepUntil(t0 + i * step);
    ctx.drawImage(video, 0, 0);
    const blob = new Promise((r) => canvas.toBlob(r, "image/jpeg", quality));
    if (i < n - 1) {
      uploads.push(blob.then((b) => {
        const fd = new FormData();
        fd.append("idx", i);
        fd.append("frame", b, `f${i}.jpg`);
        return fetch(`/api/burst/${burstId}/frame`, { method: "POST", body: fd }).then((r) => r.ok).catch(() => false);
      }));
    } else {
      last = blob;
    }
    onProgress?.((i + 1) / n);
  }
  const lastBlob = await last;
  return { burstId, lastBlob, captureMs: performance.now() - t0, n, uploads };
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
    const via = id.api_ms !== undefined ? ` · Rekognition ${id.api_ms} ms` : "";
    rows.push(`<tr><td><b>Identity (${esc(id.mode)})</b></td><td>${bar(id.similarity)}</td><td class="muted small">${esc(id.error || id.name || "no match")}${esc(via)} ${esc(JSON.stringify(id.scores || {}))}</td></tr>`);
  }
  if (res.occlusion && res.occlusion.eye_ratio !== undefined) {
    const o = res.occlusion;
    rows.push(`<tr><td>Accessories</td><td>${o.sunglasses || o.mask ? [o.sunglasses && "sunglasses", o.mask && "face mask"].filter(Boolean).join(", ") : "none detected"}</td><td class="muted small">eyes/cheeks ${o.eye_ratio} · lip redness ${o.lip_redness} · lower-face colour shift ${o.lower_dist}</td></tr>`);
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

/* ---------------------------------------------------------------- staff console */
// Two separate apps share these helpers:
//   /          the kiosk - public, recognition only, no links to the back office
//   /console/  the staff console - its own sign-in screen, then role-filtered pages
const CONSOLE = [
  ["enroll", "/console/enroll.html", "Enroll", "manage_people"],
  ["policy", "/console/policy.html", "Policy", "view_policy"],
  ["audit", "/console/audit.html", "Audit", "view_audit"],
  ["record", "/console/record.html", "Record", "manage_people"],
  ["report", "/console/report.html", "Results", "view_results"],
];

export async function getMe() {
  try { return await (await fetch("/api/admin/me")).json(); } catch { return { can: {} }; }
}

/** First console page this role may open (where sign-in lands). */
export function homeFor(me) {
  const first = CONSOLE.find(([, , , perm]) => me.can?.[perm]);
  return first ? first[1] : "/console/";
}

export async function signOut() {
  await fetch("/api/admin/logout", { method: "POST" });
  location.href = "/console/";
}

/** Console header nav: the pages this role may use, who is signed in, kiosk link, sign out. */
export async function staffNav(active) {
  const me = await getMe();
  const can = me.can || {};
  const links = CONSOLE.filter(([k, , , perm]) => can[perm] || (k === "enroll" && me.bootstrap))
    .map(([k, href, label]) => `<a href="${href}"${k === active ? ' class="active"' : ""}>${label}</a>`).join("");
  const right = me.user_id
    ? `<span class="who">${esc(me.name || me.user_id)} · ${esc(me.role_name)}</span><a href="/" target="_blank" rel="noopener">Kiosk ↗</a><a href="#" id="signOut">Sign out</a>`
    : `<a href="/console/">Sign in</a>`;
  const nav = document.getElementById("nav");
  if (nav) {
    nav.innerHTML = links + `<span class="navsep"></span>` + right;
    document.getElementById("signOut")?.addEventListener("click", (e) => { e.preventDefault(); signOut(); });
  }
  return me;
}

/** Console page gate: signed-out visitors go to the sign-in screen; signed-in staff without
 *  the permission see a notice. Returns true when the page may load. */
export function guard(me, permission, { firstRun = false } = {}) {
  if (me.can?.[permission] || (firstRun && me.bootstrap)) return true;
  if (!me.user_id) {
    location.replace(`/console/?next=${encodeURIComponent(location.pathname)}`);
    return false;
  }
  const main = document.querySelector("main") || document.body;
  document.querySelectorAll("main ~ section, section.wide").forEach((el) => el.remove());
  main.className = "";
  main.innerHTML = `<section class="card" style="max-width:520px;margin:10vh auto 0">
    <h2>Not available for your role</h2>
    <p style="margin:0 0 18px">You're signed in as ${esc(me.role_name)}, which doesn't include this page.</p>
    <a class="btn" href="${homeFor(me)}" style="text-decoration:none;display:inline-block">Go to my console</a>
  </section>`;
  return false;
}

export const halt = () => new Promise(() => {}); // stop a page script after guard()

/** Kiosk header: nothing for the public; a quiet console link when staff are signed in. */
export async function kioskNav() {
  const me = await getMe();
  const nav = document.getElementById("nav");
  if (nav && me.user_id) {
    nav.innerHTML = `<span class="who">${esc(me.name || me.user_id)} · ${esc(me.role_name)}</span><a href="${homeFor(me)}">Console</a>`;
  }
  return me;
}
