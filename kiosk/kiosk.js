import { captureStreamed, esc, framesForm, getConfig, postForm, renderDebug, staffNav, startCamera } from "./common.js";
import { createTracker, hint, track } from "./facetrack.js";

const $ = (id) => document.getElementById(id);
const video = $("video"), cam = $("cam"), banner = $("banner"), progress = $("progress");
const decisionEl = $("decision"), startBtn = $("start"), instruction = $("instruction");
let cfg, busy = false, resetTimer = null, otpSession = null;
let staff = false, lastStatus = null, latest = null, armed = true, alignedSince = null, absentSince = null, lastOcclusion = null;
const CUE_LABEL = { blink: "Blink", turn_left: "Turn left", turn_right: "Turn right" };

const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};

const STATE = {
  ALLOW: { cls: "allow", word: "Allow", note: "Access granted" },
  DENY: { cls: "deny", word: "Deny", note: "Access denied" },
  REQUEST_VERIFICATION: { cls: "verify", word: "Step-up", note: "Step-up verification required" },
};
const VERDICT = { inside: "Within window", grace: "Grace period", outside: "Outside window" };

function setCamState(state) {
  cam.classList.remove("capturing", "allow", "deny", "verify");
  if (state) cam.classList.add(state);
}

/* ---------------------------------------------------------------- context: zone + clock */
function context() {
  const live = $("liveTime").checked;
  return { zone: $("zone").value, at: live ? "" : $("at").value };
}

function initContext() {
  const zones = cfg.zones || {};
  $("zone").innerHTML = Object.entries(zones)
    .map(([id, z]) => `<option value="${esc(id)}">${esc(z.name)}${z.security === "high" ? " · high security" : ""}</option>`).join("");
  $("zone").value = store.get("hid.zone") in zones ? store.get("hid.zone") : cfg.default_zone;
  $("zone").addEventListener("change", () => store.set("hid.zone", $("zone").value));
  const pad = (n) => String(n).padStart(2, "0");
  const d = new Date();
  $("at").value = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  const sync = () => { $("at").disabled = $("liveTime").checked; };
  $("liveTime").addEventListener("change", sync);
  sync();
}

/* ---------------------------------------------------------------- central-server status */
async function refreshStatus() {
  try {
    const s = await (await fetch("/api/node/status")).json();
    lastStatus = s;
    const off = s.mode === "offline";
    const el = $("netStatus");
    el.className = `net ${off ? "off" : "on"}`;
    const role = s.role === "edge" ? "Central server" : "Network";
    el.innerHTML = off
      ? `<i></i>Offline · fail-safe mode${s.pending ? ` · ${s.pending} queued` : ""}`
      : `<i></i>Online${s.role === "edge" ? " · central reachable" : ""}${s.last_reconnect_synced && s.mode === "online" ? ` · synced ${s.last_reconnect_synced}` : ""}`;
    el.title = `${role}: ${s.central || "none configured"} · kiosk ${s.kiosk} · policy: ${s.policy_source}${s.error ? " · " + s.error : ""}`;
    $("simulate").textContent = s.simulated ? "Restore network" : "Simulate network loss";
    $("simulate").dataset.on = s.simulated ? "1" : "";
  } catch {
    $("netStatus").className = "net off";
    $("netStatus").innerHTML = "<i></i>Kiosk server unreachable";
  }
}

/* ---------------------------------------------------------------- decision rendering */
function showDecision(res, totalMs) {
  const st = STATE[res.decision] || { cls: "", word: "", note: "" };
  decisionEl.className = `decision ${st.cls}`;
  const t = res.timing_ms || {};
  const c = res.context;
  const rows = [];
  if (totalMs) rows.push(["Decision time", `${(totalMs / 1000).toFixed(2)} s`]);
  if (res.confidence !== undefined) rows.push(["Confidence", res.confidence.toFixed(2)]);
  if (res.identity?.name) rows.push(["Identity", esc(res.identity.name) + (c?.role ? ` · ${esc(c.role)}` : "")]);
  if (res.identity?.similarity !== undefined && res.identity?.user_id) rows.push(["Face match", res.identity.similarity.toFixed(2)]);
  if (res.liveness?.score !== undefined) rows.push(["Liveness", res.liveness.score.toFixed(2)]);
  if (c) {
    rows.push(["Zone", esc(c.zone_name) + (c.security === "high" ? " · high security" : "")]);
    rows.push(["Schedule", `${VERDICT[c.verdict] || c.verdict} · ${esc(c.time)}`]);
  }
  const occ = res.occlusion;
  if (occ && (occ.sunglasses || occ.mask)) rows.push(["Accessories", [occ.sunglasses && "sunglasses", occ.mask && "face mask"].filter(Boolean).join(", ") + " detected"]);
  if (res.challenge) rows.push(["Live cue", `${CUE_LABEL[res.challenge.action] || res.challenge.action} · ${res.challenge.passed ? "passed" : "not detected"}`]);
  if (res.mode) rows.push(["Mode", res.mode === "offline" ? "Offline fail-safe" : "Online"]);
  const network = Math.max(0, Math.round(totalMs - (res.capture_ms || 0) - (t.processing || 0)));
  decisionEl.innerHTML = `
    <div class="eyebrow">${esc(st.note)}</div>
    <div class="big">${st.word}</div>
    <p class="why">${(res.reasons || []).map(esc).join("<br>")}</p>
    ${rows.length ? `<dl class="readout">${rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>` : ""}
    ${totalMs ? `<p class="breakdown">capture ${Math.round(res.capture_ms || 0)} ms · server ${Math.round(t.processing || 0)} ms · network ${network} ms</p>` : ""}`;
  setCamState(st.cls);
  if (staff) $("debug").innerHTML = renderDebug(res);
}

function idle(msg = "Look at the camera and press Start.") {
  decisionEl.className = "decision idle";
  decisionEl.innerHTML = `<div class="eyebrow">Decision</div><div class="big">Ready</div><p class="why">${esc(msg)}</p>`;
  banner.textContent = "Align your face within the frame";
  setCamState(null);
  instruction.style.display = "none";
  $("otpCard").classList.add("hidden");
  otpSession = null;
}

function scheduleReset(ms = 8000) {
  clearTimeout(resetTimer);
  resetTimer = setTimeout(() => !busy && !otpSession && idle(), ms);
}

async function burst(n, durationMs) {
  setCamState("capturing");
  progress.style.transition = "none";
  progress.style.width = "0";
  const out = await captureStreamed(video, n, durationMs, cfg.capture.jpeg_quality, (p) => {
    progress.style.transition = "width .1s linear";
    progress.style.width = `${p * 100}%`;
  });
  setTimeout(() => (progress.style.width = "0"), 300);
  return out;
}

/* ---------------------------------------------------------------- flow */
async function runCheck() {
  if (busy) return;
  armed = false;
  busy = true;
  startBtn.disabled = true;
  clearTimeout(resetTimer);
  $("otpCard").classList.add("hidden");
  otpSession = null;
  try {
    decisionEl.className = "decision idle";
    decisionEl.innerHTML = `<div class="eyebrow">Decision</div><div class="big">Checking</div><p class="why">Capturing and analysing.</p>`;
    // Risk-based friction: high-security zones and offline mode get a random live cue up front
    const zoneSec = cfg.zones?.[context().zone]?.security;
    let cue = null;
    if (zoneSec === "high" || lastStatus?.mode === "offline") {
      // eyes hidden last time (sunglasses)? ask for a head turn instead of a blink
      cue = await postForm("/api/cue", framesForm([], { zone: context().zone, avoid: lastOcclusion?.sunglasses ? "blink" : "" }));
      instruction.textContent = `Get ready: ${cue.instruction}`;
      instruction.style.display = "block";
      await new Promise((r) => setTimeout(r, 900));
      instruction.textContent = `${cue.instruction} now`;
      banner.textContent = cue.instruction;
    } else {
      banner.textContent = "Hold still";
    }
    const shot = cue ? await burst(cue.frames, cue.duration_ms) : await burst(cfg.capture.frames, cfg.capture.duration_ms);
    instruction.style.display = "none";
    banner.textContent = "Analysing";
    const t0 = performance.now() - shot.captureMs;
    const res = await postForm("/api/verify", framesForm([shot.lastBlob], {
      claimed_id: $("claimed").value, capture_ms: shot.captureMs, burst_id: shot.burstId, n_frames: shot.n,
      challenge_id: cue ? cue.id : "", ...context() }));
    const total = performance.now() - t0;
    postForm("/api/timing", framesForm([], { total_ms: total })).catch(() => {});
    lastOcclusion = res.occlusion || null;
    showDecision(res, total);
    await handleNext(res);
  } catch (e) {
    idle(`Error: ${e.message}`);
  } finally {
    busy = false;
    startBtn.disabled = false;
    refreshStatus();
  }
}

async function handleNext(res) {
  if (res.decision !== "REQUEST_VERIFICATION") {
    banner.textContent = res.decision === "ALLOW" ? "Access granted" : "Access denied";
    scheduleReset();
    return;
  }
  if (res.challenge_request) {
    const ch = res.challenge_request;
    banner.textContent = "Extra check needed";
    instruction.textContent = ch.instruction;
    instruction.style.display = "block";
    await new Promise((r) => setTimeout(r, 900)); // time to read the instruction
    banner.textContent = ch.instruction;
    const shot = await burst(ch.frames, ch.duration_ms);
    instruction.style.display = "none";
    banner.textContent = "Analysing";
    const t0 = performance.now() - shot.captureMs;
    const res2 = await postForm("/api/verify", framesForm([shot.lastBlob], {
      challenge_id: ch.id, capture_ms: shot.captureMs, burst_id: shot.burstId, n_frames: shot.n, ...context() }));
    showDecision(res2, performance.now() - t0);
    return handleNext(res2);
  }
  if (res.otp_request) {
    const o = res.otp_request;
    otpSession = o.session_id;
    const app = o.methods.includes("authenticator"), sms = o.methods.includes("sms");
    banner.textContent = "Second factor required";
    $("otpSent").textContent = app && sms ? `Enter the code from your authenticator app, or the code sent to ${o.sent_to}.`
      : app ? "Enter the 6-digit code from your authenticator app. It works without a network connection."
      : `Enter the code sent to ${o.sent_to}.`;
    $("otpCard").classList.remove("hidden");
    $("phoneCard").classList.toggle("hidden", !(sms && staff));
    $("otpCode").value = "";
    $("otpCode").focus();
    if (sms && staff) refreshPhone();
    return;
  }
  banner.textContent = (res.reasons || [])[0] || "Please try again";
  scheduleReset(6000);
}

$("otpForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!otpSession) return;
  const res = await postForm("/api/otp/verify", framesForm([], { session_id: otpSession, code: $("otpCode").value }));
  if (res.retry) {
    $("otpSent").textContent = "Wrong code. Try again.";
    $("otpCode").value = "";
    return;
  }
  otpSession = null;
  $("otpCard").classList.add("hidden");
  showDecision({ ...res, timing_ms: {}, quality: {} });
  banner.textContent = res.decision === "ALLOW" ? "Access granted" : "Access denied";
  refreshStatus();
  scheduleReset();
});

async function refreshPhone() {
  const msgs = await (await fetch("/api/otp/outbox")).json();
  $("phone").innerHTML = msgs.length
    ? msgs.slice(0, 3).map((m) => `<div class="sms"><div class="small muted">to ${esc(m.to)}</div>${esc(m.text)}</div>`).join("")
    : '<span class="muted small">No messages</span>';
}


$("simulate").addEventListener("click", async () => {
  await postForm("/api/node/simulate_offline", framesForm([], { on: $("simulate").dataset.on ? "false" : "true" }));
  refreshStatus();
});
$("toggleDebug").addEventListener("click", () => {
  const hidden = $("debugCard").classList.toggle("hidden");
  $("toggleDebug").textContent = hidden ? "Signals" : "Hide signals";
  if (!hidden) $("debugCard").scrollIntoView({ behavior: "smooth", block: "start" });
});
startBtn.addEventListener("click", runCheck);
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !["INPUT", "SELECT"].includes(document.activeElement?.tagName)) {
    e.preventDefault();
    runCheck();
  }
});

function onTrack(s) {
  latest = s;
  if (busy || otpSession) return;
  const now = performance.now();
  if (!s.present) {
    absentSince ??= now;
    alignedSince = null;
    if (now - absentSince > 1200) {
      armed = true;  // the previous person has left: ready for the next one
      if (!decisionEl.classList.contains("idle")) idle();
    }
    if (decisionEl.classList.contains("idle")) banner.textContent = "Step in front of the camera";
    return;
  }
  absentSince = null;
  if (!decisionEl.classList.contains("idle")) return;
  const h = hint(s);
  if (h || !armed || !$("auto").checked) {
    alignedSince = null;
    banner.textContent = h || (armed ? "Align your face within the frame" : "Step away and back to check again, or press Start");
    return;
  }
  alignedSince ??= now;
  banner.textContent = "Hold still";
  if (now - alignedSince > 700) {
    armed = false;
    alignedSince = null;
    runCheck();
  }
}

(async () => {
  const me = await staffNav("kiosk");
  staff = !!me.can?.kiosk_controls;
  cfg = await getConfig();
  initContext();
  // Students and visitors only see the decision: no demo clock, zone picker, signals or SMS demo
  document.querySelectorAll(".staffonly").forEach((el) => el.classList.toggle("hidden", !staff));
  if (!staff) {
    $("zone").classList.add("hidden");
    $("zoneName").classList.remove("hidden");
    $("zoneName").textContent = cfg.zones?.[cfg.default_zone]?.name || cfg.default_zone;
    $("liveTime").checked = true;
  }
  refreshStatus();
  setInterval(refreshStatus, 3000);
  try {
    await startCamera(video, cfg.capture.width);
    idle();
    try {
      const tracker = await createTracker();
      track(tracker, video, onTrack, 10);
    } catch {
      $("auto").checked = false;
      $("auto").disabled = true;  // live guidance unavailable - Start / Space still work
    }
  } catch (e) {
    banner.textContent = `Camera unavailable: ${e.message}`;
  }
})();
