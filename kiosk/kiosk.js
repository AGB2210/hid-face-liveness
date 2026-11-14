import { captureStreamed, esc, framesForm, getConfig, postForm, renderDebug, startCamera } from "./common.js";

const $ = (id) => document.getElementById(id);
const video = $("video"), cam = $("cam"), banner = $("banner"), progress = $("progress");
const decisionEl = $("decision"), startBtn = $("start"), instruction = $("instruction");
let cfg, busy = false, resetTimer = null, otpSession = null, flowStart = 0;

function setCamState(state) {
  cam.classList.remove("capturing", "allow", "deny", "verify");
  if (state) cam.classList.add(state);
}

const STATE = {
  ALLOW: { cls: "allow", word: "Allow", note: "Access granted" },
  DENY: { cls: "deny", word: "Deny", note: "Access denied" },
  REQUEST_VERIFICATION: { cls: "verify", word: "Verify", note: "Additional verification required" },
};

function showDecision(res, totalMs) {
  const st = STATE[res.decision] || { cls: "", word: "", note: "" };
  decisionEl.className = `decision ${st.cls}`;
  const t = res.timing_ms || {};
  const rows = [];
  if (totalMs) rows.push(["Decision time", `${(totalMs / 1000).toFixed(2)} s`]);
  if (res.confidence !== undefined) rows.push(["Confidence", res.confidence.toFixed(2)]);
  if (res.identity?.name) rows.push(["Identity", esc(res.identity.name)]);
  if (res.identity?.similarity !== undefined && res.identity?.user_id) rows.push(["Face match", res.identity.similarity.toFixed(2)]);
  if (res.liveness?.score !== undefined) rows.push(["Liveness", res.liveness.score.toFixed(2)]);
  const network = Math.max(0, Math.round(totalMs - (res.capture_ms || 0) - (t.processing || 0)));
  decisionEl.innerHTML = `
    <div class="eyebrow">${esc(st.note)}</div>
    <div class="big">${st.word}</div>
    <p class="why">${esc((res.reasons || []).join(". "))}</p>
    ${rows.length ? `<dl class="readout">${rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>` : ""}
    ${totalMs ? `<p class="breakdown">capture ${Math.round(res.capture_ms || 0)} ms · server ${Math.round(t.processing || 0)} ms · network ${network} ms</p>` : ""}`;
  setCamState(st.cls);
  $("debug").innerHTML = renderDebug(res);
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

function scheduleReset(ms = 7000) {
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

async function runCheck() {
  if (busy) return;
  busy = true;
  startBtn.disabled = true;
  clearTimeout(resetTimer);
  $("otpCard").classList.add("hidden");
  try {
    flowStart = performance.now();
    banner.textContent = "Hold still";
    decisionEl.className = "decision idle";
    decisionEl.innerHTML = `<div class="eyebrow">Decision</div><div class="big">Checking</div><p class="why">Capturing and analysing.</p>`;
    const shot = await burst(cfg.capture.frames, cfg.capture.duration_ms);
    const captureMs = shot.captureMs;
    banner.textContent = "Analysing";
    const t0 = performance.now() - captureMs;
    const res = await postForm("/api/verify", framesForm([shot.lastBlob], {
      claimed_id: $("claimed").value, capture_ms: captureMs, burst_id: shot.burstId, n_frames: shot.n }));
    const total = performance.now() - t0;
    postForm("/api/timing", framesForm([], { total_ms: total })).catch(() => {});
    showDecision(res, total);
    await handleNext(res);
  } catch (e) {
    idle(`Error: ${e.message}`);
  } finally {
    busy = false;
    startBtn.disabled = false;
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
    const captureMs = shot.captureMs;
    instruction.style.display = "none";
    banner.textContent = "Analysing";
    const t0 = performance.now() - captureMs;
    const res2 = await postForm("/api/verify", framesForm([shot.lastBlob], {
      challenge_id: ch.id, capture_ms: captureMs, burst_id: shot.burstId, n_frames: shot.n }));
    showDecision(res2, performance.now() - t0);
    return handleNext(res2);
  }
  if (res.otp_request) {
    otpSession = res.otp_request.session_id;
    banner.textContent = "Enter the code sent to your phone";
    $("otpSent").textContent = `A one-time code was sent to ${res.otp_request.sent_to}.`;
    $("otpCard").classList.remove("hidden");
    $("phoneCard").classList.remove("hidden");
    $("otpCode").value = "";
    $("otpCode").focus();
    refreshPhone();
    return;
  }
  banner.textContent = (res.reasons || [])[0] || "Please try again";
  scheduleReset(5000);
}

$("otpForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!otpSession) return;
  const res = await postForm("/api/otp/verify", framesForm([], { session_id: otpSession, code: $("otpCode").value }));
  if (res.retry) {
    $("otpSent").textContent = "Wrong code - try again.";
    $("otpCode").value = "";
    return;
  }
  otpSession = null;
  $("otpCard").classList.add("hidden");
  showDecision({ ...res, timing_ms: {}, quality: {} });
  banner.textContent = res.decision === "ALLOW" ? "Access granted" : "Access denied";
  scheduleReset();
});

async function refreshPhone() {
  const msgs = await (await fetch("/api/otp/outbox")).json();
  $("phone").innerHTML = msgs.length
    ? msgs.slice(0, 3).map((m) => `<div class="sms"><div class="small muted">to ${esc(m.to)}</div>${esc(m.text)}</div>`).join("")
    : '<span class="muted small">No messages</span>';
}

async function loadUsers() {
  const users = await (await fetch("/api/users")).json();
  const sel = $("claimed");
  sel.innerHTML = '<option value="">None, search by face</option>' +
    users.map((u) => `<option value="${esc(u.id)}">${esc(u.name)} (${esc(u.id)})</option>`).join("");
}

$("toggleDebug").addEventListener("click", () => {
  const hidden = $("debugCard").classList.toggle("hidden");
  $("toggleDebug").textContent = hidden ? "Signals" : "Hide signals";
  if (!hidden) $("debugCard").scrollIntoView({ behavior: "smooth", block: "start" });
});
startBtn.addEventListener("click", runCheck);
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && document.activeElement?.tagName !== "INPUT") {
    e.preventDefault();
    runCheck();
  }
});

(async () => {
  cfg = await getConfig();
  await loadUsers();
  try {
    await startCamera(video, cfg.capture.width);
    idle();
  } catch (e) {
    banner.textContent = `Camera unavailable: ${e.message}`;
  }
})();
