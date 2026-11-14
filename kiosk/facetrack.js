// Real-time face tracking in the browser (MediaPipe Face Landmarker, WASM).
// Drives the live guidance: is one face centred and close enough, which way is the head turned,
// is the person blinking. The server still makes every security decision.
const VERSION = "0.10.14";
const BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}`;

// Column-major vs row-major element of the 4x4 pose matrix holding R[2][0] (sin of yaw).
// Verified against the Python pipeline on a reference image (see README).
let R20 = 2;

export async function createTracker() {
  const { FilesetResolver, FaceLandmarker } = await import(`${BASE}/vision_bundle.mjs`);
  const files = await FilesetResolver.forVisionTasks(`${BASE}/wasm`);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: "/assets/face_landmarker.task", delegate },
    runningMode: "VIDEO", numFaces: 2, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
  });
  let lm;
  try { lm = await FaceLandmarker.createFromOptions(files, opts("GPU")); }
  catch { lm = await FaceLandmarker.createFromOptions(files, opts("CPU")); }
  let lastTs = 0;
  const detect = (source) => {
    const ts = Math.max(performance.now(), lastTs + 1);
    lastTs = ts;
    return describe(lm.detectForVideo(source, ts));
  };
  return { detect, close: () => lm.close() };
}

export function setMatrixIndex(i) { R20 = i; }

/** Summarise a landmarker result into what the UI needs. Coordinates are in the raw
 *  (un-mirrored) camera frame; yaw > 0 means the person turned to THEIR left. */
export function describe(res) {
  const faces = res.faceLandmarks || [];
  if (!faces.length) return { faces: 0, present: false };
  // biggest face
  let bi = 0, bw = 0;
  faces.forEach((f, i) => {
    const xs = f.map((p) => p.x);
    const w = Math.max(...xs) - Math.min(...xs);
    if (w > bw) { bw = w; bi = i; }
  });
  const f = faces[bi];
  const xs = f.map((p) => p.x), ys = f.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, w = x1 - x0;
  let yaw = null, pitch = null;
  const m = res.facialTransformationMatrixes?.[bi]?.data;
  if (m && m.length === 16) {
    yaw = (Math.asin(Math.max(-1, Math.min(1, -m[R20]))) * 180) / Math.PI;
  }
  const bs = {};
  (res.faceBlendshapes?.[bi]?.categories || []).forEach((c) => (bs[c.categoryName] = c.score));
  const blink = ((bs.eyeBlinkLeft || 0) + (bs.eyeBlinkRight || 0)) / 2;
  const second = faces.length > 1 && faces.some((g, i) => i !== bi && (Math.max(...g.map((p) => p.x)) - Math.min(...g.map((p) => p.x))) > 0.5 * w);
  return {
    faces: faces.length, present: true, multiple: second,
    box: { x: x0, y: y0, w, h: y1 - y0 }, cx, cy, size: w,
    centered: Math.abs(cx - 0.5) < 0.14 && Math.abs(cy - 0.47) < 0.17,
    tooFar: w < 0.2, tooClose: w > 0.62,
    yaw, blink, smile: ((bs.mouthSmileLeft || 0) + (bs.mouthSmileRight || 0)) / 2,
  };
}

/** One short instruction for the person in front of the kiosk, or null when aligned. */
export function hint(s) {
  if (!s || !s.present) return "Step in front of the camera";
  if (s.multiple) return "One person at a time";
  if (s.tooFar) return "Move a little closer";
  if (s.tooClose) return "Move back a little";
  if (!s.centered) return "Centre your face in the frame";
  return null;
}

/** Run the tracker on a video at ~fps, calling onState(state) each time. Returns stop(). */
export function track(tracker, video, onState, fps = 12) {
  let stop = false, last = 0;
  const loop = (t) => {
    if (stop) return;
    if (t - last >= 1000 / fps && video.readyState >= 2) {
      last = t;
      try { onState(tracker.detect(video)); } catch { /* skip a frame */ }
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
  return () => { stop = true; };
}
