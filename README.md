# HID Hackathon - Face Liveness & Identity Kiosk

Decides **ALLOW / REQUEST VERIFICATION / DENY** from a ~1 s webcam burst, in well under 2 s.
Everything runs locally on CPU with pretrained models; identity can be switched to Amazon Rekognition later.

## Run

**Windows: double-click `start.bat`.**
- **First run:** it creates `.venv`, installs the packages, downloads the models (~450 MB including the one-time PyTorch install for the export) and exports MiniFASNet to ONNX.
- **Every run:** it starts the server and opens http://localhost:8000.
- **Stopping:** press Ctrl+C or close the window.
- **Options:**
  - `set PORT=8010` before running uses another port.
  - `set NO_BROWSER=1` stops it opening the browser.
  - Deleting `.venv\.installed` forces a package reinstall.

Manual equivalent:

```bash
py -3.12 -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python scripts/get_models.py
.venv/Scripts/pip install torch onnx --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python scripts/export_minifas_onnx.py
.venv/Scripts/python -m uvicorn app:app --app-dir backend --port 8000
```

Open http://localhost:8000 in Chrome or Edge, which need camera access.

| Page | Purpose |
|---|---|
| `/enroll.html` | Enroll a person (2 s capture, 5 ArcFace templates) |
| `/` | Kiosk: press Start or Space; "Show details" shows every signal |
| `/record.html` | Record labelled test clips (live / print / screen / video / mask) for the evaluation |

## Pipeline

1. **Capture**: 10 frames over 1 s, sent as JPEGs.
2. **Detect + landmarks + quality**: YuNet face box, then MediaPipe 478-point mesh, blendshapes and head pose.
   - Quality checks: blur, brightness, face size, pose, multiple faces.
3. **Liveness**, each check scored 0-1:
   - **MiniFASNet V2 + V1SE** (pretrained anti-spoof CNNs)
   - **Flatness**: optical-flow tracks inside the face. If one homography explains all the motion, the face is a flat photo or screen.
   - **Moire**: whitened FFT spectrum peaks, which come from a screen's pixel grid.
   - **Glare**: saturated specular pixels.
   - **Blink**: a bonus only.
4. **Identity**: ArcFace `w600k_r50` cosine similarity, 1:N or 1:1 with a badge.
5. **Fusion** (`backend/config.yaml`): a weighted liveness score plus hard vetoes, then:
   - ALLOW: live and matched
   - DENY: spoof, veto, no match, or a failed challenge
   - REQUEST VERIFICATION: borderline
6. **Verification flow**:
   - First, a random one-time challenge (blink, turn left or turn right).
   - If still borderline, a mock OTP sent to the enrolled phone. It appears on the "demo phone" panel.

## Models and licences
- YuNet: OpenCV Model Zoo (MIT)
- MediaPipe Face Landmarker: Google (Apache-2.0)
- MiniFASNet: Minivision Silent-Face-Anti-Spoofing (Apache-2.0)
- ArcFace `w600k_r50`: InsightFace buffalo_l. **Non-commercial research use only**; it is replaced by Rekognition in the AWS version.
