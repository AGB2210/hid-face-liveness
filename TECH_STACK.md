# Tech Stack - HID Face Liveness & Access Kiosk

## Application

| Layer | Technology | Version | Role |
|---|---|---|---|
| Language (backend) | Python | 3.12 | Server, pipeline, scripts |
| Web framework | FastAPI + Uvicorn | latest | REST API, static pages, streaming frame upload |
| Frontend | Plain HTML / CSS / JavaScript (ES modules) | - | Kiosk, Enroll, Policy, Audit, Record, Results pages, with no build step |
| Typography | IBM Plex Sans / IBM Plex Mono | Google Fonts | Minimal, instrument-style UI |
| Camera capture | `getUserMedia` + canvas JPEG encoding | browser | 10-frame burst, streamed to the server while capturing |
| In-browser face tracking | MediaPipe Tasks Vision (WASM) | 0.10.14 | Live pose guidance, auto-start, head-turn meter |

## Computer vision and AI models (all pretrained, no training)

| Function | Model / method | Source / licence | Runs on |
|---|---|---|---|
| Face detection | YuNet (`face_detection_yunet_2023mar`) | OpenCV Model Zoo, MIT | Server (OpenCV 5.0) |
| Landmarks, head pose, blink | MediaPipe Face Landmarker (478 points + blendshapes) | Google, Apache-2.0 | Server (Python 1.2.3) and browser (JS) |
| Passive anti-spoof | MiniFASNet V2 + V1SE (exported to ONNX) | Minivision Silent-Face-Anti-Spoofing, Apache-2.0 | Server (ONNX Runtime 1.32, CPU) |
| 3-D motion test | Optical-flow tracks + homography fit (flat surface vs real head) | Own code (OpenCV) | Server |
| Screen detection | Whitened FFT moiré peak + specular glare (glasses lenses excluded) | Own code | Server |
| Occlusion | Sunglasses / face-mask detection from mesh regions vs the person's own skin | Own code | Server |
| Face recognition (local / offline) | ArcFace `w600k_r50` | InsightFace buffalo_l (non-commercial weights) | Server / edge kiosk |
| Face recognition (cloud) | Amazon Rekognition (IndexFaces, SearchFacesByImage) | AWS | us-east-1 |
| Active liveness | Random server-issued cues: blink, turn left/right (one-time) | Own code | Server-enforced |

## Security and access logic

| Concern | Technology | Notes |
|---|---|---|
| Context-aware access | YAML campus schedule (roles × zones × weekly windows, grace, holidays, expiry) | `backend/access_policy.yaml`, timezone-aware (`zoneinfo` + `tzdata`) |
| Second factor | TOTP authenticator app (`pyotp`, QR via `qrcode`), mock SMS OTP | TOTP works offline |
| Offline fail-safe | Edge/central roles, health monitor, on-device templates, mandatory cue + TOTP | Never fails open; high-security zones close offline |
| Audit log | SQLite, SHA-256 hash chain per kiosk, idempotent sync | No images or face templates stored in audit |
| Access to the site | Shared-password login (HTTP-only cookie) | Sync API uses a separate shared key header |
| Data storage | SQLite (`data/hid.db`) | Users, templates, attempts, audit |

## Cloud and infrastructure (AWS, us-east-1)

| Service | Use |
|---|---|
| Amazon EC2 (Ubuntu 24.04, c6i.xlarge, CPU only) | Central server: API, liveness, policy, audit ingest |
| Amazon CloudFront | HTTPS endpoint (needed for browser camera access), caching disabled |
| Amazon S3 | Code package and model artefacts for deployment |
| Amazon Rekognition | Cloud face collection and search |
| AWS IAM | Instance role (Rekognition + SSM) |
| AWS Systems Manager (SSM) | In-place updates without SSH |
| Security group | App port open only to CloudFront and the deployer's IP |

## Tooling

| Tool | Purpose |
|---|---|
| `start.bat` | One-click local setup (venv, packages, models, ONNX export) and launch; edge-kiosk mode via `edge.bat` |
| `deploy.bat` / `scripts/deploy_aws.py` (boto3) | Deploy, update, status, destroy, generate edge config |
| `scripts/get_models.py`, `scripts/export_minifas_onnx.py` (PyTorch, export only) | Model download and conversion |
| `scripts/sync_rekognition.py` | Push locally enrolled people to Rekognition |
| `eval/run_eval.py` + Results page | APCER / BPCER / ACER (ISO/IEC 30107-3), FAR / FRR, latency, ablation, accessory impact, threshold sweeps |
| AWS CLI v2 (`aws login`, profile `hid`) | Credentials for deployment and Rekognition |
| Git + GitHub (private repo) | Version control |

## Performance (measured)

| Path | Decision time |
|---|---|
| Local laptop (CPU) | ~1.35-1.40 s (1.0 s capture + ~0.35 s processing) |
| Cloud from India via CloudFront (frame streaming) | ~1.40-1.65 s |
| High-security / offline (with live cue) | ~3 s by design (extra friction only where risk is higher) |
