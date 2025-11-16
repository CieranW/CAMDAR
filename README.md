# 🏥 CAMDAR

_A real-time, multi-person, multi-sensor computer vision pipeline for detecting medical emergencies._

---

## 📌 Overview

This project implements a **real-time medical distress detection system** that monitors human posture, motion, facial expression, and 3D position to determine whether an individual may be experiencing a medical emergency.

The system fuses:

- **YOLOv11-Pose** – real-time person + keypoint detection
- **ByteTrack** – multi-object tracking
- **ST-GCN** – temporal pose analysis
- **DeepFace** – facial expression recognition
- **Monocular depth estimation** – 3D coordinate reconstruction
- **Radar alignment** – directing mmWave sensors at distressed individuals

Built to run on the **NVIDIA Jetson AGX Orin**, but compatible with any CUDA-enabled system.

---

## Table of Contents

1. [Pipeline Architecture](#pipeline-architecture)
2. [Repository Structure](#repository-structure)
3. [Rationale Behind Structure](#rationale-behind-structure)
4. [Installation & Setup](#installation--setup)
5. [Datasets](#datasets)
6. [Training Models](#training-models)
7. [Running Real-Time Inference](#running-real-time-inference)
8. [Docker Deployment](#docker-deployment)
9. [Configuration Files](#configuration-files)
10. [Alerts & GUI Dashboard](#alerts--gui-dashboard)
11. [Development Guidelines](#development-guidelines)
12. [Troubleshooting / FAQ](#troubleshooting--faq)

---

## Pipeline Architecture

The system operates **continuously and concurrently per camera**, analyzing all visible individuals with no per-person model duplication.

Our Pipeline Architecture is shown below:

```
Camera Feed
↓
YOLOv11-Pose (Body Keypoints + Boundary Boxes)
↓
ByteTrack (Per-Person Track IDs)
↓
Pose Buffers (Sliding Window)
↓
Depth Estimation (3D Coordinates)
↓
DeepFace (Facial Expression Analysis)
↓
ST-GCN (Temporal Action + Distress Classification)
↓
Fusion Engine (Final Smoothed Distress Status)
↓
Radar Controller (Aims Radar at Targets)
↓
Alert (Notification Upon Abnormal Detection)
↓
GUI (Constant Visual Output)
```

Each component runs in real time to produce continuous status updates for every tracked individual.

---

## Repository Structure

```
CAMDAR/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── config/
│ ├── cameras.yaml
│ ├── models.yaml
│ ├── pipeline.yaml
│ └── alerts.yaml
│
├── data/
│ ├── raw/
│ ├── processed/
│ ├── annotations/
│ ├── examples/
│ └── README.md
│
├── training/
│ ├── train_yolo_pose.py
│ ├── train_stgcn.py
│ ├── dataset_loaders/
│ └── utils/
│
├── models/
│ ├── yolo/
│ ├── stgcn/
│ ├── depth/
│ └── face/
│
├── src/
│ ├── alerts/
│ ├── camera/
│ ├── depth/
│ ├── detection/
│ ├── face/
│ ├── fusion/
│ ├── pipeline/
│ ├── radar/
│ ├── temporal/
│ ├── tracking/
│ ├── ui/
│ └── utils/
│
├── runtime/
│ ├── logs/
│ └── recordings/
│
├── tests/
└── notebooks/
```

---

## Rationale Behind Structure

### Clear Separation of Concerns

- `src/` → **Runtime system** for real-time detection and analysis
- `training/` → **Model training**, preprocessing, dataset handling
- `models/` → **Model wrappers** for inference
- `config/` → Centralized configuration (cameras, weights, thresholds)
- `runtime/` → Logs and recordings generated during execution

This isolation makes debugging, testing, and scaling much easier.

### Scalable Multi-Camera Support

- One Docker container can run **multiple camera pipelines**
- Each pipeline shares model instances → lower GPU memory footprint

### Dataset Hygiene

- `data/raw/` and `data/processed/` are **ignored by Git & Docker**
- Only small metadata (`annotations/`, `examples/`) are versioned
- Ensures repo stays lightweight and reproducible

### Deployment-First Design

- Dockerfile builds a stable runtime environment
- `docker-compose.yml` mounts volumes, passes env vars
- Works seamlessly on Jetson AGX Orin

---

## Installation & Setup

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

```bash
cp .env .env
```

### Edit Configuration Files

Modify YAML files under config/ for:

- Camera sources
- Model weight paths
- Pipeline hyperparameters

---

## Datasets

Large datasets are not stored in Git.

### Dataset Workflow

- Place raw video/image datasets in:

  ```
  data/raw/
  ```

- Place COCO-style labels or metadata in:

  ```
  data/annotations/
  ```

- Preprocessing outputs (skeleton sequences, splits) saved in:
  ```
  data/processed/
  ```

### Automated Dataset Download:

```bash
python tools/download_datasets.py
```

### See Details

data/README.md

---

## Training Models

### Train YOLOv11-Pose

```bash
python training/train_yolo_pose.py
```

### Train ST-GCN

```bash
python training/train_stgcn.py
```

### Preprocess Raw Datasets

```bash
python training/utils/preprocessing.py
```

Training outputs automatically saved under:

```
models/*/weights/
```

---

## Running Real-Time Inference

To start the full real-time pipeline:

```bash
python src/main.py
```

This will:

- Initialize cameras
- Run detection + tracking
- Update pose buffers
- Estimate depth
- Run ST-GCN + facial expression model
- Fuse predictions
- Output statuses + radar tracking commands

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t medical-inference .
```

### Run Inference Container

```bash
docker compose up -d
```

This mounts:

- Logs → runtime/logs/
- Video outputs → runtime/recordings/
- Optional model weight volumes

Designed for Jetson AGX Orin deployment.

---

## Configuration Files

### cameras.yaml

Defines:

- Camera IDs
- RTSP/USB URLs
- Resolution & FPS

### models.yaml

Paths for:

- YOLO models
- ST-GCN weights
- Depth model
- DeepFace config

### pipeline.yaml

Includes:

- Pose buffer length
- Update frequencies for depth + face
- Distress thresholds
- Radar target rules

---

## Alerts & GUI Dashboard

The CAMDAR system includes a built-in **alerting engine** and a **web-based monitoring dashboard** for viewing real-time system status, track states, view logs, view real-time video feed, and medical distress alerts

This functionality is powered by:

- **AlertManager** -- Central dispatcher for all alerts
- **Alert Channels** -- Where alerts are sent (UI, logs, webhooks, email, etc.)
- **StateStore** -- Shared in-memory state for GUI consumption
- **FastAPI-Based GUI/API Server** -- Real-time view of the system
- **Websocket/REST API Endpoints** -- For dashboards, mobile apps, or external systems

### Alerting System

CAMDAR raises alerts when the Decision Fusion module determines an individual is exhibiting potential medical distress based on:

- ST-GCN Distress Probability
- Facial Expression Changes
- Pose or Movement Anomalies
- 3D Motion Instability
- Duration-Based Thresholds (e.g., "sustained distress for > 3 seconds")
- Unusual Vital Sign Activity

Alerts include:

- Severity (Info/Warning/Critical)
- Camera ID
- Track ID
- Timestamp
- Message
- Optional Metadata:

  - 3D Coordinates
  - Classification Probabilities
  - Recent Pose Evolution

### Alert File Structure

Alerts are implemented under:

```
src/
├── alerts/
│   ├── alert_models.py      # Alert dataclasses
│   ├── alert_manager.py     # Orchestrates alert routing
│   └── channels/
│       ├── log_channel.py
│       ├── ui_channel.py
│       ├── email_channel.py
│       └── webhook_channel.py
```

Alert behavior is configured via:

```
config/alerts.yaml
```

This file defines thresholds, durations, and which alert channels are active.

### GUI Dashboard

CAMDAR includes a lightweight **web dashboard** developed with FastAPI.
It provides real-time insight into:

- Active Tracks & Individuals
- Distress Classifications
- Fused Status (Normal/Elevated Risk/High-Distress/Uncertain)
- Recent Alerts
- Per-Camera Live State
- System Health Metrics

The dashboard runs inside the same container as CAMDAR by default.

### Accessing the GUI

Once CAMDAR is running (Docker or local), the GUI becomes available at:

```
http://<device-ip>:8000
```

On Jetson, this may be:

```
http://localhost:8000
```

Or from another machine on the network:

```
http://jetson-orin.local:8000
```

### GUI Architecture

```
src/ui/
├── server.py        # FastAPI web server
├── routes.py        # API & websocket endpoints
└── state_store.py   # Shared system state for UI
```

**StateStore** keeps:

- Current Status per Track (per Camera)
- Historical Alerts
- Radar Engagement Info
- Optional Live Coordinates

The UIChannel pushes alerts into this store so the GUI updates instantly in real-time.

### API Endpoints

Examples of API routes exposed by the GUI server:

```
GET /api/state
GET /api/alerts
GET /api/cameras
WS  /api/stream/alerts       # (optional) realtime alert stream
```

This makes integration possible with:

- Mobile Apps
- Medical Response Dashboards
- Edge Monitoring Platforms
- IoT Systems
- ROS Nodes

### Docker Support for GUI

The GUI is build directly into the CAMDAR container.

Make sure your docker-compose.yml exposes the UI port:

```yaml
services:
  camdar:
    build: .
    ports:
      - "8000:8000"
```

Logs and recordings remain under:

```
runtime/logs/
runtime/recordings/
```

### Integration into Pipeline

The CameraPipeline calls into AlertManager after DecisionFuser updates a track's state:

```
DecisionFuser → AlertManager → UI/Log/Webhook/Email
```

At runtime, the GUI then displays:

- Per-Track Distress Levels
- Per-Camera View
- Alerts in Chronological Order
- Optional Probability Heatmaps/Motion Metrics

---

## Development Guidelines

- Keep training code out of src/
- Add tests for every major module (tests/)
- Never commit large datasets or weights
- Document dataset preparation in data/README.md
- Keep Docker image lean using .dockerignore
- Use environment variables for device-specific paths

---

## Troubleshooting & FAQ

**Q: Can this track multiple people at once?**

Yes — tracking + pose buffers + ST-GCN handle multiple individuals simultaneously.

**Q: Do I need a separate model per person?**

No — all individuals share the same model instances.

**Q: Does this require Kubernetes?**

No.
Docker alone handles single-device deployments well.

**Q: Why isn’t my camera feed opening?**

Check:

- Camera URL in cameras.yaml
- GPU memory usage
- Docker permissions (/dev/video\* mappings)

**Q: Where do logs go?**

```

runtime/logs/

```

**Q: Where are debug videos saved?**

```

runtime/recordings/

```

---

## Final Notes

This repository provides:

- A modular runtime pipeline
- A complete training workflow
- A scalable deployment solution
- A clean dataset management strategy
- Multi-sensor capabilities (vision + radar)

```

```
