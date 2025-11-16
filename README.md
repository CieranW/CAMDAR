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
10. [Development Guidelines](#development-guidelines)
11. [Troubleshooting / FAQ](#troubleshooting--faq)

---

## Pipeline Architecture

The system operates **continuously and concurrently per camera**, analyzing all visible individuals with no per-person model duplication.

Our Pipeline Architecture is shown below:

```
Camera Feed
↓
YOLOv11-Pose (keypoints + boxes)
↓
ByteTrack (per-person track IDs)
↓
Pose Buffers (sliding window)
↓
Depth Estimation (3D coordinates)
↓
DeepFace (emotion analysis)
↓
ST-GCN (temporal action + distress classification)
↓
Fusion Engine (final smoothed distress status)
↓
Radar Controller (aims radar at targets)
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
│ └── pipeline.yaml
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
│ ├── camera/
│ ├── detection/
│ ├── tracking/
│ ├── temporal/
│ ├── depth/
│ ├── face/
│ ├── fusion/
│ ├── radar/
│ ├── utils/
│ └── pipeline/
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
