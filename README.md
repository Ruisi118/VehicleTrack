# VehicleTrack

Video-to-trajectory traffic analysis for urban policy.

> CUSP-GX 8873A Urban Computing and Artificial Intelligence · Spring 2026 · Final Project

VehicleTrack is an offline pipeline that converts fixed-camera roadway video into structured traffic data: trajectories, speeds (with confidence labels), counts, and policy indicators. It pairs a project-generated 60-second pilot clip with FHWA TGSIM Foggy Bottom as a public trajectory-analysis anchor.

---

## 1. Pipeline overview

```
video.mp4
  → Faster R-CNN (vehicle detection)
  → IoU baseline / ByteTrack (tracking)
  → Multi-segment Homography (ground-plane projection)
  → Speed estimation + confidence zoning
  → trajectories.csv, metrics.json, annotated.mp4
  → policy analysis
```

Three calibration variants are provided under `calibration/`:

| File | Purpose |
|---|---|
| `video.calibration.json` | Single-ROI baseline |
| `video.multisegment.calibration.json` | Near / mid / far segments (recommended) |
| `video.far-split.experimental.calibration.json` | Experimental far-field split |

---

## 2. Repository layout

```
project/
├── src/                  # Core library
│   ├── detection.py        Faster R-CNN ResNet-50 FPN vehicle detector
│   ├── tracking.py         IoU baseline + ByteTrack
│   ├── homography.py       Single-ROI and multi-segment projection
│   ├── visualize.py        Annotated-frame drawing
│   └── pipeline.py         End-to-end run entrypoint
├── calibrate.py          Interactive 4-point calibration tool
├── scripts/              # Analysis CLIs (post-pipeline)
│   ├── analyze_policy_indicators.py
│   ├── analyze_tgsim.py
│   ├── check_tgsim_video_alignment.py
│   └── evaluate_calibration_ab.py
├── calibration/          # Homography JSON files
├── data/
│   ├── video.mp4           60s fixed-camera pilot clip (25 fps, 1501 frames)
│   └── tgsim/              FHWA TGSIM Foggy Bottom sample + schema
├── outputs/              # Pipeline run artifacts (see §5)
├── tests/                # Unit tests for homography + tracking
└── requirements.txt
```

---

## 3. Setup

Requires Python 3.10+.

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On first run the Faster R-CNN weights (~160 MB) are downloaded into `project/.cache/torch/`. CPU-only PyTorch works; a CUDA build is faster but optional.

Sanity-check the install:

```bash
pytest tests/
```

---

## 4. Running the pipeline

### 4.1 Full pipeline (detection → tracking → speed)

```bash
python src/pipeline.py \
  --video data/video.mp4 \
  --calib calibration/video.multisegment.calibration.json \
  --out outputs/my_run \
  --tracker bytetrack \
  --max-frames 1501
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--tracker` | `iou` | `iou` (baseline) or `bytetrack` (recommended) |
| `--score-threshold` | `0.7` | Detection confidence cutoff |
| `--max-frames` | `50` | Use `1501` for the full 60s clip |
| `--start-frame` / `--start-sec` | — | Seek into the video |
| `--device` | auto | Force `cpu` if no GPU |

Each run folder contains `annotated.mp4`, `trajectories.csv`, `metrics.json`, `config.json`.

### 4.2 Calibration (only when re-calibrating)

```bash
python calibrate.py data/video.mp4 --time-sec 16.9
```

Click 4 road-plane points, enter real-world width/depth in meters, save under `calibration/`.

### 4.3 Calibration A/B diagnostic

Reprojects an existing `trajectories.csv` through alternative calibration files (does not rerun detection).

```bash
python scripts/evaluate_calibration_ab.py
```

Outputs land in `outputs/calibration_ab/`.

### 4.4 Policy indicators

```bash
python scripts/analyze_policy_indicators.py
```

Defaults read `outputs/run_bytetrack_smooth_1min/trajectories.csv` and write CSV tables to `outputs/policy_analysis/`.

### 4.5 TGSIM analysis (public dataset layer)

```bash
python scripts/analyze_tgsim.py            # samples + profiles the public CSV
python scripts/check_tgsim_video_alignment.py   # feasibility check vs. CV pipeline
```

---

## 5. Output artifacts

| Folder | Description |
|---|---|
| `outputs/run_baseline_1min/` | IoU greedy + single-ROI calibration |
| `outputs/run_multiseg_1min/` | IoU greedy + multi-segment calibration |
| `outputs/run_bytetrack_1min/` | ByteTrack + multi-segment (same-segment speed) |
| `outputs/run_bytetrack_smooth_1min/` | **Best current run** — adds cross-segment smoothing |
| `outputs/calibration_ab/` | Geometry-only A/B between calibration variants |
| `outputs/policy_analysis/` | Speeding, counting-line flow, event-screening tables |
| `outputs/tgsim_analysis/` | TGSIM sample + schema profile |
| `outputs/tgsim_pipeline_validation/` | TGSIM-vs-pipeline feasibility evidence |
| `outputs/speed_sanity_check/` | Lane-marking transit ground-truth check |

Each run folder schema:

| File | Content |
|---|---|
| `trajectories.csv` | Per-frame rows: bbox, track ID, world coords, display speed, `speed_source` |
| `metrics.json` | Run KPIs, speed histogram, segment diagnostics |
| `config.json` | Reproducibility metadata |
| `annotated.mp4` | Bounding boxes + track IDs + speed labels |

The `speed_source` column flags confidence:

| Value | Meaning |
|---|---|
| `same_segment` | Within one calibration segment (most reliable) |
| `cross_segment_smoothed` | Conservative smoothing across boundary |
| `filtered_outlier` | Exceeded 180 km/h plausibility cap |
| `unavailable` | Not enough calibrated history |

---

## 6. Datasets

- **Pilot video** (`data/video.mp4`): 60.04 s, 1501 frames, 25 fps, fixed-camera highway scene. Pilot scale only — do not draw population-level claims from one clip.
- **FHWA TGSIM Foggy Bottom** (`data/tgsim/`, 50k-row sample): physically grounded urban trajectories from 12 stationary 4K cameras. [Source](https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-foggy-bottom-trajectories). Used as the analysis-layer anchor, not as detector/tracker training data.

---

## 7. Reproducing the best run

```bash
python src/pipeline.py \
  --video data/video.mp4 \
  --calib calibration/video.multisegment.calibration.json \
  --out outputs/run_bytetrack_smooth_1min_repro \
  --tracker bytetrack \
  --max-frames 1501
python scripts/evaluate_calibration_ab.py
python scripts/analyze_policy_indicators.py \
  --trajectories outputs/run_bytetrack_smooth_1min_repro/trajectories.csv \
  --out outputs/policy_analysis_repro
```

---

## 8. Scope and limitations

- Offline analysis only — not real-time.
- Speeds outside the calibrated near/mid segments are explicitly uncertain; see the `speed_source` column.
- Single-clip pilot; not an enforcement-grade measurement system.

---

## 9. References

### 9.1 Methods

| # | Reference | Used in |
|---|---|---|
| M1 | Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* [arXiv:1506.01497](https://arxiv.org/abs/1506.01497) | `src/detection.py` — primary detector |
| M2 | He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | ResNet-50 backbone |
| M3 | Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). *Feature Pyramid Networks for Object Detection.* [arXiv:1612.03144](https://arxiv.org/abs/1612.03144) | FPN neck |
| M4 | Lin, T.-Y. et al. (2014). *Microsoft COCO: Common Objects in Context.* [arXiv:1405.0312](https://arxiv.org/abs/1405.0312) | Source of pretrained weights |
| M5 | Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* [arXiv:2110.06864](https://arxiv.org/abs/2110.06864) | `src/tracking.py` — ByteTrack backend |
| M6 | Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). *Simple Online and Realtime Tracking (SORT).* [arXiv:1602.00763](https://arxiv.org/abs/1602.00763) | IoU greedy tracking baseline |

### 9.2 Data

| # | Reference | Notes |
|---|---|---|
| D1 | FHWA TGSIM Foggy Bottom Trajectories (dataset catalog). [catalog.data.gov](https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-foggy-bottom-trajectories) | Public trajectory-analysis anchor; not used to train detector/tracker. Sample in `data/tgsim/`. |
| D2 | FHWA Third Generation Simulation (TGSIM) Program — overview. [highways.dot.gov](https://highways.dot.gov/research/operations/Third-Generation-Simulation) | Acquisition methodology: 12 stationary 4K cameras |
| D3 | `data/video.mp4` — self-recorded fixed-camera pilot clip, 60.04 s, 25 fps, 1501 frames | Project-generated; used as the CV pipeline's primary input. Pilot scale only. |

### 9.3 Software

| # | Reference | Used in |
|---|---|---|
| S1 | PyTorch / torchvision detection models. [pytorch.org/vision](https://pytorch.org/vision/main/models/faster_rcnn.html) | Faster R-CNN implementation + pretrained weights |
| S2 | OpenCV — *Basic concepts of the homography explained with code.* [docs.opencv.org](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html) | `src/homography.py`, calibration |
| S3 | Supervision (Roboflow). [github.com/roboflow/supervision](https://github.com/roboflow/supervision) | Annotation utilities (requirements.txt) |
