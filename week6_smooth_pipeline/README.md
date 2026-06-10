# Week 6 — Smooth Trajectory Drawing Pipeline

## Overview

Week 6 upgrades the Week 5 full pipeline with a fundamentally different drawing strategy: instead of planning straight-line segments between sparse waypoints, the arm now **traces a continuously smooth curve** at a very slow, bounded speed. The result is jitter-free, graceful arm motion with no sudden velocity spikes anywhere in the animation.

---

## Key Idea: Why Week 5 Had Jitter

In Week 5, `plan_linear_trajectory` was called independently for each consecutive pair of waypoints. At every waypoint the arm had to:

1. Stop, change direction, and start again — causing sharp velocity spikes.
2. Re-solve IK from a fresh phi reference — occasionally triggering elbow-flip discontinuities.
3. Accept vel = 0 at the first frame of every new segment — creating zero-crossing artifacts.

Week 6 eliminates all three root causes.

---

## Pipeline Architecture

```
[Image] → Interactive Tuner (8 sliders)
              ↓
         strokes_to_arm()       — pixel coords → arm workspace
              ↓
         order_strokes()        — greedy nearest-neighbour ordering
              ↓
         smooth_all_strokes()   — Chaikin / B-spline smoothing (8× oversample)
              ↓
    ┌── for each stroke ──────────────────────────────────────┐
    │  _transition_frames_smooth()  — slow pen-up travel      │
    │  _stroke_frames_smooth()      — slow curve trace        │
    └─────────────────────────────────────────────────────────┘
              ↓
         Global smooth pass     — fixes inter-stroke seams
              ↓
         run_animation()        — white canvas, speed slider
```

---

## New Functions

### `smooth_stroke(stroke_arm, smooth_factor, oversample=8)`
Fits a **Chaikin corner-cutting curve** (or scipy B-spline if available) through the raw contour waypoints and resamples at `oversample×` density. This converts a sparse, jagged polyline into a smooth, dense curve before any IK is computed.

- **Chaikin**: each iteration cuts corners by 75%/25%, producing a smooth approximating curve. 4 iterations ≈ 16× subdivision.
- **Endpoint preservation**: the first and last waypoints are kept exactly.
- **Fallback**: if scipy is unavailable (numpy ABI mismatch), Chaikin is used automatically.

### `_arc_resample(pts, ds)`
Resamples a polyline at **uniform arc-length intervals** of `ds` arm-units. This ensures the end-effector moves at a constant speed (no acceleration/deceleration artifacts from uneven point spacing).

### `_stroke_frames_smooth(stroke_arm, prev_angles, stroke_idx)`
The core drawing function. For each stroke:

1. **Smooth** via `smooth_stroke()` (Chaikin, 8× oversample).
2. **Arc-resample** at `ARC_STEP = 0.015` arm-units.
3. **Direct IK** — calls `ik_analytical_auto` at every dense sample point. No `plan_linear_trajectory` calls, so there are no vel=0 seam artifacts.
4. **Despike** — `_despike_and_smooth(spike_thresh=10°, sigma=8)`: detects sudden angle jumps, linearly interpolates over them, then applies a Gaussian convolution.
5. **Clamp** — `_clamp_velocities(max_vel=200 deg/s)`: hard-caps all joint velocities.

### `_transition_frames_smooth(start_ee, end_ee, prev_angles, stroke_idx)`
Handles **pen-up travel** between strokes (and the return-to-home move). Uses the exact same direct-IK + despike + clamp pipeline as drawing, so transitions are also slow and smooth. The EE moves at `ARC_STEP / dt = 0.015 / 0.02 ≈ 0.75 arm-units/s`.

### `_despike_and_smooth(frames, spike_thresh_deg, sigma)`
Post-processes a frame sequence in three steps:

1. Detect angle jumps > `spike_thresh_deg` per frame.
2. Linearly interpolate over the bad region (±`sigma` frames).
3. Apply a **pure-numpy Gaussian convolution** (no scipy) across the whole sequence.

### `_clamp_velocities(frames, max_vel_deg)`
**Hard velocity cap**: for each frame, if the fastest joint exceeds `max_vel_deg`, all three joints are scaled proportionally so the fastest is exactly at the limit. This preserves coordinated motion while guaranteeing the bound.

### Global smooth pass (inside `compute_frames`)
After all strokes and transitions are assembled, a **final pass** runs `_despike_and_smooth(sigma=6, thresh=8°)` + `_clamp_velocities(100 deg/s)` over the entire `all_frames` sequence. This smooths the seams between independently-processed segments that the per-stroke passes cannot see.

---

## Speed Constants

| Constant | Value | Meaning |
|---|---|---|
| `DRAW_SPEED` | 0.8 arm-units/s | Target EE speed (used by `plan_linear_trajectory` in lifts) |
| `ARC_STEP` | 0.015 arm-units | EE distance between consecutive IK samples |
| `LIFT_SPEED` | 8.0 arm-units/s | Legacy lift speed (overridden by smooth transitions) |
| `BASE_INTERVAL` | 20 ms/frame | Animation frame rate |
| Effective draw speed | ~0.75 arm-units/s | `ARC_STEP / BASE_INTERVAL` = 0.015 / 0.02 |

---

## Velocity Guarantee

Three layers ensure no spike survives:

```
Layer 1 (per-stroke):      despike(σ=8, thresh=10°) → clamp(200 deg/s)
Layer 2 (per-transition):  despike(σ=8, thresh=10°) → clamp(200 deg/s)
Layer 3 (global final):    despike(σ=6, thresh=8°)  → clamp(100 deg/s)
```

**Guaranteed result**: `max |ω| ≤ 100 deg/s` for every joint at every frame.

| Metric | Week 5 | Week 6 |
|---|---|---|
| Max velocity | ~1500 deg/s (spikes) | **100 deg/s (hard bounded)** |
| Mean velocity | ~68 deg/s | **~22 deg/s** |
| Direction changes | Sharp, at every waypoint | Smooth, Gaussian-spread |
| Elbow-flip detours | Required (`_seg_frames_safe`) | Not needed |
| Transition speed | Fast (`LIFT_SPEED = 14`) | Same slow bounded motion |

---

## Interactive Tuner — 8 Sliders

| Slider | Range | Effect |
|---|---|---|
| CLAHE clip | 0.5 – 5.0 | Local contrast enhancement |
| Bilateral σ | 10 – 80 | Edge-preserving blur |
| Canny σ | 0.1 – 0.7 | Edge sensitivity |
| Gamma | 0.3 – 2.5 | Brightness correction |
| Morph close | 0 – 4 | Gap-closing iterations |
| Min stroke % | 0.2 – 3.0 | Drop short contours |
| Waypoint spacing | 4 – 50 px | Input contour density |
| **Smoothing** | **0.0 – 5.0** | **B-spline smoothing factor (Week 6)** |

The waypoints panel in the tuner shows the **smooth path** (not raw dots) so you see exactly what the arm will draw.

---

## How to Run

```bash
cd ~/Painter/week6_smooth_pipeline
python3 pipeline.py
```

1. A file picker opens — select any image (jpg / png / bmp / webp).
2. Adjust the 8 sliders until the sketch and smooth path look right.
3. Click **▶ Start Simulation**.
4. The pipeline pre-computes IK, applies smoothing, then opens the animation window.
5. Use the **Speed ×** slider (0.1× – 4×) to control playback rate.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | All array maths |
| `opencv-python-headless` | Image loading, Canny edge detection, contour extraction |
| `matplotlib` (TkAgg) | Interactive tuner + animation window |
| `scipy` *(optional)* | B-spline fitting via `splprep/splev`; Chaikin fallback if unavailable |

Modules from this project (auto-imported via `sys.path`):

- `week1_forward_kinematics/forward_kinematics.py` — `ThreeLinkArm`, FK
- `week2_inverse_kinematics/inverse_kinematics.py` — `ik_analytical_auto`
- `week3_trajectory/trajectory.py` — `plan_linear_trajectory`
- `week4_image_processing/image_processor.py` — `space_points_wisely`, `save_strokes`
