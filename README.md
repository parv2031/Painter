# Bob Ross without ROS — 3-Link Robotic Arm Painter

> **Robotics Society Summer Project**  
> Mentors: Anjaneya Damle & Parv Dixit

A complete image-to-painting pipeline for a three-link planar robotic manipulator.  
Feed in any image; the arm traces its contours on a physical canvas.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Tech Stack](#3-tech-stack)
4. [Week-by-Week Breakdown](#4-week-by-week-breakdown)
   - [Week 1 — Forward Kinematics](#week-1--forward-kinematics)
   - [Week 2 — Inverse Kinematics](#week-2--inverse-kinematics)
   - [Week 3 — Trajectory Planning](#week-3--trajectory-planning)
   - [Week 4 — Image Processing](#week-4--image-processing)
   - [Week 5 — Full Drawing Pipeline](#week-5--full-drawing-pipeline)
   - [Week 6 — Smooth Trajectory Pipeline](#week-6--smooth-trajectory-pipeline)
5. [End-to-End Pipeline Summary](#5-end-to-end-pipeline-summary)
6. [Hardware Implementation](#6-hardware-implementation)
   - [Bill of Materials](#61-bill-of-materials)
   - [System Architecture](#62-system-architecture)
   - [Calibration Procedure](#63-calibration-procedure)
   - [Serial Communication Protocol](#64-serial-communication-protocol)
   - [ESP32 Firmware Sketch](#65-esp32-firmware-sketch)
   - [Physical Workspace Setup](#66-physical-workspace-setup)
   - [Future Extensions](#67-future-extensions)
7. [Quick Start](#7-quick-start)
8. [References](#8-references)

---

## 1. Project Overview

This project builds a complete **image → robot motion** pipeline from first principles, covering:

- Custom **forward and inverse kinematics** for a 3-DoF planar arm
- **Cartesian-space trajectory planning** with joint-velocity continuity
- **Computer vision** edge extraction and adaptive contour sampling
- **Simulation** of the full drawing process with real-time velocity monitoring
- **Smooth trajectory generation** using B-spline / Chaikin path smoothing with globally bounded joint velocities

The same mathematical foundation underpins industrial robots, surgical manipulators, and autonomous inspection systems. By building every layer from scratch participants gain a true end-to-end understanding of intelligent robotic systems.

---

## 2. Repository Structure

```
Painter/
├── P3.pdf                         ← Original project specification
├── README.md                      ← This file
│
├── week1_forward_kinematics/
│   ├── forward_kinematics.py      ← ThreeLinkArm class, FK solver
│   ├── visualizer.py              ← Matplotlib arm visualizer
│   ├── demo.py                    ← Interactive demo
│   └── test_forward_kinematics.py ← Unit tests
│
├── week2_inverse_kinematics/
│   ├── inverse_kinematics.py      ← Analytical IK, phi-sweep, auto-select
│   ├── visualizer_ik.py           ← Click-to-reach demo
│   ├── demo_ik.py
│   └── test_inverse_kinematics.py
│
├── week3_trajectory/
│   ├── trajectory.py              ← plan_linear_trajectory, phi-continuity
│   ├── simulator.py               ← Trajectory visualizer
│   ├── demo_trajectory.py
│   └── test_trajectory.py
│
├── week4_image_processing/
│   ├── image_processor.py         ← CLAHE, Canny, contour → stroke pipeline
│   ├── visualizer.py              ← Edge / waypoint preview
│   ├── demo_image.py
│   └── README.md
│
├── week5_full_pipeline/
│   ├── pipeline.py                ← Full pipeline (interactive tuner + simulation)
│   └── README.md
│
└── week6_smooth_pipeline/
    ├── pipeline.py                ← Smooth-trajectory pipeline
    └── README.md
```

---

## 3. Tech Stack

### Software

| Package | Role |
|---|---|
| `numpy` | All array maths, kinematics, trajectory computation |
| `opencv-python-headless` | CLAHE, bilateral filter, Canny edges, contour extraction |
| `matplotlib` (TkAgg) | Interactive tuner, real-time simulation, velocity plots |
| `scipy` *(optional)* | B-spline fitting (`splprep/splev`); Chaikin fallback if ABI mismatch |
| `tkinter` | File-picker dialog |

### Hardware (for physical deployment)

| Component | Purpose |
|---|---|
| 3-link planar robotic arm | The drawing manipulator |
| **ESP32** microcontroller | Receives joint angles over serial, drives servos via PWM |
| High-torque servo motors (×3) | Each arm joint (shoulder, elbow, wrist) |
| Pen / brush end-effector | Mounted at the wrist; Z-axis servo for pen lift |
| Host computer (Linux) | Runs Python pipeline; sends angle stream to ESP32 |

---

## 4. Week-by-Week Breakdown

### Week 1 — Forward Kinematics

**Goal:** Given joint angles `(θ₁, θ₂, θ₃)`, find the end-effector (EE) position.

**Key class:** `ThreeLinkArm(link_lengths=(L1, L2, L3))`

```
ThreeLinkArm
  .forward_kinematics(angles) → {
      joint_positions: (4, 2)  ← [base, J1, J2, EE]
      ee_position: (2,)
      ee_orientation: float    ← total angle from X-axis
  }
  .max_reach          ← L1 + L2 + L3
  .min_reach          ← |L1 - L2 - L3|  (approximate)
```

**Key concept — DH-style chaining:**  
Each link `i` contributes rotation `θᵢ`. Joint positions are accumulated:
```
x₀ = 0, y₀ = 0  (base)
xᵢ = xᵢ₋₁ + Lᵢ·cos(θ₁+…+θᵢ)
yᵢ = yᵢ₋₁ + Lᵢ·sin(θ₁+…+θᵢ)
```

---

### Week 2 — Inverse Kinematics

**Goal:** Given a target EE position `(x, y)`, find joint angles that reach it.

**Key function:** `ik_analytical_auto(arm, x, y)`

The analytical approach uses a **phi-sweep** over the arm's orientation angle φ (total angle of all joints from base):

1. For each candidate φ, `(x, y)` is reached by the 3-DoF arm in a unique configuration.
2. IK is solved geometrically using the law of cosines for the reduced 2-link sub-problem.
3. The solution with smallest joint-limit violations and best workspace positioning is selected.

**Configuration continuity:**  
When solving IK for consecutive waypoints, the solution closest in joint space to the previous configuration is always chosen. This prevents elbow flips.

---

### Week 3 — Trajectory Planning

**Goal:** Plan a smooth Cartesian straight-line path from `p1 → p2`.

**Key function:** `plan_linear_trajectory(arm, p1, p2, ee_speed, min_steps, max_steps, init_angles)`

**Returns:** `TrajectoryResult`
```
waypoints        : (N, 2)   ← EE positions along the line
joint_angles     : (N, 3)   ← IK solution at each waypoint
joint_velocities : (N-1, 3) ← Δθ/Δt (rad/s) between consecutive waypoints
dt               : float    ← time step
success_mask     : (N,) bool
```

**Servo velocity:**
```
ωᵢ[k] = shortest_angle_diff(θᵢ[k], θᵢ[k-1]) / Δt
```
`shortest_angle_diff` wraps to `(−π, π]` so a servo moving 170° → −170° sees `Δθ = +20°`, not `−340°`.

---

### Week 4 — Image Processing

**Goal:** Convert a photograph into a set of drawable strokes (ordered waypoint lists).

**Pipeline:**

```
Raw image
   ↓  Gamma correction          (brightness)
   ↓  CLAHE                     (local contrast)
   ↓  Bilateral filter          (edge-preserving denoise)
   ↓  Canny edge detector       (edge map)
   ↓  Morphological close       (gap filling)
   ↓  findContours              (contour extraction)
   ↓  Length filter             (drop short noise contours)
   ↓  space_points_wisely()     (adaptive arc-length resampling)
   ↓
Stroke list: [ (N₁, 2), (N₂, 2), … ]  ← pixel coordinates
```

**Adaptive spacing (`space_points_wisely`):**  
Each contour is resampled so consecutive waypoints are exactly `step` pixels apart regardless of the original OpenCV contour density. This gives uniform waypoint density across all strokes.

**Outputs saved:**
- `*_edges.png` — Canny edge image
- `*_strokes.json` — strokes as JSON list of point arrays
- `*_strokes.npz` — compact numpy archive

---

### Week 5 — Full Drawing Pipeline

**Entry point:** `week5_full_pipeline/pipeline.py`

**Key additions over Week 4:**

| Feature | Description |
|---|---|
| `strokes_to_arm()` | Maps pixel coords → arm workspace (upper-right quadrant, uniform scale, aspect-ratio preserved) |
| `order_strokes()` | Greedy nearest-neighbour ordering to minimize total pen-lift travel |
| `_seg_frames()` | Wraps `plan_linear_trajectory` into frame dicts (angles, vel_deg, ee, is_drawing) |
| `_seg_frames_safe()` | Detects velocity spikes (elbow flips); inserts pen-up detour via-point if needed |
| `compute_frames()` | Assembles all drawing + lift frames; adds return-to-home at the end |
| `interactive_preview()` | 7-slider tuning window (CLAHE, Bilateral σ, Canny σ, Gamma, Morph, Min stroke %, Spacing) |
| `run_animation()` | White canvas + black strokes; real-time joint velocity plots; speed slider (0.1×–4×) |

**Workspace geometry:**
```
Arm base at (0, 0) — bottom-left corner of canvas
Drawing box: [MARGIN, MARGIN+BOX]² = [0.4, 4.6]²  arm-units
Max reach check: √2 × 4.6 ≈ 6.51 < 7.0 (= L1+L2+L3) ✓
```

**Elbow-flip detour strategy:**  
If `plan_linear_trajectory` on segment `p1→p2` produces `|ω| > 600 deg/s`, the planner tries via-points at perpendicular offsets from the segment midpoint. The first via-point that reduces the peak by ≥ 35% is accepted as a pen-up detour.

---

### Week 6 — Smooth Trajectory Pipeline

**Entry point:** `week6_smooth_pipeline/pipeline.py`

**Core upgrade:** replaces discrete waypoint-to-waypoint motion with **continuous smooth curve tracing** at a globally velocity-bounded slow speed.

#### What changed and why

| Root cause of jitter (W5) | Week 6 fix |
|---|---|
| Straight segments between sparse waypoints → sharp direction changes | B-spline / Chaikin smoothing + arc-length resampling |
| `vel=0` at `k=0` of every `plan_linear_trajectory` call → zero-crossing artifacts | Direct IK per dense sample — no `plan_linear_trajectory` for drawing |
| Independent phi reference per segment → IK branch drift at junctions | `_despike_and_smooth` detects branch switches and interpolates |
| Per-segment processing leaves inter-stroke seams unsmoothed | Global smooth pass over entire `all_frames` at the end |

#### Velocity guarantee — three layers

```
Per-stroke:       despike(σ=8, thresh=10°) → clamp(200 deg/s)
Per-transition:   despike(σ=8, thresh=10°) → clamp(200 deg/s)
Global final:     despike(σ=6, thresh=8°)  → clamp(100 deg/s)
─────────────────────────────────────────────────────────────
Result: max |ω| ≤ 100 deg/s  for ALL joints  at ALL frames
```

#### Speed constants

| Constant | Value |
|---|---|
| `ARC_STEP` | 0.015 arm-units per IK sample |
| Effective EE speed | 0.015 / 0.02 s = **0.75 arm-units/s** |
| `DRAW_SPEED` | 0.8 arm-units/s (legacy, used by lift fallback) |
| `BASE_INTERVAL` | 20 ms/frame |

#### 8-slider interactive tuner

| Slider | Range | Effect |
|---|---|---|
| CLAHE clip | 0.5–5.0 | Local contrast |
| Bilateral σ | 10–80 | Edge-preserving blur strength |
| Canny σ | 0.1–0.7 | Edge sensitivity |
| Gamma | 0.3–2.5 | Brightness correction |
| Morph close | 0–4 | Gap-closing iterations |
| Min stroke % | 0.2–3.0 | Drop short noise contours |
| Waypoint spacing | 4–50 px | Input contour density |
| **Smoothing** | **0.0–5.0** | **B-spline smoothing factor** |

---

## 5. End-to-End Pipeline Summary

```
         ┌─────────────────────────────────────────────────────────────┐
         │  INPUT: any photograph / line art                           │
         └───────────────────────────┬─────────────────────────────────┘
                                     │
         ┌───────────────────────────▼─────────────────────────────────┐
         │  Week 4: Image Processing                                   │
         │  Gamma → CLAHE → Bilateral → Canny → Morph → Contours      │
         │  → space_points_wisely() → stroke list (pixel coords)       │
         └───────────────────────────┬─────────────────────────────────┘
                                     │
         ┌───────────────────────────▼─────────────────────────────────┐
         │  Coordinate Transform                                       │
         │  pixel (x,y) → arm (x,y)  [upper-right quadrant, uniform   │
         │  scale, aspect-ratio preserved, y-axis flipped]             │
         └───────────────────────────┬─────────────────────────────────┘
                                     │
         ┌───────────────────────────▼─────────────────────────────────┐
         │  Greedy stroke ordering (nearest-neighbour)                 │
         │  → minimise total pen-lift distance                         │
         └───────────────────────────┬─────────────────────────────────┘
                                     │
         ┌───────────────────────────▼─────────────────────────────────┐
         │  Week 6: Path Smoothing                                     │
         │  Chaikin (8× oversample) → arc-length resample (0.015 AU)  │
         └───────────────────────────┬─────────────────────────────────┘
                                     │
         ┌───────────────────────────▼─────────────────────────────────┐
         │  Week 2: Inverse Kinematics  (per dense sample)             │
         │  ik_analytical_auto → phi-continuous joint angles           │
         └───────────────────────────┬─────────────────────────────────┘
                                     │
         ┌───────────────────────────▼─────────────────────────────────┐
         │  Post-processing                                            │
         │  _despike_and_smooth → _clamp_velocities (100 deg/s)       │
         └───────────────────────────┬─────────────────────────────────┘
                                     │
         ┌───────────────────────────▼─────────────────────────────────┐
         │  Simulation  (matplotlib TkAgg)                             │
         │  White canvas · black strokes · joint velocity plots        │
         │  Speed slider · return-to-home at end                       │
         └───────────────────────────┬─────────────────────────────────┘
                                     │
         ┌───────────────────────────▼─────────────────────────────────┐
         │  PHYSICAL ARM  (future / hardware week)                     │
         │  Joint angles → serial → ESP32 → servo PWM → canvas        │
         └─────────────────────────────────────────────────────────────┘
```

---

## 6. Hardware Implementation

### 6.1 Bill of Materials

| Component | Specification | Qty |
|---|---|---|
| Arm links | Aluminium extrusion or 3D-printed PLA | 3 |
| Servo motors | MG996R or DS3218 (high-torque, 180°) | 3 + 1 (pen lift) |
| Microcontroller | ESP32 DevKit v1 | 1 |
| Power supply | 5 V / 5 A regulated (for servos) | 1 |
| Logic level converter | 3.3 V ↔ 5 V (ESP32 to servo signal) | 1 |
| Pen / brush holder | 3D-printed clamp at wrist | 1 |
| Canvas board | A4 or A3 white cartridge paper | — |
| USB-UART cable | Host ↔ ESP32 serial | 1 |

---

### 6.2 System Architecture

```
┌──────────────────────┐        USB / UART        ┌──────────────────┐
│   Host Computer      │ ───────────────────────► │  ESP32           │
│   (Python pipeline)  │   "θ1,θ2,θ3,pen\n"       │  Firmware        │
│                      │ ◄─────────────────────── │                  │
│  image processing    │   "ACK\n"                 │  Servo PWM out   │
│  IK + smoothing      │                           │  (4 channels)    │
│  frame generation    │                           └──────┬───────────┘
└──────────────────────┘                                  │
                                                ┌─────────┴──────────┐
                                                │  Servos            │
                                                │  S1 (shoulder)     │
                                                │  S2 (elbow)        │
                                                │  S3 (wrist)        │
                                                │  S4 (pen lift)     │
                                                └────────────────────┘
```

---

### 6.3 Calibration Procedure

Before running any drawing sequence:

1. **Zero-position calibration**  
   Move all joints to their mechanical zero (arm pointing along +X axis).  
   Record the PWM pulse width at zero for each servo.

2. **Scale factor measurement**  
   Command joint 1 to +90° and measure the actual arm angle with a protractor.  
   Compute `deg_per_us = 90 / (pw_90 - pw_0)` for each servo.

3. **Base-to-paper offset**  
   Measure the physical distance (in cm) from the arm base pivot to the paper corner.  
   Update `x_off` and `y_off` in `strokes_to_arm()` to account for this offset.

4. **Reachability check**  
   Command the EE to `(_MARGIN, _MARGIN)` and `(_MARGIN+_BOX, _MARGIN+_BOX)` in simulation.  
   Confirm the physical arm reaches these positions on the paper without collision.

5. **Pen-down height**  
   Adjust the pen-lift servo's `PEN_DOWN_US` / `PEN_UP_US` constants until the pen just touches the paper at down and clears it by ~5 mm at up.

---

### 6.4 Serial Communication Protocol

The host Python script streams one command per animation frame over serial:

```
Format:   "θ1,θ2,θ3,pen\n"
Example:  "45.2,-12.7,33.0,1\n"

Fields:
  θ1, θ2, θ3  — joint angles in degrees (float, 1 decimal place)
  pen          — 1 = pen down (drawing), 0 = pen up (lift)
```

ESP32 replies with `"ACK\n"` after each command is executed.  
The host waits for `ACK` before sending the next frame, providing natural back-pressure.

**Python sender snippet:**
```python
import serial, time

ser = serial.Serial("/dev/ttyUSB0", baudrate=115200, timeout=2.0)
time.sleep(2)   # let ESP32 reset

for f in all_frames:
    θ = f["angles"]
    pen = 1 if f["is_drawing"] else 0
    cmd = f"{np.degrees(θ[0]):.1f},{np.degrees(θ[1]):.1f},{np.degrees(θ[2]):.1f},{pen}\n"
    ser.write(cmd.encode())
    ser.readline()   # wait for ACK
    time.sleep(BASE_INTERVAL / 1000.0)

ser.close()
```

---

### 6.5 ESP32 Firmware Sketch

```cpp
#include <ESP32Servo.h>

Servo s1, s2, s3, sPen;

const int PIN_S1 = 13, PIN_S2 = 12, PIN_S3 = 14, PIN_PEN = 27;
const int PEN_DOWN_US = 1200, PEN_UP_US = 1600;

// Calibration: pulse width (µs) at 0° and degrees per µs
const float PW0[]       = {1500, 1500, 1500};
const float DEG_PER_US  = 0.18f;   // adjust per servo model

int degToUs(float deg, int ch) {
    return (int)(PW0[ch] + deg / DEG_PER_US);
}

void setup() {
    Serial.begin(115200);
    s1.attach(PIN_S1);  s2.attach(PIN_S2);
    s3.attach(PIN_S3);  sPen.attach(PIN_PEN);
    sPen.writeMicroseconds(PEN_UP_US);
}

void loop() {
    if (!Serial.available()) return;
    String line = Serial.readStringUntil('\n');
    // Parse "θ1,θ2,θ3,pen"
    float t1, t2, t3; int pen;
    sscanf(line.c_str(), "%f,%f,%f,%d", &t1, &t2, &t3, &pen);

    s1.writeMicroseconds(degToUs(t1, 0));
    s2.writeMicroseconds(degToUs(t2, 1));
    s3.writeMicroseconds(degToUs(t3, 2));
    sPen.writeMicroseconds(pen ? PEN_DOWN_US : PEN_UP_US);

    Serial.println("ACK");
}
```

---

### 6.6 Physical Workspace Setup

```
          ┌──────────────────────────────┐
          │         PAPER / CANVAS       │
          │                              │
          │   Drawing box                │
          │   [MARGIN, MARGIN+BOX]²      │
          │   (0.4 → 4.6 arm-units)      │
          │                              │
          └──────────────────────────────┘
   ◉  ← arm base pivot point (0, 0)
   (bottom-left corner of paper boundary)
```

**Physical scale:**  
If `L1=30 cm, L2=25 cm, L3=15 cm`, then 1 arm-unit = 10 cm.  
The drawing box spans `0.4×10 = 4 cm` to `4.6×10 = 46 cm` — fits an A3 sheet.

---

### 6.7 Future Extensions

| Extension | Description |
|---|---|
| **Vision feedback** | Mount a camera above the canvas. Compare the drawn result to the target image frame-by-frame and re-plan missed strokes. |
| **Jacobian-based path correction** | Replace analytical IK with a Jacobian pseudo-inverse controller for smoother real-time path following and obstacle avoidance. |
| **Multi-colour drawing** | Add a tool-changer servo that selects between multiple pen colours. Insert colour-switch commands between strokes of different layers. |
| **Energy-optimal trajectories** | Use time-optimal trajectory planning (e.g., TOPP-RA) to minimise total motion time subject to velocity and acceleration constraints. |
| **Closed-loop force feedback** | Add strain gauges to measure pen contact force. Modulate pen-down depth to maintain consistent stroke pressure. |
| **WiFi streaming** | Replace USB serial with ESP32 WiFi (WebSocket) for a fully wireless setup. |

---

## 7. Quick Start

### Simulation only (no hardware required)

```bash
# Clone or navigate to the project
cd ~/Painter

# Week 6 smooth pipeline (recommended)
cd week6_smooth_pipeline
python3 pipeline.py
# → file picker → select image → adjust sliders → click ▶ Start Simulation

# Week 5 pipeline (waypoint-based, faster pre-computation)
cd ../week5_full_pipeline
python3 pipeline.py
```

### Individual module demos

```bash
# Forward kinematics interactive demo
cd week1_forward_kinematics && python3 demo.py

# Click-to-reach IK demo
cd week2_inverse_kinematics && python3 demo_ik.py

# Straight-line trajectory demo
cd week3_trajectory && python3 demo_trajectory.py

# Image processing / edge extraction
cd week4_image_processing && python3 demo_image.py
```

### Running unit tests

```bash
python3 week1_forward_kinematics/test_forward_kinematics.py
python3 week2_inverse_kinematics/test_inverse_kinematics.py
python3 week3_trajectory/test_trajectory.py
```

---

## 8. References

1. OpenCV Documentation — https://opencv.org
2. NumPy Documentation — https://numpy.org
3. Peter Corke, *Robotics, Vision and Control*, Springer, 2017
4. Siciliano et al., *Robotics: Modelling, Planning and Control*, Springer, 2009
5. Chaikin, G.M., *An Algorithm for High Speed Curve Generation*, Computer Graphics and Image Processing, 1974

---

*Project Coordinators: Anjaneya Damle (9867297091) · Parv Dixit (6398612592)*
