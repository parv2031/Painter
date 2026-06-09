# Week 5 — Full Drawing Simulation Pipeline

**Project:** Bob Ross without ROS — Robotics Society Summer Project  
**Mentors:** Anjaneya and Parv  
**Milestone:** Complete integration — image → IK → animation + servo velocities

---

## Run

```bash
cd week5_full_pipeline
python3 pipeline.py
```

A file dialog opens. Select any image. The rest is automatic.

---

## What Happens (Step by Step)

```
Image file
   │
   ▼
[Week 4] process_image()
   → Edges (CLAHE + Bilateral + Sigma-Canny + Morphological close)
   → Strokes (ordered pixel sequences, ~30 px spacing)
   │
   ▼
strokes_to_arm_coords()
   → Scales image bounding box → arm workspace circle (radius 5.0 units)
   → Flips Y axis (image y-down → arm y-up)
   │
   ▼
order_strokes()  — greedy nearest-neighbour
   → Reorders strokes so that pen-lift moves are minimised
   → Also tries reversing each candidate stroke (picks shorter option)
   │
   ▼
compute_frames()  — [Week 3] plan_linear_trajectory()
   For each stroke:
     For each consecutive waypoint pair (p_i, p_{i+1}):
       → plan_linear_trajectory(ARM, p_i, p_{i+1}, ee_speed=6.0)
       → Stores joint_angles[k] + joint_velocities[k] for every step
   Between each stroke:
     → plan_linear_trajectory(ARM, stroke_end, next_stroke_start, ee_speed=14.0)
     → Pen-lift (no drawing, higher speed)
   │
   ▼
Preview window
   → Panel 1: Original image
   → Panel 2: Sketch (Canny edges, black on white)
   → Panel 3: Waypoints (coloured strokes)
   Close preview to start simulation.
   │
   ▼
Simulation window
   Left:  Arm + canvas (links animated, drawn lines accumulate)
   Right: 3 live-scrolling servo velocity plots (ω1, ω2, ω3)
   Bottom: Status bar — stroke number, phase, EE position, velocities
```

---

## Architecture Details

### Coordinate Transform

```
arm_x =  (pixel_x - img_width/2)  * scale
arm_y = -(pixel_y - img_height/2) * scale    ← Y flipped

where scale = (2 * CANVAS_R) / max(image_height, image_width)
      CANVAS_R = 5.0  arm-units
```

This maps the entire image into a 10×10 arm-unit square, fitting within the arm's 7-unit max reach.

### Greedy Stroke Ordering

After extraction, strokes are unordered. Drawing them in the original order would require long pen-lifts. The greedy algorithm:
1. Start at origin (0, 0)
2. Find the undrawn stroke with the nearest endpoint (start **or** end) to the current pen position
3. If the end-point was nearer, reverse the stroke direction
4. Move to next nearest undrawn stroke

This reduces total pen-lift distance significantly (NP-hard in general, but greedy works well in practice).

### Trajectory Planning Per Segment

Each pair of adjacent waypoints `(p_i, p_{i+1})` within a stroke is a call to `plan_linear_trajectory` from Week 3. This guarantees:
- **Straight-line** EE motion in Cartesian space
- **φ-continuity**: wrist orientation evolves smoothly (no elbow flips)
- **Servo velocity output**: `ω = Δθ / Δt` per joint

Pen-lifts between strokes are also planned with `plan_linear_trajectory` (higher speed, no line drawn).

### Servo Velocity Display

The right panel shows a rolling window of the last 200 frames:
- **ω₁** (red)  — base joint
- **ω₂** (blue) — elbow joint
- **ω₃** (green) — wrist joint

Units: **degrees/second**. These are the signals that would be sent to physical servo controllers.

---

## Files Saved

After processing, Week 4 automatically saves:
- `<image>_strokes.json` — all strokes and metadata (human-readable)
- `<image>_strokes.npz`  — fast-load numpy format for the pipeline
- `<image>_edges.png`    — the sketch image
