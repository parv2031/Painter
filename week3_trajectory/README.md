# Week 3 — Linear Trajectory Planning

**Project:** Bob Ross without ROS — Robotics Society Summer Project  
**Mentors:** Anjaneya and Parv  
**Milestone:** Straight-Line Cartesian Trajectory + Servo Velocity Output

---

## Files

| File | Purpose |
|------|---------|
| `trajectory.py` | Core module — interpolation, IK loop, servo velocity computation |
| `simulator.py` | Interactive GUI — click P1 / P2, animate, live velocity plots |
| `demo_trajectory.py` | Terminal demo — prints joint angles + velocities per step |
| `test_trajectory.py` | 21 unit tests |

---

## How to Run

```bash
cd week3_trajectory

python3 demo_trajectory.py   # terminal output
python3 simulator.py         # interactive GUI
python3 -m pytest test_trajectory.py -v
```

---

## Simulator Workflow

| Step | Action |
|------|--------|
| 1 | **Left-click** → set start point **P1** (green dot) |
| 2 | **Left-click** → set end point **P2** (red dot) — planned path appears |
| 3 | **SPACE** or `▶ Play` → animation runs |
| Live | Right panel shows ω₁, ω₂, ω₃ servo velocity time-series updating each frame |
| `EE speed slider` | Set end-effector travel speed (units/s) |
| `R` | Reset | `Q` | Quit |

---

## 1. Problem Statement

Given two reachable points P1 and P2 in the arm's workspace, move the end-effector along the **straight line** between them at **constant speed**, while:

- Solving IK at every waypoint along the line
- Maintaining smooth, continuous joint motion (no sudden flips)
- Outputting the **angular velocity of each servo** at every time step

---

## 2. Cartesian Linear Interpolation

A straight line from P1 to P2 is parameterised by `t ∈ [0, 1]`:

```
p(t) = (1 − t)·P1 + t·P2
```

With `N` uniformly spaced `t` values, each consecutive pair of waypoints is separated by:

```
Δs = ‖P2 − P1‖ / (N − 1)    ← arc-length step (constant)
```

Because arc-length is parameterised uniformly, the end-effector speed is exactly:

```
v_ee = Δs / Δt = constant
```

This is **constant-speed Cartesian motion** — a fundamental requirement for smooth painting strokes.

### Choosing N (number of steps)

```
N = clip( ‖P2 − P1‖ / (v_ee · Δt),  N_min,  N_max )
```

where `Δt = 20 ms` (50 Hz servo update rate). Minimum 50 steps, maximum 500.

---

## 3. IK at Each Waypoint

At each waypoint `p(t)`, we solve IK to get joint angles. The naive approach — calling `ik_analytical_auto` independently at each step — causes **elbow flips** (see Difficulties). The correct approach uses **φ-continuity**.

### φ-Continuity Algorithm

The key insight is that the EE orientation `φ = θ₁ + θ₂ + θ₃` must evolve **smoothly** along the trajectory. If φ jumps discontinuously, joint angles jump too — producing velocity spikes.

**Step 1 — Extract previous φ:**
```
φ_prev = θ₁[k−1] + θ₂[k−1] + θ₃[k−1]
```

**Step 2 — Compute valid φ range at the new waypoint:**

From the wrist distance equation (★ from Week 2):
```
d_w² = r² + L₃² − 2·L₃·r·cos(φ − α)
```

For the wrist to be reachable: `d_min ≤ d_w ≤ d_max`, which gives:
```
c_lo = (r² + L₃² − d_max²) / (2·L₃·r)    ← lower bound on cos(φ − α)
c_hi = (r² + L₃² − d_min²) / (2·L₃·r)    ← upper bound on cos(φ − α)
```

The valid range of `|φ − α|` (denoted δ) is therefore `[arccos(c_hi), arccos(c_lo)]`:
- `δ < arccos(c_hi)` → wrist too close to origin (forbidden)
- `δ > arccos(c_lo)` → wrist too far from origin (also forbidden)

**Step 3 — Clamp φ_prev to the valid range:**

```
δ_prev = φ_prev − α   (wrapped to (−π, π])

if |δ_prev| < δ_min:   φ* = α ± δ_min   ← snap to inner boundary
if |δ_prev| > δ_max:   φ* = α ± δ_max   ← snap to outer boundary
else:                   φ* = φ_prev      ← already valid, keep it
```

This produces the φ **closest to the previous value** that keeps the wrist reachable — eliminating discontinuous φ jumps.

**Step 4 — Solve `ik_analytical(arm, x, y, φ*, elbow_up)` for both elbow modes, pick the one with minimum joint displacement:**

```
cost(solution) = Σ |angle_diff(θᵢ_new, θᵢ_prev)|
```

---

## 4. Servo Velocity Computation

For a servo motor, the commanded quantity is **angular velocity** `ω`. Given consecutive joint angle arrays `θ[k]` and `θ[k-1]` separated by time step `Δt`:

```
ω_i[k] = angle_diff(θ_i[k], θ_i[k−1]) / Δt       [rad/s]
```

The **shortest angular difference** function handles wrap-around correctly:

```
angle_diff(a, b) = ((a − b + π) mod 2π) − π       ∈ (−π, π]
```

**Why shortest diff matters:**  
A servo rotating from 170° to −170° physically travels only 20° (not 340°). Using the raw difference `170 − (−170) = 340°` would command 17× too high a velocity. The wrap-corrected formula gives the true servo motion.

### Output format

```
At each timestep k (k = 1 … N−1):
  ω₁[k],  ω₂[k],  ω₃[k]    [rad/s or deg/s]
```

These are the signals that would be sent to physical servo controllers.

---

## 5. Kinematic Singularities in Trajectory Following

A straight Cartesian path does **not** in general correspond to smooth, low-velocity joint motion. Near a **kinematic singularity** (a configuration where the Jacobian rank drops), small EE displacements require arbitrarily large joint velocities.

### Types of singularity for a 3-link arm

| Type | Configuration | Symptom |
|------|--------------|---------|
| Boundary | All links extended/folded along the same line | Cannot move EE perpendicular to arm |
| Elbow | θ₂ = 0 — links 1 & 2 collinear | θ₁ and θ₃ become degenerate |
| Wrist fold | θ₂ = ±180° — arm fully folded back | Small EE motion → large θ₂ rate |

### How to identify a singularity

The **condition number** of the Jacobian `J` diverges at singularities:

```
κ(J) = σ_max / σ_min → ∞
```

where `σ` are the singular values. In the simulator, a smooth velocity hill in the plot indicates a singularity region, not a bug.

### Singularity vs Elbow Flip

| | Elbow Flip (bug) | Singularity (physics) |
|---|---|---|
| Shape in velocity plot | Single isolated spike | Smooth rise-and-fall hill |
| Duration | 1 step | Several steps |
| Cause | Discontinuous φ jump in code | Degenerate Jacobian |
| Fix | φ-continuity algorithm | Path re-routing (future work) |

---

## 6. Difficulties & Fixes

### D1 — Elbow flip: arm suddenly jumped to a completely different configuration mid-trajectory

**Difficulty:**  
The simulator showed massive velocity spikes of **8000+ deg/s** in a single step. Visually, the arm's elbow would flip from pointing up to pointing down mid-stroke. The velocity charts showed a single isolated spike flanked by normal values.

**Root cause:**  
`_nearest_ik` (the original implementation) called `ik_analytical_auto` independently at each waypoint. `ik_analytical_auto` freely re-selects φ based only on the current point's geometry. As the arm moved along the line, φ could jump by up to 90° between two adjacent steps. This caused the wrist position to shift discontinuously, and the 2-link sub-problem to suddenly flip between elbow-up and elbow-down solutions.

The comparison `min joint displacement` between only two pre-selected solutions (elbow-up auto, elbow-down auto) was insufficient — both solutions at the new φ could be far from the previous configuration.

**Fix — φ-continuity algorithm:**  
Three new functions were introduced:

- **`_valid_phi_range`** — analytically computes `[δ_min, δ_max]`, the range of `|φ − α|` that keeps the wrist reachable.
- **`_clamp_phi_to_valid`** — takes φ_prev and clamps it to the nearest point in the valid range. This ensures φ evolves with the smallest possible step at each waypoint.
- **Updated `_nearest_ik`** — uses φ* from the clamped value, tries both elbow modes at φ*, picks minimum displacement. Falls back to a 72-candidate dense scan only if the primary candidates fail.

**Result:** Isolated spikes (8000+ deg/s over 1 step) eliminated. Remaining high-velocity regions are smooth hills caused by genuine singularities.

---

### D2 — Step-0 velocity spike from cold-start arm position

**Difficulty:**  
Even with the φ-continuity fix, there remained a large velocity spike at step 0. The ratio of max-to-median velocity for ω₂ was 11× and the spike was at index 0.

**Root cause:**  
The bootstrap used `_nearest_ik` with `prev_angles = arm.joint_angles = [0, 0, 0]` (zero config). From the zero config, `φ_prev = 0`, and clamping that to the valid range at P1 landed the arm at `θ₂ = 0°` — a singularity (links 1 and 2 collinear). The very next step then required a large `θ₂` change to escape the singularity, causing the spike.

**Fix:**  
The bootstrap was changed to use `ik_analytical_auto` (not `_nearest_ik`) for the very first waypoint (P1). `ik_analytical_auto` picks the geometrically natural starting orientation (φ = α, pointing radially), which places the arm in a well-conditioned configuration. φ-continuity then takes over from step 1 onward.

```python
# Before: _nearest_ik from zero-config (lands in singularity)
# After:  ik_analytical_auto at P1 (picks natural, non-singular config)
boot_res = ik_analytical_auto(arm, x1, y1, elbow_up=True)
prev_angles = boot_res["angles"]
```

**Result:** Step-0 spike eliminated.

---

### D3 — `angle_diff` wrap direction confused in tests

**Difficulty:**  
Two tests in `TestAngleDiff` failed with the assertion error:

```
angle_diff(radians(170), radians(-170)) == radians(20)  → FAILED
```

The actual result was `−0.349 rad (−20°)`.

**Root cause:**  
The test comment was wrong about the sign convention. `angle_diff(a, b) = a − b` (wrapped). Going from `b = −170°` to `a = +170°` is a **+20°** step (CCW). Going from `b = +170°` to `a = −170°` is a **−20°** step (CW). The test had the arguments and expected signs swapped.

**Fix:**  
Swapped the arguments and corrected the expected signs:

```python
# +20°: -170° to +170° (a - b = +20°)
assert angle_diff(radians(-170), radians(170)) ≈ radians(+20)
# -20°: +170° to -170° (a - b = -20°)
assert angle_diff(radians(+170), radians(-170)) ≈ radians(-20)
```

---

### D4 — High-velocity regions near singularities are expected, not bugs

**Difficulty:**  
After fixes D1 and D2, cross-quadrant trajectories (e.g., P1=(−4.5, 2.2) → P2=(1.5, −0.2)) still showed maximum velocities of ~1500 deg/s in a smooth hill shape. Initial concern was that elbow flips had not been fully resolved.

**Root cause (clarified):**  
Diagnostic output showed that for this trajectory, `θ₂` smoothly decreases from −101° toward −179° (near full fold), then reverses. At the fold point, even a tiny EE displacement requires a large `θ₂` motion — this is a **wrist-fold singularity**. The velocity increase is physically real, not a code artefact.

**Confirmation:**  
```
Velocity profile is smooth (gradual ramp, no isolated spike)
→ This is a kinematic singularity, not an elbow flip
```

**Resolution:**  
Documented as expected behaviour. The distinction between a flip (bug) and a singularity (physics) is now explicitly stated in the README (see Section 5 table above). Future work: re-route paths to avoid singularity regions using Jacobian condition-number monitoring.
