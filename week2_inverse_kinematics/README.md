# Week 2 — Inverse Kinematics

**Project:** Bob Ross without ROS — Robotics Society Summer Project  
**Mentors:** Anjaneya and Parv  
**Milestone:** Inverse Kinematics Implementation

---

## Files

| File | Purpose |
|------|---------|
| `inverse_kinematics.py` | Three IK solvers — Analytical, Auto-φ, Jacobian pseudo-inverse |
| `visualizer_ik.py` | Click-to-reach interactive GUI with solver selector |
| `demo_ik.py` | Terminal demo comparing all solvers |
| `test_inverse_kinematics.py` | 56 unit tests covering full workspace sweep |

---

## How to Run

```bash
cd week2_inverse_kinematics

python3 demo_ik.py               # terminal output
python3 visualizer_ik.py         # interactive click-to-reach GUI
python3 -m pytest test_inverse_kinematics.py -v
```

---

## 1. The IK Problem

**Forward Kinematics (FK)** maps joint angles to end-effector pose — it is always unique:

```
FK:  [θ₁, θ₂, θ₃]  ──────→  (x, y, φ)
```

**Inverse Kinematics (IK)** goes the other way: given a desired position (and possibly orientation), find the joint angles that achieve it:

```
IK:  (x, y, φ)  ──────→  [θ₁, θ₂, θ₃]   (up to 2 solutions)
IK:  (x, y)     ──────→  [θ₁, θ₂, θ₃]   (infinitely many — redundant arm)
```

IK is fundamentally harder than FK because:
- Solutions may not exist (out-of-reach target)
- Multiple solutions may exist (elbow-up vs elbow-down)
- The redundancy of 3 DOF with 2 task variables means we must choose among infinitely many valid configurations

---

## 2. Workspace Analysis

The reachable workspace of a 3-link arm is determined by the link lengths.

### End-effector position
```
x = L₁·cos(θ₁) + L₂·cos(θ₁+θ₂) + L₃·cos(θ₁+θ₂+θ₃)
y = L₁·sin(θ₁) + L₂·sin(θ₁+θ₂) + L₃·sin(θ₁+θ₂+θ₃)
```

### Workspace bounds

| Bound | Formula | Value (our arm) |
|-------|---------|-----------------|
| Max reach | L₁ + L₂ + L₃ | **7.0** (all links fully extended) |
| Min reach | max(0, L₁ − L₂ − L₃) | **0** (full disc — L₂+L₃ ≥ L₁) |

Since `L₂ + L₃ = 4.0 ≥ L₁ = 3.0`, the workspace is a **full disc of radius 7.0**. The arm can reach its own base.

---

## 3. Analytical IK — `ik_analytical(arm, x, y, φ, elbow_up)`

### When to use
Full pose target `(x, y, φ)` is specified. Gives exact, instantaneous solution.

### Step-by-step derivation

#### Step 1 — Find the wrist position

The end-effector (tip of link 3) is at `(x, y)`. Link 3 points in direction `φ`, so the joint between links 2 and 3 (the *wrist*) is:

```
wx = x − L₃·cos(φ)
wy = y − L₃·sin(φ)
d  = √(wx² + wy²)     ← distance from base to wrist
```

This reduces the problem to a **2-link IK** for the sub-arm (L₁, L₂) that must reach the wrist point `(wx, wy)`.

#### Step 2 — Law of Cosines for θ₂

In the triangle formed by links L₁ and L₂ with the base-to-wrist line `d`:

```
              J2 (wrist)
             /|
           L2 | d
           /  |
         J1   |
          \   |
           L1 |
            \ |
             J0 (base)
```

By the **law of cosines**:

```
d² = L₁² + L₂² − 2·L₁·L₂·cos(π − θ₂)
   = L₁² + L₂² + 2·L₁·L₂·cos(θ₂)
```

Rearranging:

```
cos(θ₂) = (d² − L₁² − L₂²) / (2·L₁·L₂)
θ₂      = ±arccos(cos θ₂)
```

The **± gives two solutions**:
- `θ₂ > 0` → **elbow-down** (joint 2 bends in the same direction as the base angle)
- `θ₂ < 0` → **elbow-up** (joint 2 bends the other way)

#### Step 3 — Solve θ₁

Using the **two-argument arctangent** to find the angle to the wrist:

```
θ₁ = atan2(wy, wx) − atan2(L₂·sin θ₂,  L₁ + L₂·cos θ₂)
```

**Intuition:** `atan2(wy, wx)` is the angle from the base to the wrist.
The second `atan2` term is the angle that link 1 makes relative to the base-to-wrist line,
accounting for the elbow bend.

#### Step 4 — Recover θ₃

Since the total orientation is `φ = θ₁ + θ₂ + θ₃`:

```
θ₃ = φ − θ₁ − θ₂
```

#### Reachability condition

The wrist must be reachable by the 2-link sub-arm:

```
|L₁ − L₂| ≤ d ≤ L₁ + L₂
   0.5    ≤ d ≤   5.5
```

If `cos(θ₂)` falls outside `[−1, 1]`, the target is unreachable and we clamp + return failure.

---

## 4. The φ Problem — Why Fixed φ Fails Near the Base

For `ik_analytical` to succeed, the **wrist** (not the end-effector) must be reachable by the 2-link arm.

If we naively set `φ = atan2(y, x)` (pointing radially outward), the wrist ends up at distance:

```
d_w = |r − L₃|   where r = √(x²+y²)
```

For targets at `r ∈ (|L₁−L₂|, L₃)` = `(0.5, 1.5)`, we get `d_w < 0.5` — **the wrist falls inside the 2-link dead zone**, causing failure even though the point is physically reachable by the full arm.

This is the motivation for automatic φ selection.

---

## 5. Auto-φ IK — `ik_analytical_auto(arm, x, y, elbow_up)`

### Core idea

For a target at `(x, y)` with `r = √(x²+y²)`, `α = atan2(y, x)`, the wrist distance is:

```
d_w² = (x − L₃·cos φ)² + (y − L₃·sin φ)²
     = x² + y² + L₃² − 2·L₃·(x·cos φ + y·sin φ)
     = r² + L₃² − 2·L₃·r·cos(φ − α)          ... (★)
```

> This uses the identity: `x·cos φ + y·sin φ = r·cos(φ − α)` (projection formula).

### Finding the valid φ range

We need `d_min ≤ d_w ≤ d_max`, i.e.:

```
d_min² ≤ r² + L₃² − 2·L₃·r·cos(φ − α) ≤ d_max²
```

Rearranging for `cos(φ − α)`:

```
c_lo = (r² + L₃² − d_max²) / (2·L₃·r)      ← lower bound
c_hi = (r² + L₃² − d_min²) / (2·L₃·r)      ← upper bound
```

Any `φ` satisfying `c_lo ≤ cos(φ − α) ≤ c_hi` will place the wrist in the reachable zone.

### Choosing the optimal φ

We pick the **midpoint** of the valid cosine range:

```
c* = (c_lo + c_hi) / 2
φ  = α ± arccos(c*)
```

The midpoint maximizes the conditioning of the 2-link sub-problem (wrist is as far as possible from both the inner and outer limits).

### Strategy (in priority order)

```
1. Try φ = α            (natural, radially outward — works for most of workspace)
2. Compute optimal φ    (midpoint of valid cosine range — handles near-base zone)
3. Brute-force scan     (72 uniformly spaced φ values — guaranteed fallback)
```

### Coverage proof

For any point at radius `r ∈ [0, L₁+L₂+L₃]`:

- The valid range of `cos(φ−α)` is non-empty because we can always choose
  `d_w = (d_min + d_max)/2 = 3.0` which is always achievable for `r ≤ 7.0`.
- Therefore `ik_analytical_auto` **covers the full workspace disc** — verified
  by 40 parametrized test cases spanning all radii and quadrants.

---

## 6. Jacobian Pseudo-inverse IK — `ik_jacobian(arm, x, y, ...)`

### When to use
Only position `(x, y)` is known, no orientation constraint. The arm is **redundant** (3 DOF, 2 task variables) — the Jacobian method exploits the extra DOF.

### The Jacobian

The **velocity-level relationship** between joint velocities `θ̇` and EE velocity `ṗ`:

```
ṗ = J · θ̇      where  J is 2×3
```

The 2×3 Jacobian J (derived in Week 1):

```
        ⎡ −L₁sin(α₁)−L₂sin(α₂)−L₃sin(α₃)   −L₂sin(α₂)−L₃sin(α₃)   −L₃sin(α₃) ⎤
J =     ⎣  L₁cos(α₁)+L₂cos(α₂)+L₃cos(α₃)    L₂cos(α₂)+L₃cos(α₃)    L₃cos(α₃)  ⎦

where αᵢ = θ₁ + θ₂ + … + θᵢ  (cumulative joint angles)
```

### The pseudo-inverse

J is 2×3 (more unknowns than equations). The **Moore-Penrose pseudo-inverse** J⁺ gives the minimum-norm solution:

```
θ̇_min = J⁺ · ṗ        where  J⁺ = Jᵀ·(J·Jᵀ)⁻¹   (right pseudo-inverse)
```

### Damped Least Squares (DLS)

Near **singularities** (arm fully extended/folded), `J·Jᵀ` becomes nearly singular and the pseudo-inverse explodes. The **Levenberg-Marquardt** damping term `λ²I` regularises it:

```
J⁺_DLS = Jᵀ · (J·Jᵀ + λ²·I)⁻¹
```

- `λ = 0`: pure pseudo-inverse (unstable at singularities)
- `λ > 0`: trades accuracy for stability
- Typical value: `λ = 0.03`

### Null-space redundancy resolution

The **null-space** of J is the set of joint motions that produce **zero EE motion**:

```
N = I − J⁺·J     (null-space projector, 3×3)
```

Any vector `z` can be added to the solution without changing the EE position:

```
θ̇ = J⁺ · ṗ  +  N · z
```

We use this to **pull joints toward a preferred configuration** `θ_pref`:

```
z = α · (θ_pref − θ)
```

The full iterative update rule is:

```
Δθ = step · J⁺_DLS · Δp  +  α · (I − J⁺_DLS · J) · (θ_pref − θ)
θ  ← θ + Δθ    (repeated until ||Δp|| < tolerance)
```

| Parameter | Role | Default |
|-----------|------|---------|
| `step_size` | Fraction of error corrected per step | 0.3 |
| `damping` λ | Singularity protection | 0.03 |
| `null_space_gain` α | Strength of preference pull | 0.2 |
| `max_iter` | Max iterations before giving up | 2000 |
| `tol` | Convergence threshold (world units) | 1e-4 |

### Singularities

A singularity occurs when the Jacobian loses rank (det(J·Jᵀ) → 0). For a 3-link arm:

- **Boundary singularity**: arm fully extended (`θ₂ = θ₃ = 0`) — cannot move EE along the arm axis
- **Elbow singularity**: links 2 and 3 aligned (`θ₂ = 0`) — two DOF become equivalent

The DLS damping makes the Jacobian method robust near these configurations.

---

## 7. Comparison of Solvers

| Property | `ik_analytical` | `ik_analytical_auto` | `ik_jacobian` |
|---|---|---|---|
| **Input required** | `(x, y, φ)` — full pose | `(x, y)` only | `(x, y)` only |
| **Speed** | Instant | Instant | ~500–2000 iterations |
| **Orientation** | User-specified φ | Auto-selected | Free (not constrained) |
| **Full workspace** | ❌ (fails if φ causes bad wrist) | ✅ | ✅ |
| **Elbow choice** | ✅ up/down | ✅ up/down | ❌ (depends on init) |
| **Singularity safe** | ✅ (check d bounds) | ✅ | ✅ (DLS damping) |
| **Best for** | Known brush angle | Painting (reach anywhere) | Redundancy exploitation |

---

## 8. Visualizer Controls

| Control | Action |
|---------|--------|
| **Left-click** on canvas | Arm reaches for that point |
| **Auto-φ Elbow-up/dn** | Best solver — covers entire disc |
| **Analytical Elbow-up/dn** | Fixed φ from slider |
| **Jacobian Pseudo-inv** | Iterative, position-only |
| **φ slider** | Orientation target (Analytical only) |
| `R` | Reset arm | `Q` | Quit |

---

## 9. Difficulties & Fixes

### D1 — Fixed φ caused a dead zone despite the arm being able to reach all points

**Difficulty:**  
The initial `ik_analytical` solver required the caller to supply φ (end-effector orientation). When `φ = atan2(y, x)` was used as a default, the solver silently failed for targets in the ring `r ∈ (0.5, 1.5)` — even though those points are physically reachable by the 3-link arm.

**Root cause:**  
With `φ = α = atan2(y, x)`, the wrist lands at distance `d_w = |r − L₃|`. For `r ∈ (|L₁−L₂|, L₃) = (0.5, 1.5)`:

```
d_w = L₃ − r  <  |L₁ − L₂| = 0.5
```

The wrist falls inside the 2-link arm's own dead zone, so the 2-link sub-problem fails — even though the full 3-link arm can reach the point by bending differently.

**Fix — `ik_analytical_auto`:**  
The wrist distance formula was expanded as a function of φ:

```
d_w² = r² + L₃² − 2·L₃·r·cos(φ − α)          ... (★)
```

Rearranging gives the exact range of `cos(φ − α)` for which the wrist is reachable:

```
c_lo = (r² + L₃² − d_max²) / (2·L₃·r)
c_hi = (r² + L₃² − d_min²) / (2·L₃·r)
```

The midpoint `c* = (c_lo + c_hi)/2` gives the best-conditioned φ, and `φ = α ± arccos(c*)` produces two candidate orientations. Both are tried; the one that gives a valid IK solution is returned. This approach provably covers the full workspace disc.

---

### D2 — Jacobian solver not converging (max iterations reached)

**Difficulty:**  
Three tests in `test_inverse_kinematics.py` failed because `ik_jacobian` hit `max_iter=500` with position errors of ~1e-3 (just above `tol=1e-4`).

**Root cause:**  
The initial hyperparameters — `step_size=0.5`, `damping=0.05`, `null_space_gain=0.3`, `max_iter=500` — were too aggressive. A large step size caused oscillation around the target rather than convergence. High null-space gain competed with the primary task near convergence.

**Fix:**  
Tuned parameters:

| Parameter | Before | After |
|-----------|--------|-------|
| `step_size` | 0.5 | 0.3 |
| `damping` | 0.05 | 0.03 |
| `null_space_gain` | 0.3 | 0.2 |
| `max_iter` | 500 | 2000 |

Lower step size prevents oscillation; higher max_iter allows convergence on difficult configurations. All 15 Jacobian tests pass.

---

### D3 — Orientation angle wrap causing `test_orientation_preserved` failures

**Difficulty:**  
The test asserting that `φ_actual == φ_target` after IK sometimes failed for targets where the solved angles summed to a value outside `(−π, π]`. For example, a target with `φ = 2.8 rad` would be matched by joint angles summing to `2.8 − 2π ≈ −3.48 rad`, which is the same orientation but differs numerically.

**Root cause:**  
The test used `abs(phi_actual − phi_target) < tol`, which is wrong for circular quantities. An orientation of `−3.48 rad` is identical to `2.8 rad` but the raw difference is `6.28`.

**Fix:**  
The test was updated to use the shortest angular difference:

```python
assert abs((phi_actual − phi + np.pi) % (2*np.pi) − np.pi) < 1e-4
```

This correctly handles orientation wrap-around for any φ ∈ ℝ.

---

### D4 — Auto-φ test: `phi_chosen` key absent from result dict

**Difficulty:**  
The `test_result_has_phi_chosen_on_success` test initially failed because `ik_analytical_auto` did not write `phi_chosen` into the result dict — it was only added when a specific branch succeeded, not uniformly.

**Fix:**  
The inner helper `_try(phi_candidate)` was updated to always write `res["phi_chosen"] = phi_candidate` when the IK succeeds, regardless of which strategy (natural, computed, or scan) found the solution.

