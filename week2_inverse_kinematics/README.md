# Week 3-4 — Inverse Kinematics

**Project:** Bob Ross without ROS — Robotics Society Summer Project  
**Mentors:** Anjaneya and Parv  
**Milestone:** Inverse Kinematics Implementation

---

## Files

| File | Purpose |
|------|---------|
| `inverse_kinematics.py` | Two IK solvers — Analytical + Jacobian pseudo-inverse |
| `visualizer_ik.py` | Click-to-reach interactive GUI |
| `demo_ik.py` | Terminal demo — compares both solvers |
| `test_inverse_kinematics.py` | Unit tests |

---

## How to Run

```bash
cd week2_inverse_kinematics

# Terminal demo (no display needed)
python3 demo_ik.py

# Interactive click-to-reach visualizer
python3 visualizer_ik.py

# Unit tests
python3 -m pytest test_inverse_kinematics.py -v
```

---

## Theory — Inverse Kinematics

IK is the **reverse problem** of FK:

```
FK:  [θ₁, θ₂, θ₃]  →  (x, y, φ)       ← unique
IK:  (x, y, φ)      →  [θ₁, θ₂, θ₃]   ← up to 2 solutions (elbow-up / elbow-down)
IK:  (x, y)         →  [θ₁, θ₂, θ₃]   ← infinitely many (redundant, 3 DOF for 2 tasks)
```

---

## Solver 1 — Analytical (Geometric)

**When to use:** Full pose target `(x, y, φ)` is known (e.g., we know both where to paint and at what brush angle).

**Steps:**

### 1. Find the wrist position
Remove link 3's contribution to get the point that the 2-link sub-arm must reach:
```
wx = x − L₃·cos(φ)
wy = y − L₃·sin(φ)
d  = √(wx² + wy²)
```

### 2. Solve θ₂ (law of cosines)
```
cos θ₂ = (d² − L₁² − L₂²) / (2·L₁·L₂)
θ₂ = ±arccos(cos θ₂)
```
- `+` → **elbow-down**
- `−` → **elbow-up**

### 3. Solve θ₁
```
θ₁ = atan2(wy, wx) − atan2(L₂·sin θ₂,  L₁ + L₂·cos θ₂)
```

### 4. Recover θ₃
```
θ₃ = φ − θ₁ − θ₂
```

**Properties:**
- ✅ Exact, instantaneous (no iterations)
- ✅ Two explicit solutions (elbow-up / elbow-down)
- ❌ Requires the full target orientation φ to be specified

---

## Solver 2 — Jacobian Pseudo-inverse (Numerical)

**When to use:** Only a position target `(x, y)` is known (no φ constraint). The arm is **redundant** (3 DOF, 2 task variables), so infinitely many configurations exist — the pseudo-inverse picks the one requiring the smallest joint motion.

**Update rule (per iteration):**
```
Δθ = J⁺ · Δp  +  α · (I − J⁺J) · (θ_pref − θ)
```

| Term | Meaning |
|------|---------|
| `J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹` | Damped least-squares pseudo-inverse |
| `J⁺ · Δp` | Move EE toward target |
| `(I − J⁺J)` | Null-space projector — motion that doesn't affect EE |
| `α·(θ_pref − θ)` | Drift joints toward a preferred configuration |
| `λ` | Damping — prevents instability near singularities |

**Properties:**
- ✅ No φ constraint needed
- ✅ Uses redundancy to keep joints in a preferred configuration
- ✅ Robust near singularities (damping λ)
- ❌ Iterative (slower than analytical)
- ❌ No guarantee of global optimum

---

## Workspace & Reachability

| Property | Formula | Value |
|----------|---------|-------|
| Max reach | L₁ + L₂ + L₃ | 7.0 |
| Min reach | max(0, L₁ − L₂ − L₃) | 0 (full disc) |

Since `L₂ + L₃ = 4.0 ≥ L₁ = 3.0`, the arm can reach **any point** within a radius of 7.0 from the base, including the base itself.

---

## Visualizer Controls

| Control | Action |
|---------|--------|
| **Left-click** on canvas | Set IK target — arm reaches for that point |
| **Solver radio** | Switch between Elbow-up / Elbow-down / Jacobian |
| **φ slider** | Set target orientation (Analytical solvers only) |
| `R` key | Reset arm to zero configuration |
| `Q` key | Quit |
