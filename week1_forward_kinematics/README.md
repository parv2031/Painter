# Week 1 — Forward Kinematics

**Project:** Bob Ross without ROS — Robotics Society Summer Project  
**Mentors:** Anjaneya and Parv  
**Milestone:** PS Understanding + Forward Kinematics Implementation

---

## Files

| File | Purpose |
|------|---------|
| `forward_kinematics.py` | Core FK module — `ThreeLinkArm` class |
| `visualizer.py` | Interactive matplotlib GUI with joint-angle sliders |
| `demo.py` | Terminal demo — prints FK results for known configs |
| `test_forward_kinematics.py` | Unit tests |

---

## How to Run

```bash
cd week1_forward_kinematics

# 1. Quick terminal demo (no display needed)
python demo.py

# 2. Interactive GUI simulator
python visualizer.py

# 3. Unit tests (requires pytest or just runs manually)
python test_forward_kinematics.py
```

---

## Theory — Planar 3-Link Forward Kinematics

For a 3-DoF planar arm with link lengths **L₁, L₂, L₃** and joint angles **θ₁, θ₂, θ₃**:

### Joint Positions

```
J₀ = (0, 0)                                           ← base
J₁ = ( L₁·cos(θ₁),          L₁·sin(θ₁) )
J₂ = ( J₁ₓ + L₂·cos(θ₁+θ₂), J₁ᵧ + L₂·sin(θ₁+θ₂) )
J₃ = ( J₂ₓ + L₃·cos(θ₁+θ₂+θ₃), J₂ᵧ + L₃·sin(θ₁+θ₂+θ₃) )  ← end-effector
```

### End-Effector Position

```
x  = L₁·cos(θ₁) + L₂·cos(θ₁+θ₂) + L₃·cos(θ₁+θ₂+θ₃)
y  = L₁·sin(θ₁) + L₂·sin(θ₁+θ₂) + L₃·sin(θ₁+θ₂+θ₃)
φ  = θ₁ + θ₂ + θ₃    ← end-effector orientation
```

### Analytical Jacobian (2×3)

Maps joint velocities **θ̇** → end-effector velocity **ṗ = J·θ̇**

```
J = [ -L₁·sin(α₁)-L₂·sin(α₂)-L₃·sin(α₃)   -L₂·sin(α₂)-L₃·sin(α₃)   -L₃·sin(α₃) ]
    [  L₁·cos(α₁)+L₂·cos(α₂)+L₃·cos(α₃)    L₂·cos(α₂)+L₃·cos(α₃)    L₃·cos(α₃)  ]

where αᵢ is the cumulative angle sum up to joint i.
```

> The Jacobian will be used in **Week 3–4** for Jacobian-based inverse kinematics.

### Workspace

| Property | Formula | Value (default arm) |
|----------|---------|---------------------|
| Max reach | L₁ + L₂ + L₃ | **7.0** units |
| Min reach | max(0, L₁ − L₂ − L₃) | **0** — no dead zone |

**Why min reach = 0:**  
The condition `L₂ + L₃ ≥ L₁` (4.0 ≥ 3.0) means links 2 and 3 together are long enough to fold back and cancel link 1's extension entirely. The arm can reach all the way to its own base, so the workspace is a **full disc** of radius 7.0, not an annulus.

> **Note:** The formula `|L₁ − L₂ − L₃|` for minimum reach is only correct for a **2-link arm**. For a 3-link arm with unlimited joint rotation, the correct formula is `max(0, L₁ − L₂ − L₃)`.

---

## Visualizer Controls

| Control | Action |
|---------|--------|
| θ₁ slider | Rotate base joint |
| θ₂ slider | Rotate second joint |
| θ₃ slider | Rotate wrist joint |
| `R` key | Reset all joints to zero |
| `Q` key | Quit |

---

## Difficulties & Fixes

### D1 — Wrong minimum-reach formula (geometry bug)

**Difficulty:**  
The initial implementation computed the inner workspace boundary using `|L₁ − L₂ − L₃|`, which is the correct formula only for a **2-link arm**. For our 3-link arm this produced `|3.0 − 2.5 − 1.5| = 1.0`, implying a dead zone of radius 1.0 — i.e., that the arm cannot reach its own base. The visualizer drew an inner exclusion circle, and a test asserted `min_reach == 1.0`.

**Root cause:**  
For a 2-link arm with lengths A and B, the minimum reach is `|A − B|` because the only way to minimise reach is to extend one link against the other. For a 3-link arm, links 2 and 3 can fold back together and together cancel link 1's extension. The minimum distance from base to EE is therefore:

```
min_reach = max(0,  L₁ − (L₂ + L₃))
```

For our arm: `max(0, 3.0 − 4.0) = 0` — the arm can reach its own base.

**Fix:**  
- Corrected `min_reach` formula in `forward_kinematics.py` to `max(0, L1 - L2 - L3)`.  
- Removed the inner dead-zone circle from `visualizer.py`.  
- Updated the workspace table in this README and corrected the related unit tests.

---

### D2 — NumPy / Matplotlib version conflict (`_ARRAY_API not found`)

**Difficulty:**  
Running any script that imported both `matplotlib` and `numpy` raised:

```
AttributeError: _ARRAY_API not found
```

This caused `demo.py`, `visualizer.py`, and all tests to crash on import.

**Root cause:**  
The system-installed `matplotlib` was compiled against NumPy 1.x, but `pip` had installed NumPy 2.x. When Python loaded both packages, the C-level array API ABI mismatch triggered the error.

**Fix:**  

```bash
pip3 install --upgrade "matplotlib>=3.8"
```

Matplotlib ≥ 3.8 ships wheels compiled against NumPy 2.x. After upgrading, all imports succeeded. A harmless `Axes3D` deprecation warning may still appear from a residual system package — it can be ignored.

---

### D3 — Jacobian column ordering ambiguity

**Difficulty:**  
Early drafts of the Jacobian had the columns ordered as `[∂/∂θ₃, ∂/∂θ₂, ∂/∂θ₁]` (base to tip reversed), which gave correct magnitudes but wrong signs when used for gradient-descent IK in later weeks.

**Fix:**  
Columns are now explicitly ordered `[∂/∂θ₁, ∂/∂θ₂, ∂/∂θ₃]` matching the joint-angle vector convention. A unit test verifies that `J · [1, 0, 0]` produces the EE velocity for joint-1-only motion.

