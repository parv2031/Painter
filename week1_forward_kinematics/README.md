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

| Property | Value (default arm) |
|----------|---------------------|
| Max reach | L₁+L₂+L₃ = **7.0** units |
| Min reach | \|L₁−L₂−L₃\| = **0.5** units |

---

## Visualizer Controls

| Control | Action |
|---------|--------|
| θ₁ slider | Rotate base joint |
| θ₂ slider | Rotate second joint |
| θ₃ slider | Rotate wrist joint |
| `R` key | Reset all joints to zero |
| `Q` key | Quit |
