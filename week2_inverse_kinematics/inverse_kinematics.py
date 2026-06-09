"""
Week 3-4 — Inverse Kinematics for a 3-Link Planar Robotic Arm
==============================================================
Project: Bob Ross without ROS — Robotics Society Summer Project
Mentors: Anjaneya and Parv

Two solvers are provided:

1. Analytical (Geometric) IK  ─ closed-form, fast, exact.
   Works when the full end-effector pose (x, y, φ) is specified.

   Steps:
     a) Subtract link-3 to find the *wrist* position:
            wx = x - L3·cos(φ)
            wy = y - L3·sin(φ)
     b) Solve the 2-link sub-problem (L1, L2) for the wrist:
            cos θ2 = (wx² + wy² - L1² - L2²) / (2·L1·L2)
            θ2     = ±arccos(cos θ2)           ← elbow-up / elbow-down
            θ1     = atan2(wy, wx) - atan2(L2·sin θ2, L1 + L2·cos θ2)
     c) Recover wrist angle:
            θ3 = φ - θ1 - θ2

2. Jacobian Pseudo-inverse (Numerical) IK  ─ iterative, handles
   position-only targets (no orientation constraint).
   Because the arm has 3 DOF but only 2 task-space variables (x, y),
   it is *redundant*.  The pseudo-inverse gives the minimum-norm
   joint-velocity solution at each step, with optional null-space
   damping to keep joints near a preferred configuration.

   Update rule:
       Δθ = J⁺ · Δp  +  α·(I - J⁺·J)·(θ_pref - θ)
   where J⁺ = Jᵀ·(J·Jᵀ + λ²I)⁻¹  (damped least-squares)
"""

import numpy as np
import sys, os

# Allow importing ThreeLinkArm from the week1 folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1_forward_kinematics"))
from forward_kinematics import ThreeLinkArm


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass (plain dict for zero extra dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def _ik_result(success, angles, iterations=0, error=0.0, note=""):
    return {
        "success": success,
        "angles": np.array(angles, dtype=float),
        "iterations": iterations,
        "position_error": float(error),
        "note": note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analytical IK
# ─────────────────────────────────────────────────────────────────────────────

def ik_analytical(
    arm: ThreeLinkArm,
    x: float,
    y: float,
    phi: float,
    elbow_up: bool = True,
) -> dict:
    """
    Closed-form geometric IK for a fully-specified pose (x, y, φ).

    Parameters
    ----------
    arm      : ThreeLinkArm instance (provides link lengths).
    x, y     : Target end-effector position.
    phi      : Target end-effector orientation (radians).
    elbow_up : Choose elbow-up (True) or elbow-down (False) solution.

    Returns
    -------
    dict with keys: success, angles [θ1, θ2, θ3], position_error, note.
    """
    L1, L2, L3 = arm.link_lengths

    # ── Step 1: wrist position ────────────────────────────────────────────
    wx = x - L3 * np.cos(phi)
    wy = y - L3 * np.sin(phi)
    d  = np.hypot(wx, wy)          # distance from base to wrist

    # ── Step 2: reachability check for the 2-link sub-problem ────────────
    if d > L1 + L2 + 1e-9:
        return _ik_result(False, [0, 0, 0], note="Wrist out of reach (too far)")
    if d < abs(L1 - L2) - 1e-9:
        return _ik_result(False, [0, 0, 0], note="Wrist out of reach (too close)")

    # ── Step 3: θ2 via law of cosines ────────────────────────────────────
    cos_θ2 = np.clip((d**2 - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    θ2 = np.arccos(cos_θ2)          # elbow-down (+)
    if elbow_up:
        θ2 = -θ2                    # elbow-up (-)

    # ── Step 4: θ1 ────────────────────────────────────────────────────────
    θ1 = np.arctan2(wy, wx) - np.arctan2(L2 * np.sin(θ2), L1 + L2 * np.cos(θ2))

    # ── Step 5: θ3 from the remaining orientation ─────────────────────────
    θ3 = phi - θ1 - θ2

    # ── Wrap angles to (-π, π] ────────────────────────────────────────────
    θ1 = (θ1 + np.pi) % (2 * np.pi) - np.pi
    θ2 = (θ2 + np.pi) % (2 * np.pi) - np.pi
    θ3 = (θ3 + np.pi) % (2 * np.pi) - np.pi
    angles = np.array([θ1, θ2, θ3])

    # ── Verification ──────────────────────────────────────────────────────
    fk  = arm.forward_kinematics(angles)
    err = float(np.linalg.norm(fk["end_effector"] - np.array([x, y])))

    return _ik_result(
        success=err < 1e-6,
        angles=angles,
        error=err,
        note=f"{'elbow-up' if elbow_up else 'elbow-down'} solution",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Jacobian Pseudo-inverse IK  (numerical, position-only)
# ─────────────────────────────────────────────────────────────────────────────

def ik_jacobian(
    arm: ThreeLinkArm,
    x: float,
    y: float,
    *,
    theta_init: np.ndarray | None = None,
    theta_preferred: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-4,
    step_size: float = 0.3,
    damping: float = 0.03,
    null_space_gain: float = 0.2,
) -> dict:
    """
    Iterative Jacobian pseudo-inverse IK (position-only target).

    Because the arm has 3 DOF and only 2 task-space variables (x, y),
    it is redundant. The null-space of the Jacobian is used to
    drive joints toward ``theta_preferred`` while reaching the target.

    Parameters
    ----------
    arm              : ThreeLinkArm instance.
    x, y             : Target end-effector position.
    theta_init       : Starting joint angles (default: arm.joint_angles).
    theta_preferred  : Preferred joint configuration for null-space
                       optimization (default: [0, 0, 0]).
    max_iter         : Maximum number of iterations.
    tol              : Convergence tolerance on position error (world units).
    step_size        : Fraction of Δp applied per iteration (0 < α ≤ 1).
    damping          : Levenberg-Marquardt damping λ for near-singular J.
    null_space_gain  : Gain α for null-space joint preference term.

    Returns
    -------
    dict with keys: success, angles, iterations, position_error, note.
    """
    target = np.array([x, y], dtype=float)

    if theta_init is None:
        theta = arm.joint_angles.copy()
    else:
        theta = np.array(theta_init, dtype=float)

    if theta_preferred is None:
        theta_preferred = np.zeros(3)
    else:
        theta_preferred = np.array(theta_preferred, dtype=float)

    for i in range(max_iter):
        fk  = arm.forward_kinematics(theta)
        err = target - fk["end_effector"]
        e   = float(np.linalg.norm(err))

        if e < tol:
            return _ik_result(True, theta, iterations=i, error=e,
                              note="converged")

        J  = arm.jacobian(theta)                          # (2, 3)
        # Damped least-squares pseudo-inverse:  J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹
        JJT = J @ J.T + damping**2 * np.eye(2)
        J_pinv = J.T @ np.linalg.inv(JJT)               # (3, 2)

        # Null-space projector  N = I - J⁺J
        N = np.eye(3) - J_pinv @ J                       # (3, 3)

        # Joint preference in null space
        null_term = null_space_gain * N @ (theta_preferred - theta)

        # Update
        dtheta = step_size * J_pinv @ err + null_term
        theta  = theta + dtheta

        # Wrap to (-π, π]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

    # Final error
    fk  = arm.forward_kinematics(theta)
    err = float(np.linalg.norm(target - fk["end_effector"]))
    success = err < tol * 10          # relaxed tolerance after max_iter

    return _ik_result(success, theta, iterations=max_iter, error=err,
                      note="max iterations reached" if not success else "converged (late)")
