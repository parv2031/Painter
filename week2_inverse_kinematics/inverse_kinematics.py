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
# Analytical IK with automatic φ selection
# ─────────────────────────────────────────────────────────────────────────────

def ik_analytical_auto(
    arm: ThreeLinkArm,
    x: float,
    y: float,
    elbow_up: bool = True,
) -> dict:
    """
    Closed-form IK that automatically finds a valid end-effector orientation
    φ so that ANY point within the workspace disc (radius ≤ L1+L2+L3) can
    be reached — no user-supplied φ needed.

    Mathematical basis
    ------------------
    For a given target (x, y) at distance r = √(x²+y²) and angle α = atan2(y,x),
    the wrist position after subtracting link-3 is:

        wx = x − L3·cos(φ),   wy = y − L3·sin(φ)

    The squared wrist-to-base distance is:

        d_w² = r² + L3² − 2·L3·r·cos(φ − α)          ... (★)

    For the 2-link sub-arm (L1, L2) to reach the wrist, we need:

        d_min² ≤ d_w² ≤ d_max²
        where  d_min = |L1−L2|,  d_max = L1+L2

    Rearranging (★) gives the required range of cos(φ − α):

        c_lo = (r² + L3² − d_max²) / (2·L3·r)
        c_hi = (r² + L3² − d_min²) / (2·L3·r)

    We pick the midpoint c* = (c_lo + c_hi) / 2  which places the wrist at
    the best-conditioned distance, then:

        φ = α ± arccos(c*)

    Both signs are tried; the first that produces a valid solution is returned.

    Strategy (in order of preference)
    ----------------------------------
    1. Natural orientation:  φ = α  (arm points radially outward — smooth,
       works for most of the workspace where |r − L3| ∈ [d_min, d_max]).
    2. Computed optimal φ from the midpoint of the valid cosine range.
    3. Fallback scan over 36 uniformly spaced φ ∈ [0, 2π).

    Parameters
    ----------
    arm      : ThreeLinkArm instance.
    x, y     : Target end-effector position.
    elbow_up : Elbow-up (True) or elbow-down (False).

    Returns
    -------
    dict with keys: success, angles, position_error, phi_chosen, note.
    """
    L1, L2, L3 = arm.link_lengths
    r     = np.hypot(x, y)
    alpha = np.arctan2(y, x)

    d_min = abs(L1 - L2)      # min reachable wrist distance  = 0.5
    d_max = L1 + L2            # max reachable wrist distance  = 5.5

    # ── Global reachability ───────────────────────────────────────────────
    if r > L1 + L2 + L3 + 1e-9:
        return _ik_result(False, [0, 0, 0], note="Target out of reach (too far)")

    def _try(phi_candidate):
        res = ik_analytical(arm, x, y, phi_candidate, elbow_up=elbow_up)
        if res["success"]:
            res["phi_chosen"] = phi_candidate
        return res

    # ── Strategy 1: natural (radial) orientation ──────────────────────────
    # φ = α  →  d_w = |r − L3|
    res = _try(alpha)
    if res["success"]:
        res["note"] = f"auto-φ natural α={np.degrees(alpha):.1f}°"
        return res

    # ── Strategy 2: analytically compute the optimal φ ───────────────────
    # Valid cosine range for (φ − α):
    if r < 1e-9:
        # At origin: any φ with d_w = L3 works if L3 ∈ [d_min, d_max]
        phi_candidates = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    else:
        c_lo = (r**2 + L3**2 - d_max**2) / (2 * L3 * r)
        c_hi = (r**2 + L3**2 - d_min**2) / (2 * L3 * r)
        c_lo = np.clip(c_lo, -1.0, 1.0)
        c_hi = np.clip(c_hi, -1.0, 1.0)

        if c_lo > c_hi + 1e-9:
            return _ik_result(False, [0, 0, 0], note="No valid φ exists for this target")

        # Midpoint of valid cosine range → best-conditioned wrist distance
        c_mid  = (c_lo + c_hi) / 2.0
        delta  = np.arccos(np.clip(c_mid, -1.0, 1.0))
        phi_candidates = [
            alpha + delta,
            alpha - delta,
            alpha + np.pi,      # point link3 toward base
            alpha - np.pi,
        ]

    for phi_cand in phi_candidates:
        res = _try(phi_cand)
        if res["success"]:
            res["note"] = f"auto-φ computed={np.degrees(phi_cand):.1f}°"
            return res

    # ── Strategy 3: brute-force scan ─────────────────────────────────────
    for phi_scan in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        res = _try(phi_scan)
        if res["success"]:
            res["note"] = f"auto-φ scan={np.degrees(phi_scan):.1f}°"
            return res

    return _ik_result(False, [0, 0, 0], note="No valid configuration found after full scan")


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
