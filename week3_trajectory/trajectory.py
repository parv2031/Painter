"""
Week 3 — Trajectory Planning: Cartesian Linear Interpolation
=============================================================
Project: Bob Ross without ROS — Robotics Society Summer Project
Mentors: Anjaneya and Parv

This module generates straight-line end-effector trajectories by:
  1. Interpolating dense waypoints along P1 → P2 in task space.
  2. Solving IK at each waypoint (auto-φ, nearest-configuration).
  3. Computing the angular velocity of each servo joint.

Key concept — Configuration Continuity
    IK has multiple solutions (elbow-up / elbow-down). Solving
    each waypoint independently can cause the arm to suddenly flip
    its elbow mid-motion.  We prevent this by always picking the
    IK solution closest in joint space to the *previous* configuration.

Key concept — Servo Velocity
    For a servo motor, the commanded quantity is angular velocity ω.
    Given consecutive joint angles θ[k] and θ[k-1]:

        ω_i[k] = Δθ_i / Δt
                = shortest_angle_diff(θ_i[k], θ_i[k-1]) / Δt

    The "shortest" difference wraps correctly around ±π, so a servo
    that moves from 170° to -170° sees Δθ = +20° (not −340°).
"""

import numpy as np
import sys, os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1_forward_kinematics"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week2_inverse_kinematics"))

from forward_kinematics import ThreeLinkArm
from inverse_kinematics import ik_analytical_auto


# ─── Helpers ─────────────────────────────────────────────────────────────────

def angle_diff(a: float, b: float) -> float:
    """
    Shortest signed difference (a − b), wrapped to (−π, π].

    Examples
    --------
    angle_diff(np.radians(170), np.radians(-170)) →  +0.349 rad (+20°)
    angle_diff(np.radians(-170), np.radians(170)) →  −0.349 rad (−20°)
    """
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


def angle_diff_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise angle_diff for vectors."""
    return np.array([angle_diff(ai, bi) for ai, bi in zip(a, b)])


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class TrajectoryResult:
    """
    All data produced by planning a straight-line trajectory.

    Attributes
    ----------
    waypoints        : (N, 2) — EE positions along the line
    joint_angles     : (N, 3) — joint angles at each waypoint [rad]
    joint_velocities : (N-1, 3) — servo angular velocities [rad/s]
    dt               : time step between consecutive waypoints [s]
    success_mask     : (N,) bool — True where IK succeeded
    ee_speed         : scalar end-effector speed along the line [units/s]
    n_steps          : number of waypoints
    total_time       : total motion duration [s]
    """
    waypoints        : np.ndarray
    joint_angles     : np.ndarray
    joint_velocities : np.ndarray
    dt               : float
    success_mask     : np.ndarray
    ee_speed         : float
    n_steps          : int
    total_time       : float


# ─── Core functions ───────────────────────────────────────────────────────────

def linear_interpolate(p1: np.ndarray, p2: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Generate n_steps waypoints along the straight line from p1 to p2.

    Parameters
    ----------
    p1, p2  : Start and end positions, shape (2,).
    n_steps : Number of points including both endpoints.

    Returns
    -------
    np.ndarray of shape (n_steps, 2).

    Notes
    -----
    The parameterisation is uniform in arc-length (i.e. constant EE speed).
    Waypoints are spaced by distance(p1,p2)/(n_steps−1) in world units.
    """
    t = np.linspace(0.0, 1.0, n_steps)          # shape (N,)
    return np.outer(1 - t, p1) + np.outer(t, p2)   # shape (N, 2)


def _valid_phi_range(L1: float, L2: float, L3: float,
                     x: float, y: float) -> tuple[float, float, float] | None:
    """
    Compute the valid range of φ offsets (δ = φ − α) so that the wrist
    placed at (x − L3·cos φ, y − L3·sin φ) is reachable by the 2-link
    sub-arm (L1, L2).

    Returns (alpha, delta_min, delta_max) where:
        alpha     = atan2(y, x)
        delta_min = minimum |φ − α| (0 if natural phi works)
        delta_max = maximum |φ − α|

    Valid φ values satisfy |φ − α| ∈ [delta_min, delta_max].
    Returns None if no valid φ exists.
    """
    r = np.hypot(x, y)
    alpha = np.arctan2(y, x)

    d_min = abs(L1 - L2)
    d_max = L1 + L2

    if r < 1e-9:
        # At origin, any φ with d_w = L3 is valid if L3 ∈ [d_min, d_max]
        if d_min <= L3 <= d_max:
            return alpha, 0.0, np.pi
        return None

    # cos(δ) must be in [c_lo, c_hi]
    c_lo = (r**2 + L3**2 - d_max**2) / (2.0 * L3 * r)
    c_hi = (r**2 + L3**2 - d_min**2) / (2.0 * L3 * r)
    c_lo = np.clip(c_lo, -1.0, 1.0)
    c_hi = np.clip(c_hi, -1.0, 1.0)

    if c_lo > c_hi + 1e-9:
        return None  # target unreachable

    # delta_min = arccos(c_hi), delta_max = arccos(c_lo)
    delta_min = np.arccos(c_hi)   # could be 0 when c_hi ≥ 1
    delta_max = np.arccos(c_lo)

    return alpha, delta_min, delta_max


def _clamp_phi_to_valid(phi_desired: float,
                        alpha: float, delta_min: float, delta_max: float) -> float:
    """
    Return the φ value closest to ``phi_desired`` that lies in the valid range.

    Valid zone for (φ − α):  |δ| ∈ [delta_min, delta_max]
    Forbidden zone:           |δ| < delta_min  (wrist too close to origin)
    """
    delta = float((phi_desired - alpha + np.pi) % (2 * np.pi) - np.pi)  # wrap

    abs_d = abs(delta)
    sign  = 1.0 if delta >= 0 else -1.0

    if abs_d < delta_min:
        # Inside forbidden zone → snap to nearest boundary
        delta_clamped = sign * delta_min
    elif abs_d > delta_max:
        # Outside valid zone → snap to far boundary
        delta_clamped = sign * delta_max
    else:
        delta_clamped = delta  # already valid

    return float(alpha + delta_clamped)


def _nearest_ik(arm: ThreeLinkArm, x: float, y: float,
                prev_angles: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """
    Solve IK at (x, y) with strict φ-continuity to prevent elbow flips.

    Algorithm
    ---------
    1. Extract the current EE orientation from prev_angles:
           φ_prev = θ1 + θ2 + θ3

    2. Compute the valid φ range at (x, y) analytically using the
       wrist-reachability constraint (see _valid_phi_range).

    3. Find φ* = the φ closest to φ_prev within the valid range.
       This is the key step — it ensures φ evolves smoothly, eliminating
       the discontinuous jumps that cause elbow flips.

    4. Try ik_analytical(arm, x, y, φ*, elbow_up) for both elbow modes.
       Pick the solution minimising total joint displacement.

    5. If step 4 fails (numerical edge case), fall back to a dense scan
       of 72 uniformly spaced φ values, again minimising displacement.

    Returns
    -------
    (angles, success)
    """
    from inverse_kinematics import ik_analytical

    L1, L2, L3 = arm.link_lengths

    # ── Step 1: previous EE orientation ──────────────────────────────────
    prev_phi = float(np.sum(prev_angles))   # φ = θ1 + θ2 + θ3

    # ── Step 2: valid φ range ─────────────────────────────────────────────
    vr = _valid_phi_range(L1, L2, L3, x, y)
    if vr is None:
        return None, False   # target truly out of reach

    alpha, delta_min, delta_max = vr

    # ── Step 3: clamp prev_phi to valid range ─────────────────────────────
    phi_star = _clamp_phi_to_valid(prev_phi, alpha, delta_min, delta_max)

    # ── Step 4: solve IK at phi_star, both elbow modes ────────────────────
    best_angles = None
    best_cost   = np.inf

    for phi_try in (phi_star, phi_star + 1e-4):          # tiny nudge for edge cases
        for elbow_up in (True, False):
            res = ik_analytical(arm, x, y, phi_try, elbow_up=elbow_up)
            if res["success"]:
                cost = float(np.sum(np.abs(angle_diff_vec(res["angles"], prev_angles))))
                if cost < best_cost:
                    best_cost   = cost
                    best_angles = res["angles"].copy()

    if best_angles is not None:
        return best_angles, True

    # ── Step 5: dense φ scan fallback ────────────────────────────────────
    # Scan 72 φ values, all within the valid range, starting near prev_phi
    for delta_sign in (1.0, -1.0):
        for delta_abs in np.linspace(delta_min, delta_max, 36):
            phi_cand = alpha + delta_sign * delta_abs
            for elbow_up in (True, False):
                res = ik_analytical(arm, x, y, phi_cand, elbow_up=elbow_up)
                if res["success"]:
                    cost = float(np.sum(np.abs(angle_diff_vec(res["angles"], prev_angles))))
                    if cost < best_cost:
                        best_cost   = cost
                        best_angles = res["angles"].copy()

    return (best_angles, best_angles is not None)



def plan_linear_trajectory(
    arm: ThreeLinkArm,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    ee_speed: float = 2.0,
    n_steps: int | None = None,
    min_steps: int = 50,
    max_steps: int = 500,
    init_angles: np.ndarray | None = None,
) -> TrajectoryResult:
    """
    Plan a straight-line Cartesian trajectory from p1 to p2.

    The end-effector travels at constant speed ``ee_speed`` (world units / s).
    IK is solved at each waypoint; configuration continuity is enforced by
    always selecting the elbow pose closest to the previous configuration.

    Parameters
    ----------
    arm        : ThreeLinkArm instance.
    p1, p2     : Start and end EE positions, shape (2,).
    ee_speed   : Desired EE speed along the line [world units / s].
    n_steps    : Override number of waypoints (ignores ee_speed if set).
    min_steps  : Minimum number of waypoints.
    max_steps  : Maximum number of waypoints.
    init_angles: Initial joint configuration. Defaults to arm.joint_angles.

    Returns
    -------
    TrajectoryResult
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    dist = float(np.linalg.norm(p2 - p1))
    if dist < 1e-9:
        raise ValueError("p1 and p2 are the same point.")

    # ── Determine step count ─────────────────────────────────────────────
    if n_steps is None:
        # Choose steps so that EE moves ~0.05 units per step at ee_speed
        dt_target = 0.02          # 20 ms per step (50 Hz)
        n_steps = max(min_steps, min(max_steps, int(dist / (ee_speed * dt_target))))

    total_time = dist / ee_speed
    dt         = total_time / (n_steps - 1)

    # ── Waypoints ─────────────────────────────────────────────────────────
    waypoints = linear_interpolate(p1, p2, n_steps)   # (N, 2)

    # ── IK along trajectory ───────────────────────────────────────────────
    joint_angles  = np.zeros((n_steps, 3))
    success_mask  = np.zeros(n_steps, dtype=bool)

    if init_angles is None:
        prev_angles = arm.joint_angles.copy()
    else:
        prev_angles = np.array(init_angles, dtype=float)

    # ── Bootstrap: solve IK at P1 with the geometrically natural φ ────────
    # Using ik_analytical_auto (not _nearest_ik) here because there is no
    # "previous" configuration to be continuous from — we just want the
    # best natural starting pose at P1.  From that pose, phi-continuity
    # takes over for the rest of the trajectory.
    from inverse_kinematics import ik_analytical_auto as _auto
    boot_res = _auto(arm, float(waypoints[0, 0]), float(waypoints[0, 1]),
                     elbow_up=True)
    if not boot_res["success"]:
        boot_res = _auto(arm, float(waypoints[0, 0]), float(waypoints[0, 1]),
                         elbow_up=False)
    if boot_res["success"]:
        prev_angles = boot_res["angles"]

    for k, (x, y) in enumerate(waypoints):
        angles, ok = _nearest_ik(arm, x, y, prev_angles)
        if ok:
            joint_angles[k] = angles
            prev_angles      = angles
            success_mask[k]  = True
        else:
            # Fallback: hold previous configuration
            joint_angles[k]  = prev_angles
            success_mask[k]  = False

    # ── Servo velocities (rad/s) ──────────────────────────────────────────
    # ω[k] = (θ[k] − θ[k-1]) / dt   using shortest angular difference
    diffs = np.array([
        angle_diff_vec(joint_angles[k], joint_angles[k - 1])
        for k in range(1, n_steps)
    ])                                                  # (N-1, 3)
    joint_velocities = diffs / dt                       # rad/s

    return TrajectoryResult(
        waypoints        = waypoints,
        joint_angles     = joint_angles,
        joint_velocities = joint_velocities,
        dt               = dt,
        success_mask     = success_mask,
        ee_speed         = ee_speed,
        n_steps          = n_steps,
        total_time       = total_time,
    )
