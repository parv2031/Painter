"""
Demo — Week 3 Trajectory Planning
===================================
Plans a straight-line trajectory between several point pairs and prints
the full joint-angle and servo-velocity data to the terminal.

Run:
    python3 demo_trajectory.py
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1_forward_kinematics"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week2_inverse_kinematics"))

from forward_kinematics import ThreeLinkArm
from trajectory import plan_linear_trajectory

SEP  = "─" * 70
DSEP = "═" * 70

arm = ThreeLinkArm(link_lengths=(3.0, 2.5, 1.5))


def print_trajectory(label, p1, p2, speed=2.0):
    print(f"\n{DSEP}")
    print(f"  Trajectory: {label}")
    print(f"  P1 = ({p1[0]:.2f}, {p1[1]:.2f})   P2 = ({p2[0]:.2f}, {p2[1]:.2f})")
    print(f"  EE speed = {speed:.1f} units/s")
    print(DSEP)

    traj = plan_linear_trajectory(arm, p1, p2, ee_speed=speed)

    dist      = float(np.linalg.norm(p2 - p1))
    ok_frac   = traj.success_mask.mean() * 100
    vel_deg   = np.degrees(traj.joint_velocities)
    max_vel   = np.abs(vel_deg).max(axis=0)

    print(f"  Distance       : {dist:.3f} units")
    print(f"  Waypoints      : {traj.n_steps}")
    print(f"  Δt per step    : {traj.dt*1000:.1f} ms")
    print(f"  Total time     : {traj.total_time:.3f} s")
    print(f"  IK success     : {ok_frac:.1f}%")
    print(f"  Peak |ω|       : [{max_vel[0]:.1f}, {max_vel[1]:.1f}, {max_vel[2]:.1f}] °/s")
    print()

    # Print every ~10th step
    n_print = min(15, traj.n_steps)
    step_skip = max(1, traj.n_steps // n_print)

    header = (f"  {'k':>4}  {'t(s)':>6}  "
              f"{'θ1(°)':>7}  {'θ2(°)':>7}  {'θ3(°)':>7}  "
              f"{'ω1(°/s)':>8}  {'ω2(°/s)':>8}  {'ω3(°/s)':>8}  "
              f"{'EE_x':>7}  {'EE_y':>7}")
    print(header)
    print("  " + SEP)

    for k in range(0, traj.n_steps, step_skip):
        t     = k * traj.dt
        θ_deg = np.degrees(traj.joint_angles[k])
        ee    = arm.forward_kinematics(traj.joint_angles[k])["end_effector"]
        if k > 0:
            ω_deg = np.degrees(traj.joint_velocities[k - 1])
        else:
            ω_deg = np.zeros(3)
        ok = "✓" if traj.success_mask[k] else "✗"
        print(f"  {k:>4}  {t:>6.3f}  "
              f"{θ_deg[0]:>+7.2f}  {θ_deg[1]:>+7.2f}  {θ_deg[2]:>+7.2f}  "
              f"{ω_deg[0]:>+8.2f}  {ω_deg[1]:>+8.2f}  {ω_deg[2]:>+8.2f}  "
              f"{ee[0]:>7.4f}  {ee[1]:>7.4f}  {ok}")


def main():
    print(f"\n{'█'*70}")
    print("  Week 3 — Linear Trajectory Planner Demo")
    print(f"  Links: L1=3.0  L2=2.5  L3=1.5   Max reach={arm.max_reach}")
    print(f"{'█'*70}")

    test_cases = [
        ("Horizontal stroke",           np.array([2.0, 3.0]),  np.array([5.0, 3.0]), 2.0),
        ("Vertical stroke",             np.array([3.0, 1.0]),  np.array([3.0, 5.0]), 2.0),
        ("Diagonal stroke",             np.array([1.0, 1.0]),  np.array([5.0, 5.0]), 3.0),
        ("Near-base to far (stress)",   np.array([0.5, 0.5]),  np.array([5.0, 2.0]), 2.0),
        ("Cross-quadrant stroke",       np.array([-3.0, 2.0]), np.array([3.0, -2.0]), 2.5),
        ("Fast stroke (high ω)",        np.array([2.0, 2.0]),  np.array([5.0, 4.0]), 6.0),
    ]

    for label, p1, p2, speed in test_cases:
        print_trajectory(label, p1, p2, speed)

    print(f"\n{DSEP}\n")


if __name__ == "__main__":
    main()
