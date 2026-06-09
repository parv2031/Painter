"""
Demo Script — Week 1 Forward Kinematics
========================================
Runs FK for a set of sample configurations and prints results to the terminal.
No GUI required. Good for verifying the math quickly.

Run:
    python demo.py
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from forward_kinematics import ThreeLinkArm


SEPARATOR = "─" * 60
LINK_LENGTHS = (3.0, 2.5, 1.5)


def print_fk(label: str, angles_deg: tuple, arm: ThreeLinkArm):
    angles_rad = np.radians(angles_deg)
    fk = arm.forward_kinematics(angles_rad)
    joints = fk["joint_positions"]
    ee = fk["end_effector"]
    phi = np.degrees(fk["orientation"])
    reach = np.linalg.norm(ee)

    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    print(f"  θ = [{angles_deg[0]}°, {angles_deg[1]}°, {angles_deg[2]}°]")
    print(SEPARATOR)
    for i, j in enumerate(joints):
        tag = "Origin" if i == 0 else (f"Joint {i}" if i < 3 else "End-Eff")
        print(f"  {tag:9s}  x = {j[0]:+8.4f}   y = {j[1]:+8.4f}")
    print(SEPARATOR)
    print(f"  Orientation  φ = {phi:+.2f}°")
    print(f"  Reach from origin = {reach:.4f}  (max = {arm.max_reach:.1f})")

    J = arm.jacobian(angles_rad)
    print(f"\n  Jacobian  J =")
    print(f"    [{J[0,0]:+7.3f}  {J[0,1]:+7.3f}  {J[0,2]:+7.3f}]  (∂x/∂θ)")
    print(f"    [{J[1,0]:+7.3f}  {J[1,1]:+7.3f}  {J[1,2]:+7.3f}]  (∂y/∂θ)")

    cond = np.linalg.cond(J)
    print(f"  Condition number of J = {cond:.2f}"
          + ("  ← near singularity!" if cond > 1e3 else ""))


def main():
    arm = ThreeLinkArm(link_lengths=LINK_LENGTHS)

    print("\n" + "█" * 60)
    print("  3-Link Planar Arm — Forward Kinematics Demo (Week 1)")
    print(f"  Link lengths: L1={LINK_LENGTHS[0]}  L2={LINK_LENGTHS[1]}  L3={LINK_LENGTHS[2]}")
    print("█" * 60)

    configs = [
        ("Zero configuration (fully extended along +X)",  (0,   0,   0 )),
        ("θ1=90° — arm pointing straight up",              (90,  0,   0 )),
        ("θ1=180° — arm pointing along −X",                (180, 0,   0 )),
        ("Folded back: θ1=0, θ2=180, θ3=0",               (0,   180, 0 )),
        ("Arbitrary: θ1=30, θ2=−45, θ3=60",               (30, -45, 60 )),
        ("Near singularity: θ2=θ3=0 fully extended",       (45,  0,   0 )),
    ]

    for label, angles in configs:
        print_fk(label, angles, arm)

    print(f"\n{'═'*60}")
    print("  Workspace summary")
    print(SEPARATOR)
    print(f"  Max reach : {arm.max_reach:.2f} units")
    print(f"  Min reach : {arm.min_reach:.2f} units")
    print(f"  Reachable (5, 0) ? {arm.is_reachable(np.array([5, 0]))}")
    print(f"  Reachable (9, 0) ? {arm.is_reachable(np.array([9, 0]))}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
