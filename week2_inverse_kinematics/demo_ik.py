"""
Demo — Week 3-4 Inverse Kinematics
====================================
Runs both IK solvers on a set of sample targets and prints results.
No display needed.

Run:
    python3 demo_ik.py
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1_forward_kinematics"))
from forward_kinematics import ThreeLinkArm
from inverse_kinematics import ik_analytical, ik_jacobian

SEP  = "─" * 65
DSEP = "═" * 65


def print_result(label, target, phi_deg, result, arm):
    ok = "✓" if result["success"] else "✗"
    θ  = np.degrees(result["angles"])
    fk = arm.forward_kinematics(result["angles"])
    ee = fk["end_effector"]

    print(f"\n{DSEP}")
    print(f"  {ok}  {label}")
    print(SEP)
    print(f"  Target     : ({target[0]:.2f}, {target[1]:.2f})"
          + (f"   φ = {phi_deg:.0f}°" if phi_deg is not None else "  (no φ constraint)"))
    print(f"  Solved θ   : [{θ[0]:+.2f}°, {θ[1]:+.2f}°, {θ[2]:+.2f}°]")
    print(f"  EE actual  : ({ee[0]:.4f}, {ee[1]:.4f})")
    print(f"  Pos error  : {result['position_error']:.2e}  |  {result['note']}")
    if "iterations" in result and result["iterations"]:
        print(f"  Iterations : {result['iterations']}")


def main():
    arm = ThreeLinkArm(link_lengths=(3.0, 2.5, 1.5))

    print(f"\n{'█'*65}")
    print("  3-Link Arm — IK Demo (Week 3-4)")
    print(f"  Links: L1=3.0  L2=2.5  L3=1.5   Max reach={arm.max_reach}")
    print(f"{'█'*65}")

    # ── Analytical tests ─────────────────────────────────────────────────
    print("\n\n  ══ ANALYTICAL IK (closed-form) ══")
    cases_analytical = [
        ((5.0,  2.0), 30,  True,  "Arbitrary target, φ=30°, elbow-up"),
        ((5.0,  2.0), 30,  False, "Arbitrary target, φ=30°, elbow-down"),
        ((7.0,  0.0),  0,  True,  "Fully extended along +X, φ=0°"),
        ((0.0,  6.0), 90, True,  "Straight up, φ=90°"),
        ((-3.0, 4.0), 120, True, "Second quadrant target, φ=120°"),
        ((0.5,  0.0),  0,  True,  "Near-base target, φ=0°"),
        ((9.0,  0.0),  0,  True,  "Out-of-reach target"),
    ]

    for (x, y), phi_deg, elbow_up, label in cases_analytical:
        phi = np.radians(phi_deg)
        res = ik_analytical(arm, x, y, phi, elbow_up=elbow_up)
        print_result(label, (x, y), phi_deg, res, arm)

    # ── Jacobian tests ────────────────────────────────────────────────────
    print("\n\n  ══ JACOBIAN PSEUDO-INVERSE IK (numerical, position-only) ══")
    cases_jacobian = [
        ((5.0,  2.0), "Arbitrary target — no φ constraint"),
        ((0.0,  6.0), "Straight up"),
        ((-4.0, 3.0), "Second quadrant"),
        ((0.5,  0.5), "Near-base target"),
        ((6.8,  0.0), "Near max reach"),
    ]

    for (x, y), label in cases_jacobian:
        res = ik_jacobian(arm, x, y,
                          theta_init=np.radians([15, 10, -5]),
                          theta_preferred=np.zeros(3))
        print_result(label, (x, y), None, res, arm)

    print(f"\n{DSEP}\n")


if __name__ == "__main__":
    main()
