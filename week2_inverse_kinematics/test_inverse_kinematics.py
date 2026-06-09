"""
Unit Tests — Week 3-4: Inverse Kinematics
==========================================
Run:
    python3 -m pytest test_inverse_kinematics.py -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1_forward_kinematics"))
from forward_kinematics import ThreeLinkArm
from inverse_kinematics import ik_analytical, ik_jacobian

ARM = ThreeLinkArm(link_lengths=(3.0, 2.5, 1.5))


def fk_ee(angles):
    """Helper: run FK and return EE position."""
    return ARM.forward_kinematics(angles)["end_effector"]


# ─── Analytical IK ────────────────────────────────────────────────────────────

class TestAnalyticalIK:

    def test_fully_extended(self):
        """EE at max reach along +X with φ=0."""
        res = ik_analytical(ARM, 7.0, 0.0, 0.0)
        assert res["success"]
        ee = fk_ee(res["angles"])
        assert np.allclose(ee, [7.0, 0.0], atol=1e-5)

    def test_arbitrary_target_elbow_up(self):
        res = ik_analytical(ARM, 5.0, 2.0, np.radians(30), elbow_up=True)
        assert res["success"]
        ee = fk_ee(res["angles"])
        assert np.allclose(ee, [5.0, 2.0], atol=1e-5)

    def test_arbitrary_target_elbow_down(self):
        res = ik_analytical(ARM, 5.0, 2.0, np.radians(30), elbow_up=False)
        assert res["success"]
        ee = fk_ee(res["angles"])
        assert np.allclose(ee, [5.0, 2.0], atol=1e-5)

    def test_elbow_up_and_down_give_different_angles(self):
        """Elbow-up and elbow-down are two distinct solutions."""
        r_up   = ik_analytical(ARM, 4.0, 3.0, np.radians(45), elbow_up=True)
        r_down = ik_analytical(ARM, 4.0, 3.0, np.radians(45), elbow_up=False)
        assert r_up["success"] and r_down["success"]
        assert not np.allclose(r_up["angles"], r_down["angles"], atol=1e-3)

    def test_orientation_preserved(self):
        """End-effector orientation must match the requested φ."""
        phi = np.radians(55)
        res = ik_analytical(ARM, 4.0, 3.0, phi, elbow_up=True)
        assert res["success"]
        phi_actual = ARM.forward_kinematics(res["angles"])["orientation"]
        assert abs((phi_actual - phi + np.pi) % (2*np.pi) - np.pi) < 1e-4

    def test_out_of_reach_fails(self):
        res = ik_analytical(ARM, 10.0, 0.0, 0.0)
        assert not res["success"]

    def test_negative_quadrant(self):
        res = ik_analytical(ARM, -4.0, -3.0, np.radians(-135))
        assert res["success"]
        ee = fk_ee(res["angles"])
        assert np.allclose(ee, [-4.0, -3.0], atol=1e-4)

    def test_near_base(self):
        """Target very close to base — arm should fold to reach it."""
        res = ik_analytical(ARM, 0.5, 0.0, 0.0)
        assert res["success"]
        ee = fk_ee(res["angles"])
        assert np.allclose(ee, [0.5, 0.0], atol=1e-4)

    def test_position_error_is_small(self):
        res = ik_analytical(ARM, 3.0, 4.0, np.radians(20))
        if res["success"]:
            assert res["position_error"] < 1e-5


# ─── Jacobian IK ──────────────────────────────────────────────────────────────

class TestJacobianIK:

    def test_converges_on_simple_target(self):
        res = ik_jacobian(ARM, 5.0, 2.0)
        assert res["success"]
        ee = fk_ee(res["angles"])
        assert np.allclose(ee, [5.0, 2.0], atol=1e-3)

    def test_converges_straight_up(self):
        res = ik_jacobian(ARM, 0.0, 6.0)
        assert res["success"]
        ee = fk_ee(res["angles"])
        assert np.allclose(ee, [0.0, 6.0], atol=1e-3)

    def test_near_base(self):
        res = ik_jacobian(ARM, 0.5, 0.5,
                          theta_init=np.radians([10, 5, -5]))
        assert res["success"]
        ee = fk_ee(res["angles"])
        assert np.linalg.norm(ee - np.array([0.5, 0.5])) < 1e-2

    def test_out_of_reach_fails(self):
        res = ik_jacobian(ARM, 10.0, 0.0, max_iter=100)
        assert not res["success"]

    def test_preferred_config_respected(self):
        """Null-space should bias toward preferred angles (zero config)."""
        res = ik_jacobian(ARM, 5.0, 0.0,
                          theta_preferred=np.zeros(3),
                          theta_init=np.radians([10, 5, -5]))
        assert res["success"]
        # θ2 should be close to zero (preferred), not wildly far
        assert abs(res["angles"][1]) < np.radians(90)

    def test_result_keys_present(self):
        res = ik_jacobian(ARM, 4.0, 3.0)
        for key in ("success", "angles", "iterations", "position_error", "note"):
            assert key in res
