"""
Unit Tests — Week 1: Forward Kinematics
========================================
Run with:
    python -m pytest test_forward_kinematics.py -v
or:
    python test_forward_kinematics.py
"""

import math
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from forward_kinematics import ThreeLinkArm


class TestForwardKinematics:
    """Test FK correctness for known configurations."""

    def setup_method(self):
        self.arm = ThreeLinkArm(
            link_lengths=(3.0, 2.5, 1.5),
            joint_angles=(0.0, 0.0, 0.0),
        )

    # ── Fully extended ────────────────────────────────────────────────────
    def test_zero_config_end_effector(self):
        """All zero angles → arm fully extended along +X."""
        fk = self.arm.forward_kinematics()
        expected_x = sum(self.arm.link_lengths)   # 7.0
        assert np.isclose(fk["end_effector"][0], expected_x, atol=1e-9)
        assert np.isclose(fk["end_effector"][1], 0.0,        atol=1e-9)

    def test_zero_config_joints(self):
        fk = self.arm.forward_kinematics()
        joints = fk["joint_positions"]
        assert np.allclose(joints[0], [0.0, 0.0])
        assert np.allclose(joints[1], [3.0, 0.0])
        assert np.allclose(joints[2], [5.5, 0.0])
        assert np.allclose(joints[3], [7.0, 0.0])

    # ── Single-joint tests ───────────────────────────────────────────────
    def test_theta1_90(self):
        """θ1=90°, θ2=θ3=0 → arm points straight up."""
        angles = np.radians([90, 0, 0])
        fk = self.arm.forward_kinematics(angles)
        ee = fk["end_effector"]
        assert np.isclose(ee[0], 0.0, atol=1e-9)
        assert np.isclose(ee[1], 7.0, atol=1e-9)

    def test_theta1_180(self):
        """θ1=180° → arm points along -X."""
        angles = np.radians([180, 0, 0])
        fk = self.arm.forward_kinematics(angles)
        ee = fk["end_effector"]
        assert np.isclose(ee[0], -7.0, atol=1e-9)
        assert np.isclose(ee[1],  0.0, atol=1e-9)

    # ── Folded-back configuration ────────────────────────────────────────
    def test_folded_config(self):
        """θ1=0, θ2=180, θ3=0 → link2 folds back over link1.

        J1 = (3, 0).  α2 = 180° so J2 = (3-2.5, 0) = (0.5, 0).
        α3 = 180° so EE = (0.5 - 1.5, 0) = (-1.0, 0).
        """
        angles = np.radians([0, 180, 0])
        fk = self.arm.forward_kinematics(angles)
        ee = fk["end_effector"]
        assert np.isclose(ee[0], -1.0, atol=1e-9)
        assert np.isclose(ee[1],  0.0, atol=1e-9)

    # ── Orientation ───────────────────────────────────────────────────────
    def test_orientation(self):
        """φ = θ1+θ2+θ3 for any configuration."""
        angles = np.radians([30, -45, 60])
        fk = self.arm.forward_kinematics(angles)
        expected_phi = math.radians(30 - 45 + 60)  # 45°
        assert np.isclose(fk["orientation"], expected_phi, atol=1e-9)

    # ── Jacobian shape ────────────────────────────────────────────────────
    def test_jacobian_shape(self):
        J = self.arm.jacobian()
        assert J.shape == (2, 3)

    def test_jacobian_at_zero(self):
        """At zero config, Jxy has known analytical values."""
        J = self.arm.jacobian(np.array([0.0, 0.0, 0.0]))
        # ∂x/∂θi = -l_i*sin(0) - ... = 0
        assert np.allclose(J[0, :], [0.0, 0.0, 0.0], atol=1e-9)
        # ∂y/∂θ1 = l1+l2+l3 = 7.0, ∂y/∂θ2 = l2+l3 = 4.0, ∂y/∂θ3 = l3 = 1.5
        assert np.allclose(J[1, :], [7.0, 4.0, 1.5], atol=1e-9)

    # ── Workspace ─────────────────────────────────────────────────────────
    def test_max_reach(self):
        assert np.isclose(self.arm.max_reach, 7.0)

    def test_min_reach_is_zero(self):
        """
        L2 + L3 = 4.0 >= L1 = 3.0, so the arm can reach the origin.
        min_reach = max(0, L1 - L2 - L3) = max(0, -1) = 0.
        """
        assert self.arm.min_reach == 0.0

    def test_reachable_origin(self):
        """The origin itself is reachable."""
        assert self.arm.is_reachable(np.array([0.0, 0.0]))

    def test_reachable_point_in_workspace(self):
        assert self.arm.is_reachable(np.array([5.0, 0.0]))

    def test_unreachable_point_beyond_max(self):
        assert not self.arm.is_reachable(np.array([8.0, 0.0]))

    # ── Angle clamping ────────────────────────────────────────────────────
    def test_angle_clamping(self):
        arm = ThreeLinkArm(joint_limits=[(-np.pi/2, np.pi/2)] * 3)
        arm.set_joint_angles(np.array([5.0, -5.0, 3.0]))
        assert arm.joint_angles[0] == pytest.approx(np.pi / 2)
        assert arm.joint_angles[1] == pytest.approx(-np.pi / 2)
        assert arm.joint_angles[2] == pytest.approx(np.pi / 2)


# ── Run as script ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        # Minimal manual runner
        t = TestForwardKinematics()
        passed, failed = 0, []
        tests = [m for m in dir(t) if m.startswith("test_")]
        for name in tests:
            t.setup_method()
            try:
                getattr(t, name)()
                print(f"  ✓  {name}")
                passed += 1
            except Exception as exc:
                print(f"  ✗  {name}  →  {exc}")
                failed.append(name)
        print(f"\n{passed}/{passed+len(failed)} tests passed.")
        if failed:
            sys.exit(1)
