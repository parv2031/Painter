"""
Unit Tests — Week 3: Trajectory Planning
==========================================
Run:
    python3 -m pytest test_trajectory.py -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1_forward_kinematics"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week2_inverse_kinematics"))

from forward_kinematics import ThreeLinkArm
from trajectory import (
    linear_interpolate, plan_linear_trajectory, angle_diff, angle_diff_vec
)

ARM = ThreeLinkArm(link_lengths=(3.0, 2.5, 1.5))


# ─── angle_diff ───────────────────────────────────────────────────────────────

class TestAngleDiff:
    def test_no_wrap(self):
        assert np.isclose(angle_diff(0.5, 0.2), 0.3, atol=1e-9)

    def test_wrap_positive(self):
        # -170° to 170°: shortest path is +20° (CCW)
        a = np.radians(-170)
        b = np.radians(170)
        assert np.isclose(angle_diff(a, b), np.radians(20), atol=1e-9)

    def test_wrap_negative(self):
        # 170° to -170°: shortest path is -20° (CW)
        a = np.radians(170)
        b = np.radians(-170)
        assert np.isclose(angle_diff(a, b), np.radians(-20), atol=1e-9)

    def test_zero(self):
        assert angle_diff(1.0, 1.0) == pytest.approx(0.0)

    def test_half_circle(self):
        # π difference — result should be ±π
        assert abs(angle_diff(np.pi, 0.0)) == pytest.approx(np.pi)


# ─── linear_interpolate ───────────────────────────────────────────────────────

class TestLinearInterpolate:
    def test_shape(self):
        pts = linear_interpolate(np.array([0, 0]), np.array([1, 1]), 10)
        assert pts.shape == (10, 2)

    def test_endpoints(self):
        p1 = np.array([1.0, 2.0])
        p2 = np.array([4.0, 6.0])
        pts = linear_interpolate(p1, p2, 5)
        assert np.allclose(pts[0], p1)
        assert np.allclose(pts[-1], p2)

    def test_midpoint(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([4.0, 0.0])
        pts = linear_interpolate(p1, p2, 5)
        assert np.allclose(pts[2], [2.0, 0.0])

    def test_collinear(self):
        """All points must lie on the line segment."""
        p1 = np.array([1.0, 1.0])
        p2 = np.array([4.0, 5.0])
        pts = linear_interpolate(p1, p2, 20)
        direction = p2 - p1
        direction /= np.linalg.norm(direction)
        for pt in pts:
            v = pt - p1
            cross = v[0]*direction[1] - v[1]*direction[0]
            assert abs(cross) < 1e-10, f"Point {pt} not on line"

    def test_equal_spacing(self):
        """Consecutive waypoints must be equally spaced."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])   # dist = 5
        pts = linear_interpolate(p1, p2, 11)
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert np.allclose(dists, dists[0], atol=1e-9)


# ─── plan_linear_trajectory ──────────────────────────────────────────────────

class TestPlanLinearTrajectory:
    def _plan(self, p1, p2, speed=2.0, n_steps=None):
        return plan_linear_trajectory(ARM, np.array(p1), np.array(p2),
                                      ee_speed=speed, n_steps=n_steps)

    def test_shapes(self):
        traj = self._plan([2.0, 1.0], [5.0, 3.0])
        N = traj.n_steps
        assert traj.waypoints.shape        == (N, 2)
        assert traj.joint_angles.shape     == (N, 3)
        assert traj.joint_velocities.shape == (N - 1, 3)
        assert traj.success_mask.shape     == (N,)

    def test_endpoint_waypoints(self):
        p1, p2 = np.array([2.0, 1.0]), np.array([5.0, 3.0])
        traj = self._plan(p1, p2)
        assert np.allclose(traj.waypoints[0], p1)
        assert np.allclose(traj.waypoints[-1], p2)

    def test_ik_success_on_simple_stroke(self):
        """All waypoints on a simple mid-workspace stroke must be reachable."""
        traj = self._plan([2.0, 2.0], [5.0, 2.0])
        assert traj.success_mask.all(), \
            f"IK failed at {np.where(~traj.success_mask)[0]} waypoints"

    def test_ee_tracks_straight_line(self):
        """FK at each solved joint angle must be close to the straight-line waypoint."""
        traj = self._plan([2.0, 1.0], [5.0, 3.0], n_steps=50)
        for k in range(traj.n_steps):
            if traj.success_mask[k]:
                ee = ARM.forward_kinematics(traj.joint_angles[k])["end_effector"]
                assert np.linalg.norm(ee - traj.waypoints[k]) < 1e-4, \
                    f"Step {k}: EE={ee} vs waypoint={traj.waypoints[k]}"

    def test_velocity_finite(self):
        """All servo velocities must be finite numbers."""
        traj = self._plan([3.0, 0.0], [3.0, 4.0])
        assert np.isfinite(traj.joint_velocities).all()

    def test_velocity_correct_formula(self):
        """ω = Δθ/Δt using shortest angle difference."""
        traj = self._plan([2.0, 2.0], [5.0, 2.0], n_steps=20)
        for k in range(1, traj.n_steps):
            expected = (angle_diff_vec(traj.joint_angles[k], traj.joint_angles[k-1])
                        / traj.dt)
            assert np.allclose(traj.joint_velocities[k-1], expected, atol=1e-9)

    def test_total_time(self):
        p1, p2 = np.array([1.0, 0.0]), np.array([4.0, 0.0])
        speed  = 3.0
        traj   = self._plan(p1, p2, speed=speed)
        dist   = np.linalg.norm(p2 - p1)           # = 3.0
        assert np.isclose(traj.total_time, dist / speed, rtol=1e-3)

    def test_same_point_raises(self):
        with pytest.raises(ValueError):
            self._plan([3.0, 0.0], [3.0, 0.0])

    def test_configuration_continuity(self):
        """Joint angles must not jump wildly between consecutive steps."""
        traj = self._plan([2.0, 1.0], [5.0, 3.0], n_steps=100)
        for k in range(1, traj.n_steps):
            diffs = np.abs(angle_diff_vec(traj.joint_angles[k],
                                          traj.joint_angles[k-1]))
            assert diffs.max() < np.radians(30), \
                f"Large joint jump at step {k}: {np.degrees(diffs)} °"

    def test_cross_quadrant(self):
        """Stroke spanning two quadrants."""
        traj = self._plan([-3.0, 2.0], [3.0, 2.0])
        ok   = traj.success_mask.mean()
        assert ok > 0.9, f"Too many IK failures: {100*(1-ok):.1f}%"

    def test_near_base_stroke(self):
        """Stroke passing through the near-base zone."""
        traj = self._plan([0.5, 0.5], [4.0, 2.0])
        ok   = traj.success_mask.mean()
        assert ok > 0.85, f"Too many IK failures near base: {100*(1-ok):.1f}%"
