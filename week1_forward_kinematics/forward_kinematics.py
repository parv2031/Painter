"""
Week 1 - Forward Kinematics for a 3-Link Planar Robotic Arm
============================================================
Project: Bob Ross without ROS — Robotics Society Summer Project
Mentors: Anjaneya and Parv

Description:
    Implements forward kinematics (FK) for a 3-DoF planar robotic
    manipulator using the Denavit-Hartenberg (DH) convention and
    direct geometric transformation.

Theory:
    For a planar 3-link arm with link lengths l1, l2, l3 and joint
    angles θ1, θ2, θ3 (all in radians, measured from each link's
    local x-axis):

    Joint positions:
        J0 = (0, 0)                           [base/origin]
        J1 = (l1*cos(θ1),
               l1*sin(θ1))
        J2 = (J1.x + l2*cos(θ1+θ2),
               J1.y + l2*sin(θ1+θ2))
        J3 = (J2.x + l3*cos(θ1+θ2+θ3),
               J2.y + l3*sin(θ1+θ2+θ3))     [end-effector]

    End-effector orientation:
        φ = θ1 + θ2 + θ3
"""

import numpy as np


class ThreeLinkArm:
    """
    A 3-link planar robotic arm with forward kinematics.

    Attributes
    ----------
    link_lengths : tuple[float, float, float]
        Lengths of the three links (l1, l2, l3) in world units.
    joint_angles : np.ndarray
        Current joint angles [θ1, θ2, θ3] in radians.
    joint_limits : list[tuple[float, float]]
        Min/max angle bounds for each joint (radians).
    """

    def __init__(
        self,
        link_lengths: tuple[float, float, float] = (3.0, 2.5, 1.5),
        joint_angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
        joint_limits: list[tuple[float, float]] | None = None,
    ):
        """
        Initialise the arm.

        Parameters
        ----------
        link_lengths:
            Lengths of links 1, 2, and 3.
        joint_angles:
            Initial joint angles θ1, θ2, θ3 in radians.
        joint_limits:
            [(θ_min, θ_max), ...] for each joint.
            Defaults to (-π, π) for all joints.
        """
        self.link_lengths = np.array(link_lengths, dtype=float)
        self.joint_angles = np.array(joint_angles, dtype=float)

        if joint_limits is None:
            self.joint_limits = [(-np.pi, np.pi)] * 3
        else:
            self.joint_limits = joint_limits

    # ------------------------------------------------------------------
    # Core FK
    # ------------------------------------------------------------------

    def forward_kinematics(
        self, angles: np.ndarray | None = None
    ) -> dict:
        """
        Compute forward kinematics for the given (or stored) joint angles.

        Parameters
        ----------
        angles:
            Joint angles [θ1, θ2, θ3] in radians. Uses ``self.joint_angles``
            if not provided.

        Returns
        -------
        dict with keys:
            ``joint_positions`` — array of shape (4, 2): [J0, J1, J2, J3]
            ``end_effector``    — (x, y) position of the end-effector
            ``orientation``     — total orientation φ of the end-effector
            ``angles``          — the angles used for this computation
        """
        if angles is None:
            angles = self.joint_angles

        l1, l2, l3 = self.link_lengths
        θ1, θ2, θ3 = angles

        # Cumulative angles
        α1 = θ1
        α2 = θ1 + θ2
        α3 = θ1 + θ2 + θ3

        # Joint positions
        j0 = np.array([0.0, 0.0])
        j1 = j0 + l1 * np.array([np.cos(α1), np.sin(α1)])
        j2 = j1 + l2 * np.array([np.cos(α2), np.sin(α2)])
        j3 = j2 + l3 * np.array([np.cos(α3), np.sin(α3)])

        return {
            "joint_positions": np.array([j0, j1, j2, j3]),
            "end_effector": j3,
            "orientation": α3,  # φ in radians
            "angles": angles.copy(),
        }

    # ------------------------------------------------------------------
    # Jacobian (analytical, for future use in IK)
    # ------------------------------------------------------------------

    def jacobian(self, angles: np.ndarray | None = None) -> np.ndarray:
        """
        Compute the 2×3 analytical Jacobian J such that:
            ṗ = J · θ̇
        where ṗ = [ẋ, ẏ]ᵀ is the end-effector velocity.

        Parameters
        ----------
        angles:
            Joint angles. Uses ``self.joint_angles`` if not provided.

        Returns
        -------
        np.ndarray of shape (2, 3)
        """
        if angles is None:
            angles = self.joint_angles

        l1, l2, l3 = self.link_lengths
        θ1, θ2, θ3 = angles
        α1 = θ1
        α2 = θ1 + θ2
        α3 = θ1 + θ2 + θ3

        # ∂x/∂θi = -l_i * sin(α_i) - l_{i+1}*sin(α_{i+1}) - ...
        # ∂y/∂θi =  l_i * cos(α_i) + l_{i+1}*cos(α_{i+1}) + ...
        J = np.array(
            [
                [
                    -l1 * np.sin(α1) - l2 * np.sin(α2) - l3 * np.sin(α3),
                    -l2 * np.sin(α2) - l3 * np.sin(α3),
                    -l3 * np.sin(α3),
                ],
                [
                    l1 * np.cos(α1) + l2 * np.cos(α2) + l3 * np.cos(α3),
                    l2 * np.cos(α2) + l3 * np.cos(α3),
                    l3 * np.cos(α3),
                ],
            ]
        )
        return J

    # ------------------------------------------------------------------
    # Workspace boundary (reachability)
    # ------------------------------------------------------------------

    @property
    def max_reach(self) -> float:
        """Maximum reach of the arm (fully extended)."""
        return float(np.sum(self.link_lengths))

    @property
    def min_reach(self) -> float:
        """
        Minimum reach for a 3-link planar arm with unlimited joint rotation.

        For a 3-link arm, L2 and L3 can fold independently, so the true
        minimum reachable distance from the base is:

            min_reach = max(0, L1 - L2 - L3)

        If L2 + L3 >= L1 (our case: 4.0 >= 3.0), the arm can reach all the
        way back to the origin — there is no dead zone.

        Compare with a 2-link arm where min_reach = |L1 - L2|, because the
        second link cannot fold past the base of the first.
        """
        l1, l2, l3 = self.link_lengths
        return float(max(0.0, l1 - l2 - l3))

    def is_reachable(self, point: np.ndarray) -> bool:
        """Return True if the point is within the arm's workspace."""
        dist = float(np.linalg.norm(point))
        return self.min_reach <= dist <= self.max_reach

    # ------------------------------------------------------------------
    # Setters / helpers
    # ------------------------------------------------------------------

    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Clamp and set joint angles."""
        clamped = np.array(
            [
                np.clip(angles[i], *self.joint_limits[i])
                for i in range(3)
            ]
        )
        self.joint_angles = clamped

    def __repr__(self) -> str:  # pragma: no cover
        θ_deg = np.degrees(self.joint_angles)
        return (
            f"ThreeLinkArm(links={self.link_lengths}, "
            f"angles=[{θ_deg[0]:.1f}°, {θ_deg[1]:.1f}°, {θ_deg[2]:.1f}°])"
        )
