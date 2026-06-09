"""
Interactive Simulator — Week 1
==============================
Visualise the 3-link planar robotic arm with interactive joint-angle sliders.
Run:
    python visualizer.py

Controls:
    θ1, θ2, θ3 sliders → change joint angles in real-time
    'R' key            → reset to zero configuration
    'Q' key            → quit
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
from matplotlib.patches import FancyArrowPatch
from forward_kinematics import ThreeLinkArm


# ─── Colour palette ────────────────────────────────────────────────────────────
BG_COLOR     = "#0d0d1a"
GRID_COLOR   = "#1e1e3a"
LINK_COLORS  = ["#4fc3f7", "#81d4fa", "#b3e5fc"]   # light-blue gradient
JOINT_COLORS = ["#ff6e40", "#ffa040", "#ffcc80", "#69ff47"]  # orange → green EE
EE_COLOR     = "#69ff47"
TEXT_COLOR   = "#e0e0e0"
SLIDER_COLOR = "#1e1e3a"
WORKSPACE_COLOR = (0.3, 0.5, 1.0, 0.05)


def build_workspace_patch(arm: ThreeLinkArm, ax) -> None:
    """
    Draw the reachable workspace boundary.

    For this arm (L2+L3 >= L1), min_reach=0, so the workspace is a full
    disc — no inner dead zone is drawn.  A dashed inner circle is only
    shown when min_reach > 0 (i.e. when L1 > L2+L3).
    """
    θ = np.linspace(0, 2 * np.pi, 360)
    # Outer boundary (max reach)
    x_out = arm.max_reach * np.cos(θ)
    y_out = arm.max_reach * np.sin(θ)
    ax.fill(x_out, y_out, color=(0.2, 0.4, 1.0, 0.07), zorder=0)
    ax.plot(x_out, y_out, color="#334466", lw=0.8, ls="--", zorder=0,
            label=f"Max reach = {arm.max_reach:.1f}")
    # Inner dead zone — only when it actually exists
    if arm.min_reach > 0.05:
        x_in = arm.min_reach * np.cos(θ)
        y_in = arm.min_reach * np.sin(θ)
        ax.fill(x_in, y_in, color=BG_COLOR, zorder=0)
        ax.plot(x_in, y_in, color="#aa4444", lw=0.8, ls="--", zorder=0,
                label=f"Dead zone r={arm.min_reach:.1f}")


def draw_arm(ax, arm: ThreeLinkArm, fk: dict) -> None:
    """(Re)draw the arm given a forward-kinematics result."""
    joints = fk["joint_positions"]   # shape (4, 2)
    ee     = fk["end_effector"]
    phi    = fk["orientation"]

    ax.cla()

    # Background / grid
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.8)

    # Workspace
    build_workspace_patch(arm, ax)

    # Links
    for i in range(3):
        ax.plot(
            [joints[i][0], joints[i + 1][0]],
            [joints[i][1], joints[i + 1][1]],
            color=LINK_COLORS[i],
            linewidth=5 - i,
            solid_capstyle="round",
            zorder=2,
            label=f"Link {i+1}  (L={arm.link_lengths[i]:.1f})",
        )

    # Joints
    for i, (pos, col) in enumerate(zip(joints, JOINT_COLORS)):
        ax.scatter(*pos, color=col, s=120 if i > 0 else 200, zorder=5)
        label = f"J{i}" if i < 3 else "EE"
        ax.annotate(
            label,
            pos,
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            color=col,
        )

    # End-effector orientation arrow
    arrow_len = 0.6
    ax.annotate(
        "",
        xy=(ee[0] + arrow_len * np.cos(phi), ee[1] + arrow_len * np.sin(phi)),
        xytext=ee,
        arrowprops=dict(arrowstyle="->", color=EE_COLOR, lw=1.8),
        zorder=6,
    )

    # Stats text
    θ_deg = np.degrees(fk["angles"])
    stats = (
        f"θ₁ = {θ_deg[0]:+.1f}°  θ₂ = {θ_deg[1]:+.1f}°  θ₃ = {θ_deg[2]:+.1f}°\n"
        f"End-effector  x = {ee[0]:.3f}   y = {ee[1]:.3f}\n"
        f"Orientation  φ = {np.degrees(phi):.1f}°   "
        f"Reach = {np.linalg.norm(ee):.3f} / {arm.max_reach:.1f}"
    )
    ax.text(
        0.02, 0.98, stats,
        transform=ax.transAxes,
        fontsize=9, color=TEXT_COLOR,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="#0a0a18", ec="#334", alpha=0.85),
    )

    # Legend
    handles = [
        mpatches.Patch(color=LINK_COLORS[i], label=f"Link {i+1}  (L={arm.link_lengths[i]:.1f})")
        for i in range(3)
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=8,
        framealpha=0.4,
        facecolor="#0a0a18",
        edgecolor="#334",
        labelcolor=TEXT_COLOR,
    )

    # Axes
    limit = arm.max_reach * 1.15
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("X (world units)", color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Y (world units)", color=TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.set_title(
        "3-Link Planar Arm — Forward Kinematics  |  Week 1",
        color=TEXT_COLOR, fontsize=11, pad=10,
    )


def main():
    arm = ThreeLinkArm(
        link_lengths=(3.0, 2.5, 1.5),
        joint_angles=(np.radians(30), np.radians(-20), np.radians(10)),
    )

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 9), facecolor=BG_COLOR)
    fig.suptitle(
        "Bob Ross without ROS  ·  Interactive FK Simulator",
        color=TEXT_COLOR, fontsize=12, y=0.99,
    )

    # Main axis
    ax = fig.add_axes([0.08, 0.28, 0.86, 0.68])

    # Slider axes
    s_ax1 = fig.add_axes([0.15, 0.18, 0.70, 0.025])
    s_ax2 = fig.add_axes([0.15, 0.12, 0.70, 0.025])
    s_ax3 = fig.add_axes([0.15, 0.06, 0.70, 0.025])

    slider_kwargs = dict(color="#2a4080", track_color=SLIDER_COLOR)

    slider1 = Slider(s_ax1, "θ₁  (°)", -180, 180,
                     valinit=np.degrees(arm.joint_angles[0]), **slider_kwargs)
    slider2 = Slider(s_ax2, "θ₂  (°)", -180, 180,
                     valinit=np.degrees(arm.joint_angles[1]), **slider_kwargs)
    slider3 = Slider(s_ax3, "θ₃  (°)", -180, 180,
                     valinit=np.degrees(arm.joint_angles[2]), **slider_kwargs)

    for sl in (slider1, slider2, slider3):
        sl.label.set_color(TEXT_COLOR)
        sl.valtext.set_color(TEXT_COLOR)

    # Initial draw
    fk = arm.forward_kinematics()
    draw_arm(ax, arm, fk)
    fig.canvas.draw_idle()

    # ── Callbacks ────────────────────────────────────────────────────────────
    def update(_val):
        angles = np.radians([slider1.val, slider2.val, slider3.val])
        arm.set_joint_angles(angles)
        fk = arm.forward_kinematics()
        draw_arm(ax, arm, fk)
        fig.canvas.draw_idle()

    slider1.on_changed(update)
    slider2.on_changed(update)
    slider3.on_changed(update)

    def on_key(event):
        if event.key in ("r", "R"):
            slider1.reset()
            slider2.reset()
            slider3.reset()
        elif event.key in ("q", "Q"):
            plt.close("all")

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


if __name__ == "__main__":
    main()
