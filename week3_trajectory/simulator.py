"""
Week 3 — Linear Trajectory Simulator
======================================
Interactive simulator: click two points, watch the arm move in a straight
line while servo velocities are displayed in real-time.

Usage:
    python3 simulator.py

Workflow:
    1.  Left-click on the canvas to set the START point  (green ●)
    2.  Left-click again to set the END point            (red   ●)
    3.  Press SPACE or click [▶ Play] to animate
    4.  Watch servo velocities update live in the right panel
    5.  Press R to reset and pick new points
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1_forward_kinematics"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week2_inverse_kinematics"))

from forward_kinematics import ThreeLinkArm
from trajectory import plan_linear_trajectory, TrajectoryResult

# ─── Palette ─────────────────────────────────────────────────────────────────
BG      = "#0d0d1a"
GRID    = "#1e1e3a"
LCOLORS = ["#4fc3f7", "#81d4fa", "#b3e5fc"]
JCOLORS = ["#ff6e40", "#ffa040", "#ffcc80", "#69ff47"]
TEXT    = "#e0e0e0"
P1_COL  = "#00e676"
P2_COL  = "#ff5252"
PATH_COL= "#ffd740"
TRACE_COL="#ff6e40"
VEL_COLS = ["#ef5350", "#42a5f5", "#66bb6a"]   # R, G, B for ω1, ω2, ω3


# ─── State ───────────────────────────────────────────────────────────────────
class SimState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.p1        = None
        self.p2        = None
        self.traj      = None    # TrajectoryResult
        self.frame     = 0
        self.playing   = False
        self.anim_obj  = None
        self.phase     = "set_p1"   # set_p1 → set_p2 → ready → playing → done


state = SimState()


# ─── Arm ─────────────────────────────────────────────────────────────────────
arm = ThreeLinkArm(link_lengths=(3.0, 2.5, 1.5))


# ─── Drawing helpers ─────────────────────────────────────────────────────────

def draw_workspace(ax):
    θ = np.linspace(0, 2 * np.pi, 360)
    xo = arm.max_reach * np.cos(θ)
    yo = arm.max_reach * np.sin(θ)
    ax.fill(xo, yo, color=(0.2, 0.4, 1.0, 0.05), zorder=0)
    ax.plot(xo, yo, color="#334466", lw=0.7, ls="--", zorder=0)


def draw_arm_at(ax, angles):
    fk     = arm.forward_kinematics(angles)
    joints = fk["joint_positions"]
    ee     = fk["end_effector"]
    phi    = fk["orientation"]

    for i in range(3):
        ax.plot([joints[i][0], joints[i+1][0]],
                [joints[i][1], joints[i+1][1]],
                color=LCOLORS[i], lw=5 - i, solid_capstyle="round", zorder=3)

    for i, (pos, col) in enumerate(zip(joints, JCOLORS)):
        ax.scatter(*pos, color=col, s=80 + 40*(i == 0), zorder=5)

    # EE orientation arrow
    arr = 0.4
    ax.annotate("", xy=(ee[0] + arr*np.cos(phi), ee[1] + arr*np.sin(phi)),
                xytext=ee,
                arrowprops=dict(arrowstyle="->", color=JCOLORS[-1], lw=1.5),
                zorder=6)

    return joints, ee


def setup_arm_axes(ax):
    ax.cla()
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.5, ls="--", alpha=0.7)
    draw_workspace(ax)
    lim = arm.max_reach * 1.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("X", color=TEXT, fontsize=9)
    ax.set_ylabel("Y", color=TEXT, fontsize=9)
    ax.tick_params(colors=TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)


def setup_vel_axes(axs):
    labels = ["Joint 1  ω₁", "Joint 2  ω₂", "Joint 3  ω₃"]
    for ax, lbl, col in zip(axs, labels, VEL_COLS):
        ax.cla()
        ax.set_facecolor("#0a0a18")
        ax.set_title(lbl, color=col, fontsize=8, pad=3)
        ax.tick_params(colors=TEXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.axhline(0, color="#334466", lw=0.8)
        ax.set_ylabel("deg/s", color=TEXT, fontsize=7)


# ─── Velocity panel update ────────────────────────────────────────────────────

def update_vel_panel(axs, traj: TrajectoryResult, frame: int):
    """Draw velocity time-series up to current frame, highlight current value."""
    if traj is None or frame < 1:
        setup_vel_axes(axs)
        return

    times = np.arange(traj.n_steps - 1) * traj.dt
    vels_deg = np.degrees(traj.joint_velocities)  # (N-1, 3)

    for j, (ax, col) in enumerate(zip(axs, VEL_COLS)):
        ax.cla()
        ax.set_facecolor("#0a0a18")
        ax.tick_params(colors=TEXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.axhline(0, color="#334466", lw=0.8)

        # Full trajectory (faint)
        ax.plot(times, vels_deg[:, j], color=col, alpha=0.25, lw=1)

        # Executed so far
        fi = min(frame - 1, len(times) - 1)
        ax.plot(times[:fi+1], vels_deg[:fi+1, j], color=col, lw=1.4)

        # Current value marker
        if fi >= 0:
            ax.scatter(times[fi], vels_deg[fi, j], color=col, s=40, zorder=5)
            ax.set_title(
                f"Joint {j+1}  ω{j+1} = {vels_deg[fi, j]:+.1f} °/s",
                color=col, fontsize=8, pad=3,
            )

        ax.set_ylabel("deg/s", color=TEXT, fontsize=7)
        ax.set_xlabel("t (s)", color=TEXT, fontsize=7)


# ─── Main arm panel update ────────────────────────────────────────────────────

def update_arm_panel(ax, frame):
    traj = state.traj
    setup_arm_axes(ax)

    # Planned straight-line path
    if traj is not None:
        ax.plot(traj.waypoints[:, 0], traj.waypoints[:, 1],
                color=PATH_COL, lw=1.0, ls="--", alpha=0.5, zorder=1,
                label="Planned path")
        # Traced so far
        if frame > 0:
            ax.plot(traj.waypoints[:frame+1, 0], traj.waypoints[:frame+1, 1],
                    color=TRACE_COL, lw=2.0, alpha=0.8, zorder=2,
                    label="Traced path")

    # Points
    if state.p1 is not None:
        ax.scatter(*state.p1, color=P1_COL, s=150, marker="o", zorder=7)
        ax.annotate("P1", state.p1, xytext=(6, 6),
                    textcoords="offset points", color=P1_COL, fontsize=9)
    if state.p2 is not None:
        ax.scatter(*state.p2, color=P2_COL, s=150, marker="o", zorder=7)
        ax.annotate("P2", state.p2, xytext=(6, 6),
                    textcoords="offset points", color=P2_COL, fontsize=9)

    # Arm
    if traj is not None and frame < traj.n_steps:
        angles = traj.joint_angles[frame]
        _, ee  = draw_arm_at(ax, angles)
        θ_deg  = np.degrees(angles)
        t_now  = frame * traj.dt
        vel_now = ("" if frame == 0 else
                   f"  ω=[{np.degrees(traj.joint_velocities[frame-1][0]):+.1f}, "
                   f"{np.degrees(traj.joint_velocities[frame-1][1]):+.1f}, "
                   f"{np.degrees(traj.joint_velocities[frame-1][2]):+.1f}] °/s")
        info = (f"t = {t_now:.3f} s   step {frame}/{traj.n_steps-1}\n"
                f"θ = [{θ_deg[0]:+.1f}°, {θ_deg[1]:+.1f}°, {θ_deg[2]:+.1f}°]\n"
                f"EE = ({ee[0]:.3f}, {ee[1]:.3f})" + vel_now)
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8,
                color=TEXT, va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="#0a0a18", ec="#334", alpha=0.85))
    else:
        # Idle — draw arm at rest
        draw_arm_at(ax, arm.joint_angles)

    # Phase hint
    hints = {
        "set_p1": "Step 1: Left-click to set start point P1",
        "set_p2": "Step 2: Left-click to set end point P2",
        "ready":  "Step 3: Press SPACE or [▶ Play] to animate",
        "playing": "Animating…",
        "done":   "Done! Press R to reset.",
    }
    ax.set_title(
        f"3-Link Arm — Linear Trajectory  |  {hints.get(state.phase, '')}",
        color=TEXT, fontsize=10,
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.suptitle("Bob Ross without ROS  ·  Week 3 — Linear Trajectory Simulator",
                 color=TEXT, fontsize=12, y=0.99)

    # ── Layout ──────────────────────────────────────────────────────────────
    ax_arm = fig.add_axes([0.04, 0.15, 0.52, 0.80])
    ax_v1  = fig.add_axes([0.60, 0.67, 0.37, 0.25])
    ax_v2  = fig.add_axes([0.60, 0.38, 0.37, 0.25])
    ax_v3  = fig.add_axes([0.60, 0.09, 0.37, 0.25])
    vel_axs = [ax_v1, ax_v2, ax_v3]

    # ── Speed slider ────────────────────────────────────────────────────────
    ax_spd = fig.add_axes([0.08, 0.05, 0.25, 0.025])
    s_speed = Slider(ax_spd, "EE speed", 0.5, 8.0, valinit=2.0,
                     color="#2a4080", track_color="#1e1e3a")
    s_speed.label.set_color(TEXT)
    s_speed.valtext.set_color(TEXT)

    # ── Buttons ─────────────────────────────────────────────────────────────
    ax_play  = fig.add_axes([0.36, 0.04, 0.08, 0.04])
    ax_reset = fig.add_axes([0.46, 0.04, 0.08, 0.04])
    btn_play  = Button(ax_play,  "▶ Play",  color="#1a2a4a", hovercolor="#2a4a8a")
    btn_reset = Button(ax_reset, "⟳ Reset", color="#2a1a1a", hovercolor="#4a2a2a")
    for btn in (btn_play, btn_reset):
        btn.label.set_color(TEXT)

    fig.text(0.04, 0.02,
             "Left-click: set P1 then P2  |  SPACE: play/pause  |  R: reset",
             color="#888", fontsize=8)

    # ── Initial draw ─────────────────────────────────────────────────────────
    update_arm_panel(ax_arm, 0)
    setup_vel_axes(vel_axs)
    fig.canvas.draw_idle()

    # ── Animation object ─────────────────────────────────────────────────────
    anim_holder = [None]

    def do_animate(frame):
        if state.phase not in ("playing",):
            return
        state.frame = frame
        update_arm_panel(ax_arm, frame)
        update_vel_panel(vel_axs, state.traj, frame)
        fig.canvas.draw_idle()

        if frame >= state.traj.n_steps - 1:
            state.phase = "done"
            if anim_holder[0] is not None:
                anim_holder[0].event_source.stop()

    def start_animation():
        if state.traj is None or state.phase not in ("ready", "done"):
            return
        state.frame = 0
        state.phase = "playing"
        interval_ms = max(20, int(state.traj.dt * 1000))
        anim_holder[0] = animation.FuncAnimation(
            fig, do_animate,
            frames=state.traj.n_steps,
            interval=interval_ms,
            repeat=False,
        )
        fig.canvas.draw_idle()

    def plan_trajectory():
        """(Re)plan trajectory using current P1, P2 and speed slider."""
        if state.p1 is None or state.p2 is None:
            return
        try:
            traj = plan_linear_trajectory(
                arm, state.p1, state.p2,
                ee_speed=s_speed.val,
                init_angles=arm.joint_angles.copy(),
            )
        except ValueError:
            return
        state.traj  = traj
        state.frame = 0
        state.phase = "ready"
        update_arm_panel(ax_arm, 0)
        update_vel_panel(vel_axs, traj, 0)
        fig.canvas.draw_idle()

    # ── Events ───────────────────────────────────────────────────────────────
    def on_click(event):
        if event.inaxes is not ax_arm:
            return
        if event.button != 1:
            return
        pt = np.array([event.xdata, event.ydata])
        if not arm.is_reachable(pt):
            return

        if state.phase in ("set_p1", "done", "ready"):
            if state.phase == "set_p1":
                state.p1 = pt
                state.p2 = None
                state.traj = None
                state.phase = "set_p2"
            else:
                # Re-picking P2 after a done / ready state
                state.p2 = pt
                plan_trajectory()
        elif state.phase == "set_p2":
            state.p2 = pt
            plan_trajectory()

        update_arm_panel(ax_arm, state.frame)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            if state.phase == "ready":
                start_animation()
            elif state.phase == "playing":
                if anim_holder[0] is not None:
                    anim_holder[0].event_source.stop()
                state.phase = "ready"
            elif state.phase == "done":
                # Replay
                plan_trajectory()
                start_animation()
        elif event.key in ("r", "R"):
            reset()
        elif event.key in ("q", "Q"):
            plt.close("all")

    def reset():
        if anim_holder[0] is not None:
            anim_holder[0].event_source.stop()
            anim_holder[0] = None
        state.reset()
        arm.set_joint_angles(np.zeros(3))
        update_arm_panel(ax_arm, 0)
        setup_vel_axes(vel_axs)
        fig.canvas.draw_idle()

    btn_play.on_clicked(lambda _: start_animation() if state.phase == "ready" else None)
    btn_reset.on_clicked(lambda _: reset())
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


if __name__ == "__main__":
    main()
