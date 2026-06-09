"""
Interactive IK Visualizer — Week 3-4
=====================================
Left-click anywhere on the canvas → the arm reaches for that point.
Toggle between Analytical and Jacobian IK with the radio buttons.
Set a target end-effector orientation with the φ slider (analytical only).

Run:
    python3 visualizer_ik.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RadioButtons, Slider
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1_forward_kinematics"))
from forward_kinematics import ThreeLinkArm
from inverse_kinematics import ik_analytical, ik_jacobian

# ─── Palette ─────────────────────────────────────────────────────────────────
BG      = "#0d0d1a"
GRID    = "#1e1e3a"
LCOLORS = ["#4fc3f7", "#81d4fa", "#b3e5fc"]
JCOLORS = ["#ff6e40", "#ffa040", "#ffcc80", "#69ff47"]
TARGET  = "#ff4081"
TEXT    = "#e0e0e0"
REACH   = (0.2, 0.4, 1.0, 0.07)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def draw_workspace(arm, ax):
    θ = np.linspace(0, 2 * np.pi, 360)
    xo, yo = arm.max_reach * np.cos(θ), arm.max_reach * np.sin(θ)
    ax.fill(xo, yo, color=REACH, zorder=0)
    ax.plot(xo, yo, color="#334466", lw=0.8, ls="--", zorder=0)


def draw_arm(ax, arm, fk, target=None, result=None):
    ax.cla()
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.5, ls="--", alpha=0.7)
    draw_workspace(arm, ax)

    joints = fk["joint_positions"]
    ee     = fk["end_effector"]
    phi    = fk["orientation"]

    # Links
    for i in range(3):
        ax.plot([joints[i][0], joints[i+1][0]],
                [joints[i][1], joints[i+1][1]],
                color=LCOLORS[i], lw=5-i, solid_capstyle="round", zorder=2)

    # Joints
    for i, (pos, col) in enumerate(zip(joints, JCOLORS)):
        ax.scatter(*pos, color=col, s=120 if i > 0 else 200, zorder=5)
        ax.annotate(f"J{i}" if i < 3 else "EE", pos,
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=8, color=col)

    # EE orientation arrow
    arr = 0.5
    ax.annotate("", xy=(ee[0] + arr*np.cos(phi), ee[1] + arr*np.sin(phi)),
                xytext=ee,
                arrowprops=dict(arrowstyle="->", color=JCOLORS[-1], lw=1.8),
                zorder=6)

    # Target dot
    if target is not None:
        ax.scatter(*target, color=TARGET, s=160, marker="x",
                   linewidths=2.5, zorder=7, label="Target")
        # Line from EE to target
        ax.plot([ee[0], target[0]], [ee[1], target[1]],
                color=TARGET, lw=0.8, ls=":", alpha=0.6, zorder=3)

    # Status text
    θ_deg = np.degrees(fk["angles"])
    status = ""
    if result is not None:
        ok = "✓ reached" if result["success"] else "✗ failed"
        status = (f"  IK: {ok}  |  err = {result['position_error']:.4f}"
                  f"  |  iter = {result['iterations']}"
                  f"\n  {result['note']}")

    info = (f"θ₁={θ_deg[0]:+.1f}°  θ₂={θ_deg[1]:+.1f}°  θ₃={θ_deg[2]:+.1f}°\n"
            f"EE  x={ee[0]:.3f}  y={ee[1]:.3f}  φ={np.degrees(phi):.1f}°"
            + status)

    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8.5,
            color=TEXT, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#0a0a18", ec="#334", alpha=0.85))

    lim = arm.max_reach * 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("X", color=TEXT, fontsize=9)
    ax.set_ylabel("Y", color=TEXT, fontsize=9)
    ax.tick_params(colors=TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.set_title("3-Link Arm — Inverse Kinematics  |  Click to set target",
                 color=TEXT, fontsize=11)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    arm = ThreeLinkArm(
        link_lengths=(3.0, 2.5, 1.5),
        joint_angles=(np.radians(30), np.radians(-20), np.radians(10)),
    )

    fig = plt.figure(figsize=(11, 9), facecolor=BG)
    fig.suptitle("Bob Ross without ROS  ·  IK Simulator (Week 3-4)",
                 color=TEXT, fontsize=12, y=0.99)

    # Main axis
    ax = fig.add_axes([0.08, 0.22, 0.72, 0.74])

    # Radio — solver choice
    ax_radio = fig.add_axes([0.83, 0.60, 0.14, 0.18], facecolor="#0a0a18")
    radio = RadioButtons(ax_radio, ("Analytical\nElbow-up",
                                    "Analytical\nElbow-dn",
                                    "Jacobian\nPseudo-inv"),
                         activecolor="#4fc3f7")
    ax_radio.set_title("Solver", color=TEXT, fontsize=8, pad=4)
    for lbl in radio.labels:
        lbl.set_color(TEXT); lbl.set_fontsize(8)

    # φ slider (used only by analytical solver)
    ax_phi = fig.add_axes([0.15, 0.10, 0.65, 0.025])
    s_phi  = Slider(ax_phi, "φ target (°)", -180, 180, valinit=0,
                    color="#2a4080", track_color="#1e1e3a")
    s_phi.label.set_color(TEXT); s_phi.valtext.set_color(TEXT)
    fig.text(0.15, 0.07,
             "φ slider: only used by Analytical solvers. "
             "Jacobian solver leaves orientation free.",
             color="#aaaaaa", fontsize=7.5)

    # Hint
    fig.text(0.08, 0.16,
             "Left-click on the canvas to set a target   |   R = reset arm   |   Q = quit",
             color="#888888", fontsize=8)

    state = {"target": None, "result": None}

    def solve_and_draw(target):
        phi_rad = np.radians(s_phi.val)
        label   = radio.value_selected

        if "Elbow-up" in label:
            res = ik_analytical(arm, *target, phi_rad, elbow_up=True)
        elif "Elbow-dn" in label:
            res = ik_analytical(arm, *target, phi_rad, elbow_up=False)
        else:  # Jacobian
            res = ik_jacobian(arm, *target,
                              theta_init=arm.joint_angles.copy(),
                              theta_preferred=np.zeros(3))

        if res["success"]:
            arm.set_joint_angles(res["angles"])
        state["result"] = res

        fk = arm.forward_kinematics()
        draw_arm(ax, arm, fk, target=target, result=res)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax:
            return
        if event.button != 1:
            return
        target = np.array([event.xdata, event.ydata])
        state["target"] = target
        solve_and_draw(target)

    def on_slider_change(_):
        if state["target"] is not None:
            solve_and_draw(state["target"])

    def on_key(event):
        if event.key in ("r", "R"):
            arm.set_joint_angles(np.array([0.0, 0.0, 0.0]))
            state["target"] = None
            state["result"] = None
            fk = arm.forward_kinematics()
            draw_arm(ax, arm, fk)
            fig.canvas.draw_idle()
        elif event.key in ("q", "Q"):
            plt.close("all")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    s_phi.on_changed(on_slider_change)
    radio.on_clicked(lambda _: on_slider_change(None))

    # Initial draw
    fk = arm.forward_kinematics()
    draw_arm(ax, arm, fk)
    plt.show()


if __name__ == "__main__":
    main()
