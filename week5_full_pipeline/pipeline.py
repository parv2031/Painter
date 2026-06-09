"""
Week 5 — Full Drawing Pipeline
================================
Run:
    python3 pipeline.py
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "week1_forward_kinematics"))
sys.path.insert(0, os.path.join(_HERE, "..", "week2_inverse_kinematics"))
sys.path.insert(0, os.path.join(_HERE, "..", "week3_trajectory"))
sys.path.insert(0, os.path.join(_HERE, "..", "week4_image_processing"))

import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import filedialog

from forward_kinematics import ThreeLinkArm
from trajectory import plan_linear_trajectory
from inverse_kinematics import ik_analytical_auto
from image_processor import process_image, save_strokes

ARM        = ThreeLinkArm(link_lengths=(3.0, 2.5, 1.5))
DRAW_SPEED = 6.0
LIFT_SPEED = 14.0
CANVAS_R   = 5.0
VEL_WINDOW = 200
JERK_THRESHOLD = 600.0  # deg/s — drawing segments above this trigger a pen-up detour
BASE_INTERVAL  = 20     # ms per frame at 1× speed


# ─── Coord transform — upper-right quadrant mapping ─────────────────────
#
# The arm base sits at the origin — the "bottom-left corner" of the canvas.
# The drawable region is the upper-right quadrant (x≥0, y≥0).
#
# Geometry
# --------
# We fit the image into a square box of side BOX arm-units anchored at
# (MARGIN, MARGIN).  The farthest drawable corner is then at
#   (MARGIN+BOX, MARGIN+BOX)  ⇒  distance = sqrt(2)*(MARGIN+BOX) from origin
# With MARGIN=0.4, BOX=4.2: sqrt(2)*4.6 ≈ 6.51  <  max_reach (7.0)  ✓
#
# Aspect-ratio preservation
# -------------------------
# uniform scale = BOX / max(img_W, img_H)
# The shorter axis is centered within the box (letterbox-style, no distortion).
_MARGIN = 0.4    # arm-units from origin (avoids near-zero singularity)
_BOX    = 4.2    # arm-unit side length of the target square

def strokes_to_arm(strokes_px: list, img_shape: tuple) -> list:
    """
    Map pixel (x, y) → arm (x, y) in the upper-right quadrant.

    Pixel coordinate system : (0,0) = top-left, y increases downward.
    Arm coordinate system   : (0,0) = origin,   y increases upward.
    Y is therefore flipped.

    Aspect ratio is preserved via uniform scaling (BOX / max(W, H)).
    """
    H, W  = img_shape[:2]
    scale = _BOX / max(W, H)          # arm-units per pixel (uniform)

    scaled_W = W * scale
    scaled_H = H * scale

    # Centre the image in the BOX × BOX target area
    x_off = _MARGIN + (_BOX - scaled_W) / 2.0
    y_off = _MARGIN + (_BOX - scaled_H) / 2.0

    out = []
    for s in strokes_px:
        arm_x = x_off + s[:, 0] * scale
        arm_y = y_off + (H - 1 - s[:, 1]) * scale   # flip y
        out.append(np.column_stack([arm_x, arm_y]))
    return out


# ─── Greedy ordering ──────────────────────────────────────────────────────────
def order_strokes(strokes):
    remaining = list(range(len(strokes)))
    ordered, cur = [], np.zeros(2)
    while remaining:
        bi, bd, br = None, np.inf, False
        for idx in remaining:
            s = strokes[idx]
            ds = np.linalg.norm(s[0]  - cur)
            de = np.linalg.norm(s[-1] - cur)
            if ds < bd: bd, bi, br = ds, idx, False
            if de < bd: bd, bi, br = de, idx, True
        remaining.remove(bi)
        s = strokes[bi].copy()
        if br: s = s[::-1]
        ordered.append(s); cur = s[-1]
    return ordered


# ─── Frame pre-computation ────────────────────────────────────────────────────
def _seg_frames(p1, p2, prev_angles, is_drawing, stroke_idx):
    dist = float(np.linalg.norm(p2 - p1))
    if dist < 5e-3:
        return []
    speed = DRAW_SPEED if is_drawing else LIFT_SPEED
    try:
        traj = plan_linear_trajectory(ARM, p1, p2, ee_speed=speed,
                                      min_steps=2, max_steps=300,
                                      init_angles=prev_angles)
    except Exception:
        return []
    frames = []
    for k in range(traj.n_steps):
        vel = traj.joint_velocities[k-1] if k > 0 else np.zeros(3)
        frames.append({
            "angles":     traj.joint_angles[k].copy(),
            "vel_deg":    np.degrees(vel),
            "ee":         traj.waypoints[k].copy(),
            "is_drawing": is_drawing,
            "stroke_idx": stroke_idx,
        })
    return frames


def _max_vel(frames):
    """Max absolute velocity (deg/s) across all joints, ignoring frame 0."""
    if len(frames) < 2:
        return 0.0
    return max(float(np.abs(f["vel_deg"]).max()) for f in frames[1:])


def _seg_frames_safe(p1, p2, prev_angles, stroke_idx):
    """
    Plan a *drawing* segment with automatic pen-up detour if a velocity
    spike (elbow flip / singularity) is detected.

    Strategy
    --------
    1. Plan the direct p1→p2 drawing segment.
    2. If max |vel| < JERK_THRESHOLD: return as-is.
    3. Otherwise try inserting a via-point V located at the segment midpoint
       displaced perpendicularly (both directions, three radii).
       - p1 → V : pen-UP (arm reconfigures freely)
       - V  → p2: pen-DOWN (drawing resumes)
    4. Accept the best via-point that reduces peak velocity by ≥35%.
       If no improvement found, return the original direct path.

    Why perpendicular displacement works
    -------------------------------------
    Elbow flips occur when the IK solver must pick between two geometrically
    distant solutions because the straight-line path passes near a singularity
    (full extension or full fold).  A perpendicular detour moves the EE away
    from the singularity, giving the IK a smooth, continuous path to follow.
    The pen-UP phase lets the arm reconfigure without leaving an unwanted mark.
    """
    direct = _seg_frames(p1, p2, prev_angles, True, stroke_idx)
    if not direct or _max_vel(direct) < JERK_THRESHOLD:
        return direct

    orig_max = _max_vel(direct)
    best, best_max = direct, orig_max

    direction = p2 - p1
    dist = float(np.linalg.norm(direction))
    if dist < 0.1:
        return direct

    unit = direction / dist
    perp = np.array([-unit[1], unit[0]])   # 90° CCW
    mid  = (p1 + p2) * 0.5

    for sign in (1.0, -1.0):
        for r in (0.8, 1.5, 2.5):
            via = mid + sign * perp * r
            via_r = float(np.linalg.norm(via))

            # Via-point must be inside reachable workspace
            if not (0.1 < via_r < ARM.max_reach * 0.92):
                continue

            f1 = _seg_frames(p1, via, prev_angles, False, stroke_idx)  # pen-UP
            if not f1:
                continue

            f2 = _seg_frames(via, p2, f1[-1]["angles"], True, stroke_idx)  # pen-DOWN
            if not f2:
                continue

            trial_max = max(_max_vel(f1), _max_vel(f2))
            if trial_max < best_max * 0.65:   # accept if ≥35% improvement
                best, best_max = f1 + f2, trial_max

            if best_max < JERK_THRESHOLD:
                break
        if best_max < JERK_THRESHOLD:
            break

    if best is not direct:
        improvement = 100 * (orig_max - best_max) / orig_max
        print(f"\n  ✓ Detour: {orig_max:.0f}→{best_max:.0f} deg/s ({improvement:.0f}% better)",
              end="")
    return best


def compute_frames(ordered_strokes):
    p0   = ordered_strokes[0][0]
    boot = ik_analytical_auto(ARM, float(p0[0]), float(p0[1]))
    prev = boot["angles"] if boot["success"] else np.zeros(3)

    total_segs = sum(max(len(s)-1,1) for s in ordered_strokes) + len(ordered_strokes)
    done, all_frames = 0, []
    print(f"  Pre-computing {total_segs} segments …")

    for si, stroke in enumerate(ordered_strokes):
        if si > 0 and all_frames:
            lf = _seg_frames(all_frames[-1]["ee"], stroke[0], prev, False, si)
            all_frames.extend(lf)
            if lf: prev = lf[-1]["angles"]
        done += 1

        for pi in range(len(stroke)-1):
            # Use safe planner (with automatic detour) for drawing segments
            sf = _seg_frames_safe(stroke[pi], stroke[pi+1], prev, si)
            all_frames.extend(sf)
            if sf: prev = sf[-1]["angles"]
            done += 1

        if done % 50 == 0 or done == total_segs:
            pct = 100*done/total_segs
            bar = "█"*int(pct/5) + "░"*(20-int(pct/5))
            print(f"  [{bar}] {pct:5.1f}%", end="\r")

    print(f"\n  Done — {len(all_frames)} total frames")
    return all_frames


# ─── FK helper ────────────────────────────────────────────────────────────────
def fk_joints(angles):
    """Return (J0, J1, J2, J3) joint positions — uses joint_positions array."""
    res = ARM.forward_kinematics(angles)
    pts = res["joint_positions"]   # shape (4, 2): base, j1, j2, EE
    return pts[0], pts[1], pts[2], pts[3]


# ─── Preview (non-blocking, waits for ENTER) ─────────────────────────────────
def show_preview(img_bgr, edges, ordered_strokes):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0d0d1a")
    fig.canvas.manager.set_window_title("Preview — press ENTER in terminal to start")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    axs[0].imshow(img_rgb)
    axs[0].set_title("Original Image", color="white", fontsize=11)
    axs[1].imshow(edges, cmap="gray_r")
    axs[1].set_title("Sketch",         color="white", fontsize=11)

    # Waypoints panel — arm coordinates, upper-right quadrant
    cmap = plt.get_cmap("tab20")
    axs[2].set_facecolor("#0a0a18")
    axs[2].set_title("Waypoints (arm space)", color="white", fontsize=11)
    # Y is NOT inverted — arm coords already have y going upward
    for i, s in enumerate(ordered_strokes):
        axs[2].plot(s[:, 0], s[:, 1], "o-", ms=1.8, lw=0.7, color=cmap(i % 20))
    axs[2].set_xlim(-0.2, _MARGIN + _BOX + 0.3)
    axs[2].set_ylim(-0.2, _MARGIN + _BOX + 0.3)
    axs[2].set_aspect("equal")
    axs[2].plot(0, 0, "o", color="#6060ff", ms=7)   # arm base

    for ax in axs[:2]: ax.axis("off")
    axs[2].axis("off")
    fig.tight_layout()
    plt.show(block=False); plt.pause(0.3)
    input("\n  ► Preview is open. Press ENTER to start the simulation… ")
    plt.close(fig)


# ─── Main animation ───────────────────────────────────────────────────────────
def run_animation(all_frames, ordered_strokes):
    fig = plt.figure(figsize=(17, 9.5))
    fig.patch.set_facecolor("#0d0d1a")
    fig.canvas.manager.set_window_title("Week 5 — Full Drawing Simulation")

    # Extra bottom row for the speed slider
    gs = gridspec.GridSpec(5, 2, figure=fig,
                           width_ratios=[2.2, 1],
                           height_ratios=[1, 1, 1, 0.08, 0.08],
                           hspace=0.05, wspace=0.06,
                           left=0.04, right=0.98, top=0.95, bottom=0.10)
    ax_c  = fig.add_subplot(gs[0:3, 0])
    ax_w1 = fig.add_subplot(gs[0, 1])
    ax_w2 = fig.add_subplot(gs[1, 1])
    ax_w3 = fig.add_subplot(gs[2, 1])
    ax_st = fig.add_subplot(gs[3, :])
    ax_sp = fig.add_axes([0.15, 0.025, 0.55, 0.022])  # speed slider
    ax_sp.set_facecolor("#1e1e3a")

    # Canvas — zoomed to upper-right quadrant where all drawing happens
    ax_c.set_facecolor("#0a0a18")
    ax_c.set_xlim(-0.8, ARM.max_reach + 0.5)
    ax_c.set_ylim(-0.8, ARM.max_reach + 0.5)
    ax_c.set_aspect("equal")
    ax_c.axis("off")
    ax_c.set_title("Drawing Simulation  (arm base ─●, canvas = upper-right quadrant)",
                   color="white", fontsize=11, pad=6)

    # Draw axes (x and y from origin)
    ax_c.axhline(0, color="#2a2a50", lw=0.8, zorder=0)
    ax_c.axvline(0, color="#2a2a50", lw=0.8, zorder=0)

    # Workspace arc (upper-right quarter of the max-reach circle)
    th_q = np.linspace(0, np.pi / 2, 180)
    ax_c.plot(ARM.max_reach * np.cos(th_q), ARM.max_reach * np.sin(th_q),
              color="#2a2a50", lw=1.2, ls="--", zorder=0)

    # Drawable box corners (faint rectangle showing the target area)
    bx = [_MARGIN, _MARGIN+_BOX, _MARGIN+_BOX, _MARGIN, _MARGIN]
    by = [_MARGIN, _MARGIN,      _MARGIN+_BOX,  _MARGIN+_BOX, _MARGIN]
    ax_c.plot(bx, by, color="#2a3060", lw=0.8, ls=":", zorder=0)

    ax_c.plot(0, 0, "o", color="#6060ff", ms=7, zorder=1, label="arm base")

    # Velocity axes
    vel_colors = ["#ff4c60", "#4c9fff", "#4cff80"]
    vel_labels = ["ω₁ Joint 1", "ω₂ Joint 2", "ω₃ Joint 3"]
    vel_axes   = [ax_w1, ax_w2, ax_w3]
    for ax, col, lbl in zip(vel_axes, vel_colors, vel_labels):
        ax.set_facecolor("#0a0a18"); ax.tick_params(colors="gray", labelsize=7)
        ax.spines[:].set_color("#2a2a50")
        ax.set_ylabel(lbl, color=col, fontsize=8); ax.yaxis.set_label_position("right")
        ax.axhline(0, color="#2a2a50", lw=0.8)

    ax_st.axis("off")
    status_txt = ax_st.text(0.5, 0.5, "Starting…", ha="center", va="center",
                             color="white", fontsize=9, transform=ax_st.transAxes)

    # Drawables
    link_lines = [
        ax_c.plot([], [], color="#3a8fff", lw=4,   solid_capstyle="round", zorder=5)[0],
        ax_c.plot([], [], color="#3affaf", lw=3,   solid_capstyle="round", zorder=5)[0],
        ax_c.plot([], [], color="#ff8c3a", lw=2.5, solid_capstyle="round", zorder=5)[0],
    ]
    joint_dots = [ax_c.plot([], [], "o", color="white", ms=5, zorder=6)[0] for _ in range(4)]
    ee_dot     =  ax_c.plot([], [], "o", color="#ffff00", ms=6, zorder=7)[0]
    stroke_lines = [ax_c.plot([], [], color="#ff6680", lw=1.2, alpha=0.9, zorder=3)[0]
                    for _ in ordered_strokes]
    lift_line    =  ax_c.plot([], [], "--", color="#333366", lw=0.8, zorder=2)[0]
    vel_lines    = [ax.plot([], [], color=c, lw=1.0)[0]
                    for ax, c in zip(vel_axes, vel_colors)]

    # State buffers
    stroke_pts = [[] for _ in ordered_strokes]
    lift_xs, lift_ys = [], []
    vel_xs, vel_ys   = [], [[], [], []]

    def update(fi):
        if fi >= len(all_frames): return []
        f   = all_frames[fi]
        ang, vel, ee, si = f["angles"], f["vel_deg"], f["ee"], f["stroke_idx"]
        J   = fk_joints(ang)

        link_lines[0].set_data([J[0][0],J[1][0]], [J[0][1],J[1][1]])
        link_lines[1].set_data([J[1][0],J[2][0]], [J[1][1],J[2][1]])
        link_lines[2].set_data([J[2][0],J[3][0]], [J[2][1],J[3][1]])
        for dot, pt in zip(joint_dots, J): dot.set_data([pt[0]], [pt[1]])
        ee_dot.set_data([J[3][0]], [J[3][1]])

        if f["is_drawing"]:
            stroke_pts[si].append(ee)
            xs = [p[0] for p in stroke_pts[si]]; ys = [p[1] for p in stroke_pts[si]]
            stroke_lines[si].set_data(xs, ys)
        else:
            lift_xs.append(ee[0]); lift_ys.append(ee[1])
            lift_line.set_data(lift_xs, lift_ys)

        vel_xs.append(fi)
        for j in range(3): vel_ys[j].append(vel[j])
        if len(vel_xs) > VEL_WINDOW:
            vel_xs.pop(0)
            for j in range(3): vel_ys[j].pop(0)

        for j, (vl, ax) in enumerate(zip(vel_lines, vel_axes)):
            vl.set_data(vel_xs, vel_ys[j])
            if vel_xs:
                ax.set_xlim(vel_xs[0], max(vel_xs[-1], vel_xs[0]+1))
                ym = max(abs(v) for v in vel_ys[j]) if vel_ys[j] else 1.0
                ax.set_ylim(-ym*1.3-1, ym*1.3+1)

        phase = "Drawing ✏️" if f["is_drawing"] else "Pen-lift 🚀"
        status_txt.set_text(
            f"Stroke {si+1}/{len(ordered_strokes)}  │  {phase}  │  "
            f"ω=[{vel[0]:+.1f}, {vel[1]:+.1f}, {vel[2]:+.1f}]°/s  │  "
            f"EE=({ee[0]:.2f},{ee[1]:.2f})  │  Frame {fi}/{len(all_frames)}")

        return link_lines + joint_dots + [ee_dot, lift_line, status_txt] + vel_lines + stroke_lines

    # ── Speed slider ──────────────────────────────────────────────────────────
    # 0.1× = 10× slower  |  1.0× = normal 20ms/frame  |  4.0× = fast
    # delay[0] is read on every frame — the slider updates it live via closure.
    from matplotlib.widgets import Slider
    sl_speed = Slider(
        ax_sp, "Speed ×",
        valmin=0.1, valmax=4.0,
        valinit=1.0, valstep=0.05,
        color="#3d5a80"
    )
    sl_speed.label.set_color("white")
    sl_speed.valtext.set_color("white")
    sl_speed.label.set_fontsize(8)

    delay = [BASE_INTERVAL / 1000.0]   # seconds per frame, mutable list for closure

    def on_speed(val):
        delay[0] = BASE_INTERVAL / 1000.0 / val   # e.g. 0.1× → 0.2s, 4× → 0.005s

    sl_speed.on_changed(on_speed)

    # ── Manual animation loop (plt.pause reads + executes Tk events each frame) ──
    # FuncAnimation's timer cannot be reliably adjusted at runtime in TkAgg.
    # plt.pause(delay[0]) waits the correct amount AND dispatches slider events,
    # so the next iteration picks up the updated delay immediately.
    plt.show(block=False)
    plt.pause(0.2)
    for fi in range(len(all_frames)):
        if not plt.fignum_exists(fig.number):
            break                      # window closed by user
        update(fi)
        fig.canvas.draw_idle()
        plt.pause(delay[0])

    # Keep window open after animation finishes
    if plt.fignum_exists(fig.number):
        plt.show(block=True)


# ─── Main entry ───────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Week 5 — Full Robot Drawing Pipeline                ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    root = tk.Tk(); root.withdraw()
    img_path = filedialog.askopenfilename(
        title="Select image to draw",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
    root.destroy()
    if not img_path:
        print("No image selected. Exiting."); return

    print(f"[1/5] Processing image: {os.path.basename(img_path)}")
    strokes_px, edges, img_bgr, step_px = process_image(img_path)
    print(f"      {len(strokes_px)} strokes  {sum(len(s) for s in strokes_px)} waypoints")

    print("[2/5] Transforming to arm workspace …")
    strokes_arm = strokes_to_arm(strokes_px, img_bgr.shape)

    print("[3/5] Ordering strokes (greedy nearest-neighbour) …")
    ordered = order_strokes(strokes_arm)
    total_lift = sum(np.linalg.norm(ordered[i+1][0]-ordered[i][-1])
                     for i in range(len(ordered)-1))
    print(f"      Total pen-lift distance: {total_lift:.2f} arm-units")

    print("[4/5] Saving strokes …")
    paths = save_strokes(strokes_px, img_path, step_px, edges)
    print(f"      JSON: {paths['json']}")

    print("[5/5] Pre-computing IK trajectories …")
    all_frames = compute_frames(ordered)
    dur = len(all_frames) / (1000/20)
    print(f"      {len(all_frames)} frames  ≈  {dur:.1f} s\n")

    # ── Preview first, then animation in a fresh event loop ──────────────────
    show_preview(img_bgr, edges, ordered)
    run_animation(all_frames, ordered)


if __name__ == "__main__":
    main()
