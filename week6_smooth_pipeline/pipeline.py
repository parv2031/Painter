"""
Week 6 — Smooth Drawing Pipeline
==================================
Key upgrade over Week 5: B-spline path smoothing eliminates arm jitter.
Instead of straight-line segments between sparse waypoints, a scipy B-spline
is fitted through each stroke and resampled at 4× density before IK planning.
This gives near-zero direction changes between consecutive segments → smooth,
jitter-free joint velocity profiles.

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

try:
    from scipy.interpolate import splprep, splev
    _SCIPY = True
except Exception:      # catches ImportError AND numpy ABI ValueError
    _SCIPY = False

from forward_kinematics import ThreeLinkArm
from trajectory import plan_linear_trajectory
from inverse_kinematics import ik_analytical_auto
from image_processor import process_image, save_strokes, space_points_wisely

ARM        = ThreeLinkArm(link_lengths=(3.0, 2.5, 1.5))
DRAW_SPEED = 0.8     # arm-units/s — very slow, smooth drawing
LIFT_SPEED = 8.0
ARC_STEP   = 0.015   # arm-units between consecutive EE samples on the smooth curve
CANVAS_R   = 5.0
VEL_WINDOW = 200
JERK_THRESHOLD = 600.0
BASE_INTERVAL  = 20  # ms per frame at 1× speed


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


# ─── Spline path smoothing (Week 6 addition) ────────────────────────────────────
def _chaikin(pts: np.ndarray, iterations: int = 4) -> np.ndarray:
    """Chaikin corner-cutting: each pass doubles the number of points."""
    p = pts.copy()
    for _ in range(iterations):
        new_p = [p[0]]
        for i in range(len(p) - 1):
            new_p.append(0.75*p[i] + 0.25*p[i+1])
            new_p.append(0.25*p[i] + 0.75*p[i+1])
        new_p.append(p[-1])
        p = np.array(new_p)
    return p


def smooth_stroke(stroke_arm: np.ndarray,
                  smooth_factor: float = 1.0,
                  oversample: int = 4) -> np.ndarray:
    """
    Fit a B-spline through the waypoints of one stroke and resample at
    ``oversample`` × density.  More points + smoother direction changes
    → much smaller joint-velocity steps → jitter-free arm motion.

    Parameters
    ----------
    stroke_arm    : (N, 2) EE waypoints in arm coordinates
    smooth_factor : spline smoothing (0 = exact interpolation, 5 = heavy)
    oversample    : output point density relative to input (default 4×)

    Falls back to Chaikin corner-cutting if scipy is unavailable.
    """
    pts = stroke_arm
    N   = len(pts)
    if N < 2:
        return pts
    if N == 2:                              # single segment: just subdivide
        t = np.linspace(0, 1, oversample + 1)
        return (1 - t)[:, None] * pts[0] + t[:, None] * pts[1]

    # Remove consecutive duplicate points (splprep rejects them)
    mask   = np.any(np.diff(pts, axis=0) != 0, axis=1)
    pts_c  = np.vstack([pts[:-1][mask], pts[-1:]])
    if len(pts_c) < 3:
        return stroke_arm

    n_out = max(N * oversample, 10)

    if _SCIPY:
        try:
            k   = min(3, len(pts_c) - 1)          # spline degree
            s   = smooth_factor * len(pts_c)       # smoothing parameter
            tck, _ = splprep([pts_c[:, 0], pts_c[:, 1]], s=s, k=k)
            u_fine = np.linspace(0, 1, n_out)
            x, y   = splev(u_fine, tck)
            return np.column_stack([x, y])
        except Exception:
            pass    # fall through to Chaikin

    # Chaikin fallback: ~4 iterations ≈ 16× density, then subsample
    smooth = _chaikin(pts_c, iterations=4)
    t  = np.cumsum(np.r_[0, np.linalg.norm(np.diff(smooth, axis=0), axis=1)])
    t /= t[-1]
    tf = np.linspace(0, 1, n_out)
    return np.column_stack([np.interp(tf, t, smooth[:, 0]),
                             np.interp(tf, t, smooth[:, 1])])


def smooth_all_strokes(ordered: list,
                       smooth_factor: float = 1.0,
                       oversample: int = 4) -> list:
    """Apply smooth_stroke to every stroke in the ordered list."""
    return [smooth_stroke(s, smooth_factor, oversample) for s in ordered]



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


# ─── Week 6: smooth-curve tracer ───────────────────────────────────────────────────

def _arc_resample(pts: np.ndarray, ds: float) -> np.ndarray:
    """
    Resample a polyline at uniform arc-length intervals of ``ds``.
    Returns a dense array of EE positions that the arm will follow
    at constant speed without discrete waypoint hops.
    """
    diffs   = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    arc     = np.r_[0, np.cumsum(seg_len)]
    total   = arc[-1]
    if total < ds:
        return pts
    t_uni = np.arange(0, total + ds * 0.5, ds)   # include endpoint
    t_uni = t_uni[t_uni <= total]
    x = np.interp(t_uni, arc, pts[:, 0])
    y = np.interp(t_uni, arc, pts[:, 1])
    return np.column_stack([x, y])


def _stroke_frames_smooth(stroke_arm: np.ndarray,
                           prev_angles: np.ndarray,
                           stroke_idx: int) -> tuple:
    """
    Trace one stroke as a single smooth continuous curve (Week 6 approach).

    Steps
    -----
    1. Smooth the raw contour waypoints with Chaikin / B-spline.
    2. Resample at uniform arc-length intervals (ARC_STEP).
    3. For each consecutive pair of dense samples, plan a tiny linear
       segment at the slow DRAW_SPEED.  Because the steps are tiny and
       the curve is smooth, direction changes between segments are
       near-zero → joint velocities are smooth everywhere.

    Returns (frames, final_angles).
    """
    # Step 1 — smooth
    smooth  = smooth_stroke(stroke_arm, smooth_factor=1.0, oversample=8)
    # Step 2 — arc-length resample at ARC_STEP
    dense   = _arc_resample(smooth, ARC_STEP)
    if len(dense) < 2:
        return [], prev_angles

    # Step 3 — compute IK directly at each dense sample
    #   No plan_linear_trajectory calls → no zero-velocity seam artifacts.
    #   ik_analytical_auto picks a consistent phi for nearby points.
    dt     = BASE_INTERVAL / 1000.0
    frames = []
    cur    = prev_angles.copy()

    for k, pt in enumerate(dense):
        ik = ik_analytical_auto(ARM, float(pt[0]), float(pt[1]))
        ang = np.array(ik["angles"]) if ik["success"] else cur.copy()

        if k == 0:
            vel = np.zeros(3)
        else:
            da  = ang - cur
            da  = (da + np.pi) % (2 * np.pi) - np.pi   # wrap ±π
            vel = da / dt

        frames.append({
            "angles":     ang.copy(),
            "vel_deg":    np.degrees(vel),
            "ee":         pt.copy(),
            "is_drawing": True,
            "stroke_idx": stroke_idx,
        })
        cur = ang

    # Step 4 — despike + smooth, then hard velocity clamp
    #   spike_thresh=10°/frame, sigma=8 → removes IK branch-switch spikes
    #   _clamp_velocities → guarantees max |ω| ≤ 200 deg/s always
    frames = _despike_and_smooth(frames, spike_thresh_deg=10.0, sigma=8.0)
    frames = _clamp_velocities(frames, max_vel_deg=200.0)
    final  = frames[-1]["angles"] if frames else cur
    return frames, final


def _clamp_velocities(frames, max_vel_deg: float = 200.0) -> list:
    """
    Hard-cap joint angular velocity at ``max_vel_deg`` deg/s.

    For each frame, if any joint would move faster than max_vel_deg,
    ALL joints are scaled by the same factor so the fastest joint
    stays exactly at max_vel_deg.  This preserves coordinated motion
    while guaranteeing the velocity never exceeds the limit.
    """
    dt     = BASE_INTERVAL / 1000.0
    max_da = np.radians(max_vel_deg) * dt    # max angle change per frame (rad)

    angles = [np.array(f["angles"]) for f in frames]

    for i in range(1, len(frames)):
        da       = angles[i] - angles[i - 1]
        da       = (da + np.pi) % (2 * np.pi) - np.pi   # wrap
        peak     = np.abs(da).max()

        if peak > max_da:
            scale    = max_da / peak
            angles[i] = angles[i - 1] + da * scale

        frames[i]["angles"]  = angles[i].copy()
        actual_da            = angles[i] - angles[i - 1]
        actual_da            = (actual_da + np.pi) % (2 * np.pi) - np.pi
        frames[i]["vel_deg"] = np.degrees(actual_da / dt)

    return frames


def _despike_and_smooth(frames,
                         spike_thresh_deg: float = 25.0,
                         sigma: float = 4.0) -> list:
    """
    Remove IK-discontinuity velocity spikes from a frame sequence.

    Algorithm
    ---------
    1. Extract the (N, 3) joint-angle array from the frames.
    2. Compute frame-to-frame angle differences (wrapped to ±π).
    3. For any step where |Δθ_j| > spike_thresh for any joint j,
       mark the surrounding window as "bad" and linearly interpolate
       angles over it using the last good values before and after.
    4. Apply a Gaussian convolution (width σ frames) to the entire
       angle sequence — this distributes any residual sharp change
       over several frames, keeping velocity bounded.
    5. Recompute vel_deg from the smoothed angle differences.

    Parameters
    ----------
    spike_thresh_deg : angle step in degrees that triggers spike detection
    sigma            : Gaussian half-width in frames
    """
    N  = len(frames)
    dt = BASE_INTERVAL / 1000.0      # seconds per frame
    if N < 3:
        return frames

    angles = np.array([f["angles"] for f in frames])   # (N, 3)
    thresh = np.radians(spike_thresh_deg)

    # ── Step 1: detect & interpolate over spikes ─────────────────────────
    for j in range(3):
        col  = angles[:, j]
        da   = np.diff(col)
        da   = (da + np.pi) % (2 * np.pi) - np.pi       # wrap to ±π
        bad_steps = np.abs(da) > thresh                  # shape (N-1,)

        if not np.any(bad_steps):
            continue

        # Expand each bad step into a window of ±sigma frames
        bad_frames = np.zeros(N, dtype=bool)
        for i in np.where(bad_steps)[0]:
            lo = max(0, i - int(sigma))
            hi = min(N, i + int(sigma) + 2)
            bad_frames[lo:hi] = True

        good = np.where(~bad_frames)[0]
        if len(good) < 2:
            continue                    # can't interpolate — leave as-is

        # Linear interpolation over bad regions
        angles[:, j] = np.interp(np.arange(N), good, col[good])

    # ── Step 2: Gaussian smooth ───────────────────────────────────────────
    r    = int(3 * sigma + 0.5)
    kx   = np.arange(-r, r + 1, dtype=float)
    kern = np.exp(-0.5 * (kx / sigma) ** 2)
    kern /= kern.sum()

    for j in range(3):
        col    = angles[:, j]
        padded = np.pad(col, r, mode="edge")
        smoothed = np.convolve(padded, kern, mode="valid")
        angles[:, j] = smoothed[:N]

    # ── Step 3: write back angles + recompute velocities ─────────────────
    for i, f in enumerate(frames):
        f["angles"] = angles[i].copy()
        if i > 0:
            da           = angles[i] - angles[i - 1]
            da           = (da + np.pi) % (2 * np.pi) - np.pi
            f["vel_deg"] = np.degrees(da / dt)
        else:
            f["vel_deg"] = np.zeros(3)

    return frames


def _transition_frames_smooth(start_ee, end_ee, prev_angles, stroke_idx):
    """Transition between strokes using same smoothing logic."""
    pts = np.array([start_ee, end_ee])
    dense = _arc_resample(pts, ARC_STEP)
    
    # We must treat this as a "pen-up" move (is_drawing=False)
    dt = BASE_INTERVAL / 1000.0
    frames = []
    cur = prev_angles.copy()
    for pt in dense:
        ik = ik_analytical_auto(ARM, float(pt[0]), float(pt[1]))
        ang = np.array(ik["angles"]) if ik["success"] else cur.copy()
        
        frames.append({
            "angles": ang.copy(),
            "vel_deg": np.zeros(3), # Will be recomputed
            "ee": pt.copy(),
            "is_drawing": False,
            "stroke_idx": stroke_idx,
        })
        cur = ang
    
    frames = _despike_and_smooth(frames, spike_thresh_deg=10.0, sigma=8.0)
    frames = _clamp_velocities(frames, max_vel_deg=200.0)
    final = frames[-1]["angles"] if frames else prev_angles
    return frames, final


def compute_frames(ordered_strokes):
    """
    Week 6: each stroke is traced as a smooth continuous curve.
    No waypoint-to-waypoint hops — the arm follows a B-spline / Chaikin
    smooth path at a slow, constant EE speed (DRAW_SPEED = 1.5 arm-units/s).
    """
    p0   = ordered_strokes[0][0]
    boot = ik_analytical_auto(ARM, float(p0[0]), float(p0[1]))
    prev = boot["angles"] if boot["success"] else np.zeros(3)

    all_frames = []
    total = len(ordered_strokes)
    print(f"  Smooth-tracing {total} strokes …")

    for si, stroke in enumerate(ordered_strokes):
        # Pen-up transition to the start of the next stroke
        # — uses the same slow direct-IK + despike + clamp pipeline as drawing
        if si > 0 and all_frames:
            lf, prev = _transition_frames_smooth(
                all_frames[-1]["ee"], stroke[0], prev, stroke_idx=si)
            all_frames.extend(lf)

        # Smooth continuous draw of the stroke curve
        sf, prev = _stroke_frames_smooth(stroke, prev, si)
        all_frames.extend(sf)

        pct = 100 * (si + 1) / total
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  [{bar}] {pct:5.1f}%  stroke {si+1}/{total}  (+{len(sf)} frames)",
              end="\r")

    print(f"\n  Done — {len(all_frames)} total frames")

    # Return to home — slow and smooth
    home = np.array([_MARGIN + 0.3, _MARGIN + 0.3])
    if all_frames:
        hf, _ = _transition_frames_smooth(
            all_frames[-1]["ee"], home, prev,
            stroke_idx=len(ordered_strokes))
        all_frames.extend(hf)
        print(f"  + {len(hf)} return-to-home frames")

    # ── Global smooth pass ────────────────────────────────────────────────
    # Each stroke and transition was despiked individually, but their
    # BOUNDARIES (last frame of stroke k → first frame of transition k+1)
    # are not yet smoothed.  One final pass over the full sequence removes
    # every remaining seam and hard-caps velocity at 100 deg/s globally.
    print("  Applying global smooth pass …", end="")
    all_frames = _despike_and_smooth(all_frames, spike_thresh_deg=8.0, sigma=6.0)
    all_frames = _clamp_velocities(all_frames, max_vel_deg=100.0)
    print(f"  max |ω| ≤ 100 deg/s guaranteed")

    return all_frames



# ─── # ─── Interactive Preview Tuner ────────────────────────────────────────────────────

def interactive_preview(img_path: str) -> tuple:
    """
    Opens a live-tuning window with 7 sliders.  The user adjusts parameters
    until the sketch + waypoints look right, then clicks “Start Simulation”.

    Sliders
    -------
    CLAHE clip   — local contrast enhancement (higher = more contrast)
    Bilateral σ  — edge-preserving blur (higher = smoother / fewer fine edges)
    Canny σ      — edge sensitivity (higher = more / finer edges)
    Gamma        — brightness correction (<1 darker, >1 brighter)
    Morph close  — gap-closing iterations (higher = fewer broken edges)
    Min stroke % — shortest contour kept as % of image size
    Spacing (px) — waypoint density along each stroke

    Returns
    -------
    (strokes_px, edges, img_bgr, step_px) with the settings at “Start”.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    def_step = float(np.clip(min(H, W) * 0.015, 8.0, 30.0))

    state = {"spaced": [], "edges": None, "step": def_step, "smooth": 1.0}

    # ── Figure layout ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 11))
    fig.patch.set_facecolor("#0d0d1a")
    fig.canvas.manager.set_window_title(
        "Interactive Tuner — adjust sliders, then click Start Simulation")

    ax_orig = fig.add_axes([0.01,  0.42, 0.30, 0.54])
    ax_sk   = fig.add_axes([0.345, 0.42, 0.30, 0.54])
    ax_wp   = fig.add_axes([0.68,  0.42, 0.30, 0.54])
    for ax in (ax_orig, ax_sk, ax_wp):
        ax.set_facecolor("#0a0a18")
    ax_orig.imshow(img_rgb); ax_orig.axis("off")
    ax_orig.set_title("Original Image", color="white", fontsize=10, pad=4)

    # Slider positions: 2 columns, 4 rows
    SH, SW = 0.022, 0.37
    def _sl_ax(col, row):   # col 0=left, 1=right;  row 0..3 top..bottom
        x = 0.05 + col * 0.50
        y = 0.31 - row * 0.065
        return fig.add_axes([x, y, SW, SH])

    from matplotlib.widgets import Slider, Button
    _sc = "#3d5a80"
    def _sl(ax, lbl, lo, hi, init, step=None):
        kw = {} if step is None else {"valstep": step}
        s = Slider(ax, lbl, lo, hi, valinit=init, color=_sc, **kw)
        s.label.set_color("white"); s.valtext.set_color("white")
        s.label.set_fontsize(8);   s.valtext.set_fontsize(8)
        return s

    sl_clahe   = _sl(_sl_ax(0,0), "CLAHE clip",      0.5, 5.0,  2.0,  0.1)
    sl_bil     = _sl(_sl_ax(1,0), "Bilateral σ",      10,  80,   40,   1)
    sl_canny   = _sl(_sl_ax(0,1), "Canny σ",          0.1, 0.7,  0.33, 0.01)
    sl_gamma   = _sl(_sl_ax(1,1), "Gamma",            0.3, 2.5,  1.0,  0.05)
    sl_morph   = _sl(_sl_ax(0,2), "Morph close",      0,   4,    1,    1)
    sl_minlen  = _sl(_sl_ax(1,2), "Min stroke %",     0.2, 3.0,  1.0,  0.1)
    sl_spacing = _sl(_sl_ax(0,3), "Waypoint spacing",  4,   50,   def_step, 1)
    sl_smooth  = _sl(_sl_ax(1,3), "Smoothing",         0.0, 5.0,  1.0,  0.1)  # NEW

    ax_btn = fig.add_axes([0.60, 0.04, 0.32, 0.07])
    btn_start = Button(ax_btn, "▶  Start Simulation",
                       color="#1a4a2e", hovercolor="#2d7a4e")
    btn_start.label.set_color("white"); btn_start.label.set_fontsize(10)

    # ── Processing callback ─────────────────────────────────────────────────
    def _reprocess(_=None):
        clahe_clip  = float(sl_clahe.val)
        bil_sigma   = int(sl_bil.val)
        canny_sigma = float(sl_canny.val)
        gamma       = float(sl_gamma.val)
        morph_iter  = int(round(sl_morph.val))
        min_pct     = float(sl_minlen.val) / 100.0
        step        = float(sl_spacing.val)
        smooth_fac  = float(sl_smooth.val)       # NEW

        # --- Preprocessing ---
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if abs(gamma - 1.0) > 0.02:          # gamma correction
            lut = np.array([(i/255.0)**(1.0/gamma)*255
                            for i in range(256)], dtype=np.uint8)
            gray = cv2.LUT(gray, lut)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred  = cv2.bilateralFilter(enhanced, d=7,
                                       sigmaColor=bil_sigma, sigmaSpace=bil_sigma)

        # --- Canny at normalised resolution ---
        MAX_SIDE = 1200
        h, w = blurred.shape
        sc = min(1.0, MAX_SIDE / max(h, w))
        small = cv2.resize(blurred, (int(w*sc), int(h*sc)), cv2.INTER_AREA) if sc < 1 else blurred
        median = np.median(small)
        lo = max(0,   int((1 - canny_sigma) * median))
        hi = min(255, int((1 + canny_sigma) * median))
        esmall = cv2.Canny(small, lo, hi)
        if morph_iter > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            esmall = cv2.morphologyEx(esmall, cv2.MORPH_CLOSE, k, iterations=morph_iter)
        edges = cv2.resize(esmall, (w, h), cv2.INTER_NEAREST) if sc < 1 else esmall

        # --- Contours ---
        raw, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        min_len = max(15.0, min(h, w) * min_pct)
        contours = []
        for cnt in raw:
            if cv2.arcLength(cnt, closed=False) >= min_len:
                pts = cnt.squeeze(axis=1)
                contours.append(pts.reshape(1,2) if pts.ndim==1 else pts)
        contours.sort(key=lambda c: -len(c))
        spaced = [s for cnt in contours
                  for s in [space_points_wisely(cnt, step)] if len(s) >= 2]

        # --- Update panels ---
        ax_sk.cla(); ax_sk.set_facecolor("#0a0a18")
        ax_sk.imshow(edges, cmap="gray_r")
        ax_sk.set_title(f"Sketch  |  {len(spaced)} strokes",
                        color="white", fontsize=10, pad=4)
        ax_sk.axis("off")

        ax_wp.cla(); ax_wp.set_facecolor("#0a0a18")
        ordered_arm = order_strokes(strokes_to_arm(spaced, (h, w, 3)))
        # Show smooth path in preview (Week 6)
        smooth_arm  = smooth_all_strokes(ordered_arm, smooth_fac)
        total_pts   = sum(len(s) for s in smooth_arm)
        cmap = plt.get_cmap("tab20")
        for i, s in enumerate(smooth_arm):
            ax_wp.plot(s[:, 0], s[:, 1], "-", lw=0.7, color=cmap(i % 20))
        ax_wp.plot(0, 0, "o", color="#6060ff", ms=6)
        ax_wp.set_xlim(-0.2, _MARGIN + _BOX + 0.3)
        ax_wp.set_ylim(-0.2, _MARGIN + _BOX + 0.3)
        ax_wp.set_aspect("equal"); ax_wp.axis("off")
        ax_wp.set_title(f"Smooth path (arm space)  |  {total_pts} pts",
                        color="white", fontsize=10, pad=4)

        state["spaced"] = spaced
        state["edges"]  = edges
        state["step"]   = step
        state["smooth"] = smooth_fac
        fig.canvas.draw_idle()

    for sl in (sl_clahe, sl_bil, sl_canny, sl_gamma, sl_morph, sl_minlen,
               sl_spacing, sl_smooth):           # 8 sliders
        sl.on_changed(_reprocess)

    btn_start.on_clicked(lambda _: plt.close(fig))

    _reprocess()                   # initial render
    plt.show(block=True)           # blocks until Start is clicked (closes fig)

    return state["spaced"], state["edges"], img_bgr, state["step"], state["smooth"]



# ─── FK helper ────────────────────────────────────────────────────────────────
def fk_joints(angles):
    """Return (J0, J1, J2, J3) joint positions — uses joint_positions array."""
    res = ARM.forward_kinematics(angles)
    pts = res["joint_positions"]   # shape (4, 2): base, j1, j2, EE
    return pts[0], pts[1], pts[2], pts[3]


# ─── Main animation ───────────────────────────────────────────────────────────
def run_animation(all_frames, ordered_strokes):
    fig = plt.figure(figsize=(17, 9.5))
    fig.patch.set_facecolor("#0d0d1a")
    fig.canvas.manager.set_window_title("Week 6 — Smooth Drawing Simulation")

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

    # Canvas — white background, upper-right quadrant
    # NOTE: axis("off") suppresses the facecolor in many matplotlib versions.
    # Use manual spine/tick hiding instead — this reliably keeps the white background.
    ax_c.set_facecolor("white")
    ax_c.set_xlim(-0.8, ARM.max_reach + 0.5)
    ax_c.set_ylim(-0.8, ARM.max_reach + 0.5)
    ax_c.set_aspect("equal")
    ax_c.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    ax_c.set_title("Drawing Simulation  (arm base ─●, canvas = upper-right quadrant)",
                   color="white", fontsize=11, pad=6)

    # Reference lines — light grey so visible on white canvas
    ax_c.axhline(0, color="#aaaacc", lw=0.8, zorder=0)
    ax_c.axvline(0, color="#aaaacc", lw=0.8, zorder=0)

    # Workspace arc (upper-right quarter)
    th_q = np.linspace(0, np.pi / 2, 180)
    ax_c.plot(ARM.max_reach * np.cos(th_q), ARM.max_reach * np.sin(th_q),
              color="#aaaacc", lw=1.2, ls="--", zorder=0)

    # Drawable box corners
    bx = [_MARGIN, _MARGIN+_BOX, _MARGIN+_BOX, _MARGIN, _MARGIN]
    by = [_MARGIN, _MARGIN,      _MARGIN+_BOX,  _MARGIN+_BOX, _MARGIN]
    ax_c.plot(bx, by, color="#ccccdd", lw=0.8, ls=":", zorder=0)

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
    joint_dots = [ax_c.plot([], [], "o", color="#222222", ms=5, zorder=6)[0] for _ in range(4)]
    ee_dot     =  ax_c.plot([], [], "o", color="#cc0000", ms=6, zorder=7)[0]
    stroke_lines = [ax_c.plot([], [], color="#111111", lw=1.0, alpha=1.0, zorder=3)[0]
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
    print("╔" + "═"*54 + "╗")
    print("║  Week 6 — Smooth Robot Drawing Pipeline              ║")
    print("╚" + "═"*54 + "╝\n")
    print("  Scipy B-spline smoothing eliminates arm jitter.")
    print("  Each stroke is resampled at 4× density on a smooth")
    print("  curve before IK planning.\n")

    root = tk.Tk(); root.withdraw()
    img_path = filedialog.askopenfilename(
        title="Select image to draw",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
    root.destroy()
    if not img_path:
        print("No image selected. Exiting."); return

    # [1] Interactive tuner (now returns smooth_factor too)
    print(f"[1/6] Opening interactive tuner for: {os.path.basename(img_path)}")
    print("      Adjust sliders (incl. Smoothing), then click ▶ Start Simulation")
    strokes_px, edges, img_bgr, step_px, smooth_factor = interactive_preview(img_path)
    print(f"      Tuning done: {len(strokes_px)} strokes  "
          f"{sum(len(s) for s in strokes_px)} waypoints  "
          f"step={step_px:.1f}px  smooth={smooth_factor:.1f}")

    # [2] Transform pixel strokes → arm workspace
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

    run_animation(all_frames, ordered)


if __name__ == "__main__":
    main()
