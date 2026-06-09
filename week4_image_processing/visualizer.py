"""
Week 4 — Interactive Visualizer
================================
3-panel pipeline: Original Image → Sketch → Waypoints

Controls
--------
  Waypoint spacing slider : adjusts point density live
  Canny sensitivity slider: raise to detect more (fine) edges
  [Save strokes]          : saves JSON + NPZ + edge PNG to image's directory
  [Load new image]        : pick a new image without restarting

Run:
    python3 visualizer.py
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog, messagebox

from image_processor import (
    process_image, extract_contours,
    space_points_wisely, _adaptive_step, save_strokes,
)

_CMAP = plt.get_cmap("tab20")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _pick_file() -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    root.destroy()
    return path


def _draw_panels(ax1, ax2, ax3, img_bgr, edges, spaced_strokes, step_size):
    """Re-render all three panels into their axes."""
    for ax in (ax1, ax2, ax3):
        ax.cla()

    h, w    = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Panel 1 — Original
    ax1.imshow(img_rgb)
    ax1.set_title("Original Image", fontsize=11, fontweight="bold", pad=8,
                  color="white")
    ax1.axis("off")

    # Panel 2 — Sketch (black on white, like a pencil drawing)
    ax2.imshow(edges, cmap="gray_r")
    ax2.set_title("Sketch", fontsize=11, fontweight="bold", pad=8, color="white")
    ax2.axis("off")

    # Panel 3 — Waypoints
    total_pts = sum(len(s) for s in spaced_strokes)
    ax3.set_facecolor("white")
    ax3.set_xlim(0, w)
    ax3.set_ylim(h, 0)
    ax3.set_aspect("equal")
    ax3.axis("off")
    ax3.set_title(
        f"Waypoints  ({len(spaced_strokes)} strokes · {total_pts} pts · "
        f"{step_size:.1f} px spacing)",
        fontsize=10, fontweight="bold", pad=8, color="white"
    )
    for i, stroke in enumerate(spaced_strokes):
        ax3.plot(
            stroke[:, 0], stroke[:, 1],
            marker="o", markersize=1.8,
            linestyle="-", linewidth=0.6,
            color=_CMAP(i % 20),
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    file_path = _pick_file()
    if not file_path:
        print("No file selected. Exiting.")
        return

    # State shared across callbacks
    state = {
        "file_path":    file_path,
        "img_bgr":      None,
        "edges":        None,
        "contours":     None,   # pixel-dense, unsampled
        "strokes":      None,   # spaced strokes
        "step_size":    None,
        "canny_sigma":  0.33,
    }

    def _reload_image(path, sigma):
        print(f"\nProcessing: {path}  (σ={sigma:.2f})")
        contours, edges, img_bgr = extract_contours(path, canny_sigma=sigma)
        step = _adaptive_step(img_bgr.shape)
        spaced = [s for cnt in contours
                  for s in [space_points_wisely(cnt, step)] if len(s) >= 2]
        state.update(
            file_path=path, img_bgr=img_bgr, edges=edges,
            contours=contours, strokes=spaced, step_size=step,
        )
        print(f"  Strokes: {len(spaced)}   Waypoints: {sum(len(s) for s in spaced)}"
              f"   Step: {step:.1f} px   Canny low={int((1-sigma)*np.median(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)))}")
        return step

    init_step = _reload_image(file_path, state["canny_sigma"])

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(19, 8))
    fig.patch.set_facecolor("#13131f")
    fig.canvas.manager.set_window_title("Week 4 — Image to Points Visualizer")

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[20, 1.4],
        hspace=0.15, wspace=0.04,
        left=0.02, right=0.98, top=0.93, bottom=0.13,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")
            sp.set_linewidth(1.2)

    _draw_panels(ax1, ax2, ax3,
                 state["img_bgr"], state["edges"], state["strokes"], state["step_size"])

    # ── Sliders ───────────────────────────────────────────────────────────────
    ax_sp  = fig.add_axes([0.08, 0.055, 0.35, 0.022])
    ax_sig = fig.add_axes([0.08, 0.025, 0.35, 0.022])
    for a in (ax_sp, ax_sig):
        a.set_facecolor("#1e1e3a")

    sl_step = Slider(ax_sp,  "Spacing (px)",     4,  50,
                     valinit=init_step, valstep=1, color="#4a7fbe")
    sl_sig  = Slider(ax_sig, "Canny sensitivity", 0.1, 0.7,
                     valinit=0.33, valstep=0.01, color="#6b5ea8")

    for sl in (sl_step, sl_sig):
        sl.label.set_color("white")
        sl.valtext.set_color("white")

    # ── Buttons ───────────────────────────────────────────────────────────────
    ax_save = fig.add_axes([0.52, 0.025, 0.17, 0.048])
    ax_new  = fig.add_axes([0.72, 0.025, 0.17, 0.048])

    btn_save = Button(ax_save, "💾  Save strokes",   color="#2a5f3e", hovercolor="#3d8a5c")
    btn_new  = Button(ax_new,  "📂  Load new image", color="#3d3d72", hovercolor="#5a5aab")

    for btn in (btn_save, btn_new):
        btn.label.set_color("white")
        btn.label.set_fontsize(9)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _redraw():
        _draw_panels(ax1, ax2, ax3,
                     state["img_bgr"], state["edges"],
                     state["strokes"], state["step_size"])
        fig.canvas.draw_idle()

    def on_step_change(val):
        new_step = float(sl_step.val)
        state["step_size"] = new_step
        state["strokes"]   = [
            s for cnt in state["contours"]
            for s in [space_points_wisely(cnt, new_step)] if len(s) >= 2
        ]
        _redraw()

    def on_sigma_change(val):
        new_sigma = float(sl_sig.val)
        state["canny_sigma"] = new_sigma
        _reload_image(state["file_path"], new_sigma)
        sl_step.set_val(state["step_size"])   # reset spacing to auto
        _redraw()

    def on_save(event):
        paths = save_strokes(
            strokes    = state["strokes"],
            image_path = state["file_path"],
            step_size  = state["step_size"],
            edges      = state["edges"],
        )
        print("\n── Saved ──")
        for fmt, p in paths.items():
            print(f"  {fmt:12s}: {p}")
        print()
        # Show a small confirmation balloon in tkinter
        root = tk.Tk(); root.withdraw()
        messagebox.showinfo("Strokes saved",
            f"Saved to:\n\n"
            f"JSON : {paths['json']}\n"
            f"NPZ  : {paths['npz']}\n"
            f"Edges: {paths['edges_png']}")
        root.destroy()

    def on_new_image(event):
        new_path = _pick_file()
        if not new_path:
            return
        step = _reload_image(new_path, state["canny_sigma"])
        sl_step.set_val(step)
        _redraw()

    sl_step.on_changed(on_step_change)
    sl_sig.on_changed(on_sigma_change)
    btn_save.on_clicked(on_save)
    btn_new.on_clicked(on_new_image)

    plt.show()


if __name__ == "__main__":
    main()
