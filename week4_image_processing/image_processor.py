"""
Week 4 — Image Processing Core
================================
Converts an image into a set of ordered point sequences (strokes) for the
robotic arm to draw, and saves them for the full pipeline.

Pipeline
--------
1. Resize     → Normalise to ≤ 1500 px (consistent detection across resolutions)
2. Preprocess → CLAHE contrast boost + bilateral edge-preserving blur
3. Edge detect → Sigma/median Canny (better than Otsu for portraits)
4. Gap close  → Morphological close to join broken edge segments
5. Contour trace → cv2.findContours with CHAIN_APPROX_NONE (every pixel)
6. Filter     → Remove noise fragments shorter than 1% of image short side
7. Resample   → Uniformly space points along each stroke (arc-length param)
8. Save       → JSON (human-readable) + NPZ (fast numpy format) for the pipeline
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime


# ─── Working resolution ───────────────────────────────────────────────────────

_MAX_SIDE = 1200   # px — normalise very large images to this before detecting


def _resize_for_detection(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Downscale the image so its longest side ≤ _MAX_SIDE.

    Returns (resized_img, scale_factor).  scale_factor < 1 if downscaled.
    Normalising prevents the Canny sigma and morphological kernels from
    behaving differently on a 400 px vs a 4000 px image.
    """
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= _MAX_SIDE:
        return img_bgr, 1.0
    scale = _MAX_SIDE / longest
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


# ─── Step 2: Preprocessing ────────────────────────────────────────────────────

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    Grayscale → CLAHE → bilateral blur.

    CLAHE (clipLimit=2.0, tile 8×8): boosts local contrast so faint features
    in shadowed areas (jaw, eye sockets) are as visible as bright areas.

    Bilateral (d=7, σ=40): smooths flat skin/background regions while keeping
    hard edges sharp. Lower sigma than before (was 75) to preserve fine detail.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # d=7 is lighter than d=9 — preserves eyebrow/lip fine detail
    blurred = cv2.bilateralFilter(enhanced, d=7, sigmaColor=40, sigmaSpace=40)
    return blurred


# ─── Step 3: Edge Detection ───────────────────────────────────────────────────

def _sigma_canny_thresholds(preprocessed: np.ndarray,
                             sigma: float = 0.33) -> tuple[float, float]:
    """
    Compute Canny thresholds from the image's median intensity.

    The sigma method (Adrian Rosebrock, 2015) is empirically more reliable
    than Otsu for portraits and paintings:

        median = median(all pixel values)
        low    = max(0,   (1 − σ) × median)
        high   = min(255, (1 + σ) × median)

    Otsu finds the optimal threshold to SEPARATE two regions (foreground/bg),
    which gives a HIGH threshold that discards fine-detail edges inside faces.
    The median method gives a threshold around the typical pixel value, so
    edges at all contrast levels (strong jaw, faint eyebrow) are captured.

    σ = 0.33 → catches most edges; raise σ toward 0.5 for noisier images.
    """
    v    = np.median(preprocessed)
    low  = max(0,   int((1.0 - sigma) * v))
    high = min(255, int((1.0 + sigma) * v))
    return float(low), float(high)


def detect_edges(img_bgr: np.ndarray,
                 canny_sigma: float = 0.33) -> np.ndarray:
    """
    Full edge-detection pipeline working at normalised resolution.

    Steps:
    1. Resize to ≤ 1200 px (normalise)
    2. Preprocess (CLAHE + bilateral)
    3. Sigma/median Canny
    4. Morphological close: 3×3, 1 iteration — bridges tiny gaps
       (1 iter is enough now; 2 iter was over-thickening and merging close lines)
    5. Resize result back to original dimensions

    Returns a binary uint8 edge image at ORIGINAL resolution.
    """
    small, scale = _resize_for_detection(img_bgr)
    prep         = preprocess(small)
    low, high    = _sigma_canny_thresholds(prep, sigma=canny_sigma)
    edges_small  = cv2.Canny(prep, low, high)

    # 1 iteration, 3×3 — closes 1-pixel gaps without thickening edges
    kernel      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_small = cv2.morphologyEx(edges_small, cv2.MORPH_CLOSE, kernel,
                                   iterations=1)

    # Resize edges back to original image size if we downscaled
    if scale < 1.0:
        orig_h, orig_w = img_bgr.shape[:2]
        edges = cv2.resize(edges_small, (orig_w, orig_h),
                           interpolation=cv2.INTER_NEAREST)
    else:
        edges = edges_small

    return edges


# ─── Step 4-6: Contour extraction + filter + sort ─────────────────────────────

def _adaptive_min_length(img_shape: tuple) -> float:
    """
    Minimum contour arc-length to be kept as a stroke.

    Changed from 3% to 1% of shorter dimension — the previous 3% was
    cutting valid short strokes (eyebrows, mouth corners, nostril edges).
    Hard floor of 15 px removes isolated pixel-noise speckles.
    """
    shorter = min(img_shape[0], img_shape[1])
    return max(15.0, shorter * 0.01)   # 1% of short side, min 15 px


def extract_contours(image_path: str,
                     canny_sigma: float = 0.33) -> tuple[list, np.ndarray, np.ndarray]:
    """
    Full pipeline from file path to filtered, ordered pixel-dense strokes.

    Returns
    -------
    contours : list of np.ndarray  — each (N, 2) array of [x, y] at orig resolution
    edges    : np.ndarray          — binary edge image (for visualisation)
    img_bgr  : np.ndarray          — original image (for visualisation)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    edges = detect_edges(img_bgr, canny_sigma=canny_sigma)

    raw_contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    min_len = _adaptive_min_length(img_bgr.shape)
    formatted = []
    for cnt in raw_contours:
        arc = cv2.arcLength(cnt, closed=False)
        if arc >= min_len:
            pts = cnt.squeeze(axis=1)
            if pts.ndim == 1:
                pts = pts.reshape(1, 2)
            formatted.append(pts)

    # Longest strokes first — robot draws main features before fine details
    formatted.sort(key=lambda c: -len(c))
    return formatted, edges, img_bgr


# ─── Step 7: Adaptive uniform resampling ─────────────────────────────────────

def _adaptive_step(img_shape: tuple) -> float:
    """
    Target spacing between consecutive waypoints, in pixels of the original image.

    Scales with resolution: 1.5% of the shorter image dimension,
    clamped to [8, 30] px.  This keeps the total waypoint count proportional
    regardless of whether the source image is 400 px or 4000 px.
    """
    shorter = min(img_shape[0], img_shape[1])
    return float(np.clip(shorter * 0.015, 8.0, 30.0))


def space_points_wisely(contour: np.ndarray,
                         step_size: float = 10.0) -> np.ndarray:
    """
    Uniform arc-length resampling of a pixel-dense contour.

    The raw contour from findContours has points spaced 1–√2 px apart.
    This function resamples to exactly `step_size` px spacing via linear
    interpolation on cumulative arc-length, always preserving the endpoint.

    Returns (M, 2) float array.
    """
    if len(contour) < 2:
        return contour.astype(float)

    pts      = contour.astype(float)
    diffs    = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum      = np.insert(np.cumsum(seg_lens), 0, 0.0)
    total    = cum[-1]

    if total < step_size:
        return np.array([pts[0], pts[-1]])

    targets = np.arange(0.0, total, step_size)
    if targets[-1] < total:
        targets = np.append(targets, total)

    new_x = np.interp(targets, cum, pts[:, 0])
    new_y = np.interp(targets, cum, pts[:, 1])
    return np.column_stack((new_x, new_y))


def process_image(image_path: str,
                  canny_sigma: float = 0.33,
                  step_size: float | None = None) -> tuple[list, np.ndarray, np.ndarray, float]:
    """
    Convenience wrapper: full pipeline, returns everything for visualisation.

    Parameters
    ----------
    image_path   : path to image file
    canny_sigma  : Canny sensitivity — 0.33 is standard; raise to 0.5 for more edges
    step_size    : waypoint spacing in px; None → auto (adaptive to resolution)

    Returns
    -------
    (spaced_strokes, edges, img_bgr, step_size)
    """
    contours, edges, img_bgr = extract_contours(image_path, canny_sigma=canny_sigma)

    if step_size is None:
        step_size = _adaptive_step(img_bgr.shape)

    spaced = []
    for cnt in contours:
        s = space_points_wisely(cnt, step_size=step_size)
        if len(s) >= 2:
            spaced.append(s)

    return spaced, edges, img_bgr, step_size


# ─── Step 8: Save / Load strokes ─────────────────────────────────────────────

def save_strokes(strokes: list,
                 image_path: str,
                 step_size: float,
                 edges: np.ndarray,
                 output_dir: str | None = None) -> dict[str, str]:
    """
    Save processed strokes in three formats for the future pipeline.

    Formats
    -------
    <name>_strokes.json   — Human-readable metadata + all waypoints as nested lists.
                            Suitable for inspection, sharing, and loading in any language.
    <name>_strokes.npz    — Numpy archive.  Fast binary load in Python.
                            Contains an object array of (M,2) float arrays.
    <name>_edges.png      — The binary sketch image as a PNG.
                            Useful for debugging and documentation.

    JSON Schema
    -----------
    {
      "metadata": {
        "source_image" : str,
        "processed_at" : ISO timestamp,
        "image_size"   : [height, width],
        "step_size_px" : float,
        "num_strokes"  : int,
        "total_waypoints": int
      },
      "strokes": [
        [[x0,y0],[x1,y1],...],   ← stroke 0
        [[x0,y0],[x1,y1],...],   ← stroke 1
        ...
      ]
    }

    Returns dict of {"json": path, "npz": path, "edges_png": path}.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(image_path))

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    total_pts = sum(len(s) for s in strokes)
    h, w = edges.shape[:2]

    # ── JSON ──────────────────────────────────────────────────────────────────
    payload = {
        "metadata": {
            "source_image":    os.path.abspath(image_path),
            "processed_at":    datetime.now().isoformat(timespec="seconds"),
            "image_size":      [h, w],
            "step_size_px":    round(float(step_size), 3),
            "num_strokes":     len(strokes),
            "total_waypoints": total_pts,
        },
        "strokes": [s.tolist() for s in strokes],
    }
    json_path = os.path.join(output_dir, f"{base}_strokes.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # ── NPZ ───────────────────────────────────────────────────────────────────
    # numpy can't store ragged arrays as a uniform array; use object dtype
    stroke_arr = np.empty(len(strokes), dtype=object)
    for i, s in enumerate(strokes):
        stroke_arr[i] = s.astype(np.float32)

    npz_path = os.path.join(output_dir, f"{base}_strokes.npz")
    np.savez_compressed(
        npz_path,
        strokes   = stroke_arr,
        step_size = np.float32(step_size),
        image_size = np.array([h, w], dtype=np.int32),
    )

    # ── Edge PNG ──────────────────────────────────────────────────────────────
    edges_path = os.path.join(output_dir, f"{base}_edges.png")
    cv2.imwrite(edges_path, edges)

    return {"json": json_path, "npz": npz_path, "edges_png": edges_path}


def load_strokes(npz_path: str) -> tuple[list, float]:
    """
    Load strokes from a previously saved NPZ file.

    Returns
    -------
    strokes   : list of (M, 2) float32 arrays
    step_size : float — the spacing used when strokes were generated
    """
    data      = np.load(npz_path, allow_pickle=True)
    strokes   = list(data["strokes"])
    step_size = float(data["step_size"])
    return strokes, step_size
