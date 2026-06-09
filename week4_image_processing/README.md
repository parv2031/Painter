# Week 4 — Image Processing & Edge Extraction

**Project:** Bob Ross without ROS — Robotics Society Summer Project  
**Mentors:** Anjaneya and Parv  
**Milestone:** Extracting continuous point trajectories from images

---

## Files

| File | Purpose |
|------|---------|
| `image_processor.py` | Core module — uses OpenCV to find edges, extract ordered point paths, and smartly space them using interpolation |
| `demo_image.py` | Headless demo script — extracts points and saves a plot without GUI |
| `visualizer.py` | Interactive GUI — prompts user for an image, processes it, and displays edges alongside wisely-spaced points |

---

## How to Run

```bash
cd week4_image_processing

# Run the interactive visualizer (asks for an image upload)
python3 visualizer.py
```

*Note: Ensure you have `opencv-python` and `matplotlib` installed.*

---

## Workflow: From Image to Points

To paint an image with the robotic arm, we first need to convert the image into a set of continuous trajectories (strokes) that the arm can follow.

### 1. Grayscale & Blur
We first convert the input image to grayscale, and optionally apply a small Gaussian blur. This helps reduce pixel noise which could result in jagged or disconnected edge segments.

### 2. Edge Detection
We use the **Canny Edge Detector**. This algorithm finds regions in the image with high intensity gradients (sharp transitions from light to dark). The output is a binary image where white pixels represent edges.

### 3. Extracting Continuous Points
To feed points to the trajectory planner developed in Week 3, we cannot just use a raw cloud of edge pixels. We need **ordered paths**.

We use `cv2.findContours` on the binary edge image. 
- By using the `cv2.CHAIN_APPROX_NONE` flag, OpenCV stores **every single pixel** along the contour. 
- This guarantees that the points are perfectly ordered and contiguous, leaving no ambiguity about which points are connected to form a stroke.

### 4. Spacing Points Wisely (Uniform Resampling)
Using every single pixel creates far too many points for a robotic arm to process efficiently, and can lead to jitter. We need to reduce the points while maintaining the shape.

The `space_points_wisely()` function achieves this through **uniform linear interpolation**:
1. It calculates the cumulative distance (arc-length) along the entire pixel-dense contour.
2. It generates perfectly spaced target distances (e.g., every 5.0 pixels).
3. It interpolates new `[x, y]` coordinates at these exact distances.

This removes dense clusters of points on straight lines but naturally preserves corners by sampling at consistent distances. This is mathematically the most stable way to feed waypoints to the Week 3 constant-speed trajectory planner.

### Output Structure
The result is a list of independent strokes. Each stroke is a Numpy array of shape `(M, 2)` containing the `[x, y]` coordinates spaced uniformly by the target step size. These strokes can then be fed sequentially to the robot.
