import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from image_processor import extract_contours, space_points_wisely
import sys

def main():
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()
    
    print("Please select an image file from the dialog window...")
    file_path = filedialog.askopenfilename(
        title="Select an Image to Convert to Points",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not file_path:
        print("No file selected. Exiting.")
        return
        
    print(f"\nLoading {file_path}...")
    try:
        contours, edges = extract_contours(file_path, canny_thresh1=100, canny_thresh2=200)
    except Exception as e:
        print(f"Error processing image: {e}")
        return
        
    print(f"Found {len(contours)} raw contours.")
    
    # Space points wisely
    # A step size of 4.0 or 5.0 pixels usually gives a great balance between 
    # capturing the shape and minimizing the number of points for the robot.
    step_size = 5.0 
    spaced_contours = []
    
    for cnt in contours:
        # Ignore tiny specks of noise
        if len(cnt) < 5:
            continue
            
        # Downsample/interpolate to uniform spacing
        spaced_cnt = space_points_wisely(cnt, step_size=step_size)
        
        # Only keep it if it's still a valid line (at least 2 points)
        if len(spaced_cnt) >= 2:
            spaced_contours.append(spaced_cnt)
        
    print(f"Processed into {len(spaced_contours)} strokes (points evenly spaced by {step_size}px).")
    
    # ─── Visualization ────────────────────────────────────────────────────────
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.canvas.manager.set_window_title("Week 4 — Image to Points Visualizer")
    
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    
    # Panel 1: Original Image
    if img_rgb is not None:
        ax1.imshow(img_rgb)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Panel 2: Sketch (Extracted Edges)
    # Using gray_r to show black lines on a white background, matching the reference
    ax2.imshow(edges, cmap='gray_r')
    ax2.set_title("Sketch")
    ax2.axis('off')
    
    # Panel 3: Waypoints
    ax3.set_title("Waypoints")
    ax3.set_aspect('equal')
    # Set background to white and bounds to match image
    ax3.set_facecolor('white')
    if img is not None:
        h, w = img.shape[:2]
        ax3.set_xlim(0, w)
        ax3.set_ylim(h, 0)
    else:
        ax3.invert_yaxis()
    
    # We use plt.get_cmap for compatibility
    cmap = plt.get_cmap('tab20')
    total_pts = 0
    
    # Plot each stroke (colored differently, with points and connecting lines)
    for i, cnt in enumerate(spaced_contours):
        ax3.plot(cnt[:, 0], cnt[:, 1], 
                 marker='o', markersize=2.5, 
                 linestyle='-', linewidth=0.8, 
                 color=cmap(i % 20))
        total_pts += len(cnt)
        
    print(f"Total points optimised for drawing: {total_pts}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
