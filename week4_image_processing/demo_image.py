import matplotlib.pyplot as plt
import numpy as np
from image_processor import extract_contours
import sys

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test_apple.jpg"
        
    print(f"Extracting edges from {image_path}...")
    
    try:
        contours, edge_img = extract_contours(image_path)
    except FileNotFoundError as e:
        print(e)
        return
        
    print(f"Found {len(contours)} distinct continuous paths (contours).")
    
    # Display the result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: Raw Edges
    ax1.imshow(edge_img, cmap='gray')
    ax1.set_title("Canny Edges")
    ax1.axis('off')
    
    # Plot 2: Extracted Contours as continuous points
    ax2.set_title("Extracted Continuous Points")
    ax2.set_aspect('equal')
    ax2.invert_yaxis() # Image coordinates have y pointing down
    
    cmap = plt.colormaps.get_cmap('tab20')
    total_pts = 0
    valid_contours = 0
    
    for i, contour in enumerate(contours):
        # Skip tiny noise dots to keep the paths clean
        if len(contour) < 5:
            continue 
            
        valid_contours += 1
        x = contour[:, 0]
        y = contour[:, 1]
        
        # Plot lines to show connectivity, and dots to show the points
        ax2.plot(x, y, marker='.', markersize=2, linestyle='-', linewidth=1, color=cmap(i % 20))
        total_pts += len(contour)
        
    print(f"Filtered down to {valid_contours} valid strokes.")
    print(f"Total points across all valid strokes: {total_pts}")
    
    plt.tight_layout()
    
    # Save the figure to verify it works headless, and also show it
    plt.savefig("extracted_points.png")
    print("Saved visualization to extracted_points.png")
    
    plt.show()
    
if __name__ == "__main__":
    main()
