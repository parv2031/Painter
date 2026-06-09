import cv2
import numpy as np

def extract_contours(image_path, canny_thresh1=100, canny_thresh2=200):
    """
    Extracts ordered points (contours) from an image.
    
    Returns
    -------
    formatted_contours : list of np.ndarray
        List of continuous paths. Each path is an (N, 2) array of [x, y] coordinates.
    edges : np.ndarray
        The binary edge image produced by Canny.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Optional: subtle blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Detection
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    
    # Find contours. 
    # RETR_LIST retrieves all contours without hierarchy.
    # CHAIN_APPROX_NONE keeps ALL boundary points, ensuring minimal distance 
    # (1 pixel) between consecutive points, making connectivity unambiguous.
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Format contours to simple (N, 2) arrays of [x, y]
    formatted_contours = []
    for cnt in contours:
        # cv2 contours are of shape (N, 1, 2). Squeeze to (N, 2)
        pts = cnt.squeeze(axis=1)
        
        # If it's a single point, it might squeeze to a 1D array. Ensure (N, 2)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
            
        formatted_contours.append(pts)
        
    return formatted_contours, edges


def space_points_wisely(contour, step_size=5.0):
    """
    Spaces points uniformly along the contour path using linear interpolation.
    This significantly reduces the number of points while keeping the stroke
    structure perfectly clear and ensuring minimal uniform distance between them.
    
    Parameters
    ----------
    contour : np.ndarray
        Array of shape (N, 2) containing the continuous points of a stroke.
    step_size : float
        The desired distance between consecutive points (in pixels).
        
    Returns
    -------
    np.ndarray
        Resampled contour of shape (M, 2).
    """
    if len(contour) < 2:
        return contour
        
    # Calculate cumulative distance along the contour
    diffs = np.diff(contour, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    
    total_len = cum_dists[-1]
    
    # If the whole stroke is smaller than one step, just return start and end
    if total_len < step_size:
        return np.array([contour[0], contour[-1]])
        
    # Create new evenly spaced distance points
    new_cum_dists = np.arange(0, total_len, step_size)
    
    # Ensure the very last point is always included to finish the stroke accurately
    if new_cum_dists[-1] != total_len:
        new_cum_dists = np.append(new_cum_dists, total_len)
        
    # Interpolate x and y coordinates
    new_x = np.interp(new_cum_dists, cum_dists, contour[:, 0])
    new_y = np.interp(new_cum_dists, cum_dists, contour[:, 1])
    
    return np.column_stack((new_x, new_y))

