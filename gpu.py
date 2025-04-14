import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import math
import time
import os
import glob

# --- Configuration ---
INPUT_PATH = "007.jpg"  # Can be a specific file or wildcard pattern
SEGMENT_THRESHOLD = 10  # Threshold for finding contours
SEGMENT_MIN_AREA = 100  # Min area for a contour to be processed
DP_TOLERANCE = 3.0
RANSAC_ITER = 100
RANSAC_THRESH = 5.0
OUTPUT_FILENAME = "shape_approximation_comparison.png"
CONTOURS_ONLY_FILENAME = "contours_only.png"
CONTOUR_THICKNESS = 3  # Increased thickness for better visibility

# --- Helper Functions ---

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path} (might be corrupted or wrong format)")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_rgb, img_gray

def find_contours_in_image(img_gray, threshold, min_area):
    """Finds significant contours in the entire image."""
    _, thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered_contours

def convex_hull_approximation(contour):
    points = contour.reshape(-1, 2)
    if len(points) < 3:
        return contour  # Return original if too small
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except Exception as e:
        return contour
    hull_contour = hull_points.reshape(-1, 1, 2).astype(np.int32)
    return hull_contour

def douglas_peucker_approximation_cv2(contour, tolerance):
    """Uses cv2.approxPolyDP for Douglas-Peucker approximation."""
    epsilon = tolerance
    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
    return approx_contour

def ransac_circle_fit(contour, iterations=100, threshold=5.0):
    points = contour.reshape(-1, 2)
    if len(points) < 3:
        return None  # Return None if cannot fit

    best_inliers_count = -1
    best_center = None
    best_radius = None
    num_points = len(points)

    for _ in range(iterations):
        # Randomly sample 3 points
        sample_idxs = np.random.choice(num_points, 3, replace=False)
        sample_points = points[sample_idxs]
        p1, p2, p3 = sample_points[0], sample_points[1], sample_points[2]

        # Calculate circle center and radius from 3 points (using perpendicular bisectors)
        mid1 = (p1 + p2) / 2
        mid2 = (p2 + p3) / 2
        delta1 = p2 - p1
        delta2 = p3 - p2

        # Check for collinearity (denominator close to zero)
        denom = delta1[0] * delta2[1] - delta1[1] * delta2[0]
        if abs(denom) < 1e-6:
            continue  # Points are collinear, cannot form a circle

        # Perpendicular slopes (handle vertical/horizontal lines)
        m1 = -delta1[0] / delta1[1] if abs(delta1[1]) > 1e-6 else float('inf')
        m2 = -delta2[0] / delta2[1] if abs(delta2[1]) > 1e-6 else float('inf')

        # Calculate center coordinates
        if abs(delta1[1]) < 1e-6:  # First line is horizontal
            center_x = mid1[0]
            if abs(delta2[1]) < 1e-6: continue  # Both horizontal (collinear)
            b2 = mid2[1] - m2 * mid2[0]
            center_y = m2 * center_x + b2
        elif abs(delta2[1]) < 1e-6:  # Second line is horizontal
            center_x = mid2[0]
            # m1 cannot be inf here because delta1[1] is not zero
            b1 = mid1[1] - m1 * mid1[0]
            center_y = m1 * center_x + b1
        elif m1 == m2:  # Parallel bisectors (collinear original points)
            continue
        else:  # General case
            b1 = mid1[1] - m1 * mid1[0]
            b2 = mid2[1] - m2 * mid2[0]
            center_x = (b2 - b1) / (m1 - m2)
            center_y = m1 * center_x + b1

        center = np.array([center_x, center_y])
        radius = np.linalg.norm(p1 - center)

        # Count inliers
        distances_to_center = np.linalg.norm(points - center, axis=1)
        inliers_mask = np.abs(distances_to_center - radius) < threshold
        current_inliers_count = np.sum(inliers_mask)

        # Update best model if current is better
        if current_inliers_count > best_inliers_count:
            best_inliers_count = current_inliers_count
            best_center = center
            best_radius = radius

    if best_center is None or best_radius is None or best_radius <= 0 or best_inliers_count < 3:
        return None  # Indicate failure

    # Generate points for the best fit circle contour
    theta = np.linspace(0, 2*np.pi, 50)  # 50 points for a smooth circle
    circle_x = best_center[0] + best_radius * np.cos(theta)
    circle_y = best_center[1] + best_radius * np.sin(theta)
    circle_points = np.column_stack((circle_x, circle_y))
    circle_contour = circle_points.reshape(-1, 1, 2).astype(np.int32)
    return circle_contour


def get_contours_stats(contours_list):
    """Calculates total points and size for a list of contours."""
    total_points = 0
    total_size_bytes = 0
    if not contours_list:
        return 0, 0
    for contour in contours_list:
        if contour is not None and len(contour) > 0:
            total_points += contour.shape[0]
            total_size_bytes += contour.nbytes
    return total_points, total_size_bytes

def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.2f} KB"
    else:
        return f"{size_bytes/(1024*1024):.2f} MB"

def calculate_savings(new_size, original_size):
    if original_size == 0:
        return 0.0
    savings = 1.0 - (new_size / original_size)
    return 100.0 * savings

def process_single_image(image_path):
    """Process a single image and return all contours and image data."""
    img_rgb, img_gray = load_image(image_path)
    print(f"Processing image: {os.path.basename(image_path)} ({img_gray.shape[0]}x{img_gray.shape[1]})")
    
    # Find contours in the whole image (no grid segmentation)
    all_contours = find_contours_in_image(img_gray, SEGMENT_THRESHOLD, SEGMENT_MIN_AREA)
    
    if not all_contours:
        print(f"No significant contours found in {os.path.basename(image_path)}")
        return None
    
    # Process each contour
    original_contours = []
    hull_contours = []
    dp_contours = []
    circle_contours = []
    
    for contour in all_contours:
        # Apply approximations directly to each contour
        hull_contour = convex_hull_approximation(contour)
        dp_contour = douglas_peucker_approximation_cv2(contour, tolerance=DP_TOLERANCE)
        circle_contour = ransac_circle_fit(contour, iterations=RANSAC_ITER, threshold=RANSAC_THRESH)
        
        # Store results
        original_contours.append(contour)
        hull_contours.append(hull_contour)
        dp_contours.append(dp_contour)
        if circle_contour is not None:
            circle_contours.append(circle_contour)
    
    return {
        "image": img_rgb,
        "original_contours": original_contours,
        "hull_contours": hull_contours,
        "dp_contours": dp_contours,
        "circle_contours": circle_contours,
        "image_size": img_rgb.nbytes,
        "image_shape": img_rgb.shape[:2]  # height, width
    }

# Enhanced colors for better visibility (vibrant modern colors)
CONTOUR_COLORS = {
    "original": (220, 53, 69),     # Vibrant red
    "hull": (40, 167, 69),         # Vibrant green
    "dp": (0, 123, 255),           # Vibrant blue
    "circle": (255, 193, 7)        # Vibrant yellow
}

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Find all matching image files
        image_files = glob.glob(INPUT_PATH)
        
        if not image_files:
            raise FileNotFoundError(f"No image files found matching pattern: {INPUT_PATH}")
            
        print(f"Found {len(image_files)} image(s) to process")
        
        # Process each image separately
        results = []
        for img_path in image_files:
            result = process_single_image(img_path)
            if result:
                results.append(result)
        
        if not results:
            raise ValueError("No valid results found from any images")
            
        # Calculate total stats across all images
        total_orig_points = 0
        total_orig_size = 0
        total_hull_points = 0
        total_hull_size = 0
        total_dp_points = 0
        total_dp_size = 0
        total_circle_points = 0
        total_circle_size = 0
        total_image_size = 0
        
        for result in results:
            orig_points, orig_size = get_contours_stats(result["original_contours"])
            hull_points, hull_size = get_contours_stats(result["hull_contours"])
            dp_points, dp_size = get_contours_stats(result["dp_contours"])
            circle_points, circle_size = get_contours_stats(result["circle_contours"])
            
            total_orig_points += orig_points
            total_orig_size += orig_size
            total_hull_points += hull_points
            total_hull_size += hull_size
            total_dp_points += dp_points
            total_dp_size += dp_size
            total_circle_points += circle_points
            total_circle_size += circle_size
            total_image_size += result["image_size"]
        
        # Calculate savings
        hull_savings = calculate_savings(total_hull_size, total_orig_size)
        dp_savings = calculate_savings(total_dp_size, total_orig_size)
        circle_savings = calculate_savings(total_circle_size, total_orig_size)
        
        print("\n--- Aggregated Results ---")
        print(f"Total Images Processed: {len(results)}")
        print(f"Total Original Contours: {sum(len(r['original_contours']) for r in results)}")
        print(f"Total Original Points: {total_orig_points}, Total Size: {format_size(total_orig_size)}")
        print(f"Convex Hull: {total_hull_points} points ({total_hull_points/total_orig_points*100:.1f}%), Size: {format_size(total_hull_size)}")
        print(f"  Savings vs Original: {hull_savings:.1f}%")
        print(f"Douglas-Peucker: {total_dp_points} points ({total_dp_points/total_orig_points*100:.1f}%), Size: {format_size(total_dp_size)}")
        print(f"  Savings vs Original: {dp_savings:.1f}%")
        print(f"RANSAC Circles: {total_circle_points} points, Size: {format_size(total_circle_size)}")
        print(f"  Savings vs Original: {circle_savings:.1f}%")
        
        # --- Enhanced Visualization ---
        print("\nGenerating comparison plots...")
        
        # Set up the figure with 2x2 grid layout
        n_images = len(results)
        
        # Create a modern style for plots
        plt.style.use('seaborn-v0_8-whitegrid')
        
        for i, result in enumerate(results):
            # Create a figure with 2x2 layout
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Shape Approximation Comparison", fontsize=18, fontweight='bold', y=0.98)
            fig.patch.set_facecolor('#f5f5f5')  # Light gray background for the figure
            
            img_rgb = result["image"]
            
            # Flatten the axes for easier indexing
            axs = axs.flatten()
            
            # Create the four plots in a 2x2 grid
            methods = [
                {"name": "Original", "contours": result["original_contours"], "color": CONTOUR_COLORS["original"],
                 "desc": f"{len(result['original_contours'])} contours, {get_contours_stats(result['original_contours'])[0]} points"},
                {"name": "Convex Hull", "contours": result["hull_contours"], "color": CONTOUR_COLORS["hull"],
                 "desc": f"Savings: {calculate_savings(get_contours_stats(result['hull_contours'])[1], get_contours_stats(result['original_contours'])[1]):.1f}%"},
                {"name": "Douglas-Peucker", "contours": result["dp_contours"], "color": CONTOUR_COLORS["dp"],
                 "desc": f"Tolerance: {DP_TOLERANCE}, Savings: {calculate_savings(get_contours_stats(result['dp_contours'])[1], get_contours_stats(result['original_contours'])[1]):.1f}%"},
                {"name": "RANSAC Circles", "contours": result["circle_contours"], "color": CONTOUR_COLORS["circle"],
                 "desc": f"{len(result['circle_contours'])} circles, Savings: {calculate_savings(get_contours_stats(result['circle_contours'])[1], get_contours_stats(result['original_contours'])[1]):.1f}%"}
            ]
            
            # Draw each method
            for j, method in enumerate(methods):
                img_plot = img_rgb.copy()
                ax = axs[j]
                
                cv2.drawContours(img_plot, method["contours"], -1, method["color"], CONTOUR_THICKNESS)
                ax.imshow(img_plot)
                
                # Add styled title and description
                ax.set_title(method["name"], fontsize=14, fontweight='bold', pad=10)
                ax.text(0.5, -0.1, method["desc"], horizontalalignment='center', 
                        verticalalignment='center', transform=ax.transAxes, 
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_facecolor('#f8f9fa')  # Light background for each subplot
            
            # Summary box with all stats
            summary_text = (
                f"IMAGE: {os.path.basename(image_files[i])}\n"
                f"Original: {get_contours_stats(result['original_contours'])[0]} points\n"
                f"Hull: {get_contours_stats(result['hull_contours'])[0]} points ({hull_savings:.1f}% savings)\n"
                f"D-P: {get_contours_stats(result['dp_contours'])[0]} points ({dp_savings:.1f}% savings)\n"
                f"Circles: {get_contours_stats(result['circle_contours'])[0]} points ({circle_savings:.1f}% savings)"
            )
            
            # Add summary below the plots
            fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12, 
                     bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.7'))
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            # Add timestamp and metadata
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            fig.text(0.98, 0.01, f"Generated: {timestamp}", ha='right', fontsize=8, color='gray')
            
            # Save comparison image
            output_name = f"{os.path.splitext(OUTPUT_FILENAME)[0]}_{i+1}.png"
            plt.savefig(output_name, dpi=300, bbox_inches='tight')
            print(f"Comparison results saved to {output_name}")
            
            # --- Create contours-only visualization ---
            fig_outline, axs_outline = plt.subplots(2, 2, figsize=(12, 10))
            fig_outline.suptitle(f"Contours Only - No Background", fontsize=18, fontweight='bold', y=0.98)
            fig_outline.patch.set_facecolor('white')
            
            # Flatten the axes for easier indexing
            axs_outline = axs_outline.flatten()
            
            # Get image dimensions
            h, w = result["image_shape"]
            
            # Draw each method on white background
            for j, method in enumerate(methods):
                ax = axs_outline[j]
                
                # Create a white background image
                blank_img = np.ones((h, w, 3), dtype=np.uint8) * 255
                
                # Draw contours
                cv2.drawContours(blank_img, method["contours"], -1, method["color"], CONTOUR_THICKNESS)
                ax.imshow(blank_img)
                
                # Add styled title
                ax.set_title(method["name"], fontsize=14, fontweight='bold')
                ax.text(0.5, -0.1, method["desc"], horizontalalignment='center', 
                        verticalalignment='center', transform=ax.transAxes, 
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save contours-only image
            outlines_name = f"{os.path.splitext(CONTOURS_ONLY_FILENAME)[0]}_{i+1}.png"
            plt.savefig(outlines_name, dpi=300, bbox_inches='tight')
            print(f"Contours-only image saved to {outlines_name}")
            
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()