import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import math
import time
import os
import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, Scale, Button, Label, Frame, IntVar, DoubleVar, StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl
from sklearn.cluster import DBSCAN
import svgwrite
import webbrowser
import json
from datetime import datetime

# Set matplotlib styles for modern-looking plots
plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.edgecolor'] = '#333333'

class ShapeApproximator:
    # Default configuration
    DEFAULT_CONFIG = {
        "SEGMENT_THRESHOLD": 10,  
        "SEGMENT_MIN_AREA": 100,  
        "DP_TOLERANCE": 3.0,
        "RANSAC_ITER": 100,
        "RANSAC_THRESH": 5.0,
        "CONTOUR_THICKNESS": 3,
        "OUTPUT_DIR": "output",
        "ENABLE_ANIMATION": True,
        "ANIMATION_FRAMES": 10,
        "COLOR_SCHEME": "vibrant",
        "USE_PARALLEL": True,
        "CUSTOM_METHOD_ENABLED": True,
        "CURVE_SMOOTHING": 0.5,
        "VECTOR_EXPORT": True,
        "AUTO_CLUSTER": True,
        "CLUSTER_EPS": 15.0,
        "DARK_MODE": False,
    }
    
    # Color schemes for visualization
    COLOR_SCHEMES = {
        "vibrant": {
            "original": (220, 53, 69),    # Vibrant red
            "hull": (40, 167, 69),        # Vibrant green
            "dp": (0, 123, 255),          # Vibrant blue
            "circle": (255, 193, 7),      # Vibrant yellow
            "custom": (153, 102, 255)     # Vibrant purple (for custom method)
        },
        "pastel": {
            "original": (255, 179, 186),  # Pastel red
            "hull": (186, 255, 201),      # Pastel green
            "dp": (186, 225, 255),        # Pastel blue
            "circle": (255, 236, 186),    # Pastel yellow
            "custom": (216, 191, 216)     # Pastel purple
        },
        "neon": {
            "original": (255, 0, 102),    # Neon pink
            "hull": (0, 255, 153),        # Neon green
            "dp": (0, 191, 255),          # Neon blue
            "circle": (255, 215, 0),      # Neon gold
            "custom": (204, 0, 255)       # Neon purple
        },
        "monochrome": {
            "original": (50, 50, 50),     # Dark gray
            "hull": (90, 90, 90),         # Medium gray
            "dp": (130, 130, 130),        # Light gray
            "circle": (170, 170, 170),    # Very light gray
            "custom": (0, 0, 0)           # Black
        }
    }
    
    def __init__(self, config=None):
        """Initialize with custom or default configuration."""
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        # Create output directory if it doesn't exist
        if not os.path.exists(self.config["OUTPUT_DIR"]):
            os.makedirs(self.config["OUTPUT_DIR"])
            
        # Initialize processing stats
        self.processing_stats = {
            "images_processed": 0,
            "total_contours": 0,
            "runtime_ms": 0,
            "methods": {
                "original": {"points": 0, "size": 0},
                "hull": {"points": 0, "size": 0},
                "dp": {"points": 0, "size": 0},
                "circle": {"points": 0, "size": 0},
                "custom": {"points": 0, "size": 0}
            }
        }
        
        # For visualization purposes
        self.current_image = None
        self.current_results = None
        
    def load_image(self, image_path):
        """Load and prepare an image for processing."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path} (might be corrupted or wrong format)")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.current_image = {"path": image_path, "rgb": img_rgb, "gray": img_gray}
        return img_rgb, img_gray
        
    def find_contours_in_image(self, img_gray, threshold=None, min_area=None):
        """Finds significant contours in the entire image with optional adaptive thresholding."""
        threshold = threshold or self.config["SEGMENT_THRESHOLD"]
        min_area = min_area or self.config["SEGMENT_MIN_AREA"]
        
        # Try adaptive thresholding if regular thresholding doesn't find enough contours
        _, thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If few contours found, try adaptive thresholding
        if len([cnt for cnt in contours if cv2.contourArea(cnt) > min_area]) <= 1:
            adaptive_thresh = cv2.adaptiveThreshold(
                img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        # Filter contours by area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Auto-cluster contours if enabled and more than 5 contours found
        if self.config["AUTO_CLUSTER"] and len(filtered_contours) > 5:
            filtered_contours = self.cluster_contours(filtered_contours)
            
        return filtered_contours
    
    def cluster_contours(self, contours):
        """Group nearby contours based on centroid proximity using DBSCAN."""
        # Extract centroids
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append([cx, cy])
            else:
                # Use mean of points if moments fails
                mean_pt = np.mean(cnt.reshape(-1, 2), axis=0)
                centroids.append([int(mean_pt[0]), int(mean_pt[1])])
                
        # Not enough contours to cluster
        if len(centroids) < 2:
            return contours
            
        # Perform clustering with DBSCAN
        clustering = DBSCAN(eps=self.config["CLUSTER_EPS"], min_samples=1).fit(centroids)
        labels = clustering.labels_
        
        # Group contours by cluster
        clustered_contours = []
        for label in set(labels):
            # Get indices of contours in this cluster
            indices = [i for i, l in enumerate(labels) if l == label]
            if len(indices) == 1:
                # Single contour in cluster - keep as is
                clustered_contours.append(contours[indices[0]])
            else:
                # Multiple contours in cluster - merge them
                all_points = np.vstack([contours[i].reshape(-1, 2) for i in indices])
                hull = cv2.convexHull(all_points)
                clustered_contours.append(hull)
                
        return clustered_contours
            
    def convex_hull_approximation(self, contour):
        """Compute convex hull of a contour."""
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

    def douglas_peucker_approximation(self, contour, tolerance=None):
        """Douglas-Peucker approximation with progressive simplification for animation."""
        tolerance = tolerance or self.config["DP_TOLERANCE"]
        epsilon = tolerance
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        return approx_contour
        
    def progressive_dp_approximation(self, contour, steps=10):
        """Create an animation of the Douglas-Peucker algorithm by gradually increasing tolerance."""
        # Start with a small epsilon and gradually increase
        max_tolerance = self.config["DP_TOLERANCE"]
        tolerances = np.linspace(0.1, max_tolerance, steps)
        
        frames = []
        for t in tolerances:
            frame = self.douglas_peucker_approximation(contour, tolerance=t)
            frames.append(frame)
            
        return frames

    def ransac_circle_fit(self, contour, iterations=None, threshold=None):
        """RANSAC algorithm to fit circles to contour points."""
        iterations = iterations or self.config["RANSAC_ITER"]
        threshold = threshold or self.config["RANSAC_THRESH"]
        
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

            # Calculate circle center and radius (handling potential issues)
            try:
                # Calculate circle center and radius from 3 points (using perpendicular bisectors)
                mid1 = (p1 + p2) / 2
                mid2 = (p2 + p3) / 2
                delta1 = p2 - p1
                delta2 = p3 - p2

                # Check for collinearity
                denom = delta1[0] * delta2[1] - delta1[1] * delta2[0]
                if abs(denom) < 1e-6:
                    continue  # Points are collinear

                # Perpendicular slopes
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

            except Exception:
                continue  # Skip errors in calculation

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

    def custom_approximation(self, contour):
        """
        Custom contour approximation method: Fourier Descriptor based smoothing
        This is my personal contribution to make the shape approximation more innovative.
        """
        # If disabled, just return the original contour
        if not self.config["CUSTOM_METHOD_ENABLED"]:
            return contour
            
        # Convert contour to complex representation for Fourier transform
        points = contour.reshape(-1, 2)
        if len(points) < 4:  # Need minimum points for transform
            return contour
            
        complex_pts = points[:, 0] + 1j * points[:, 1]
        
        # Apply Discrete Fourier Transform
        fourier_desc = np.fft.fft(complex_pts)
        
        # Calculate number of descriptors to keep (affects smoothing)
        smoothing_factor = self.config["CURVE_SMOOTHING"]  # 0-1 (higher = more smoothing)
        keep = max(2, int(len(fourier_desc) * (1 - smoothing_factor)))
        
        # Keep only the most significant descriptors
        fourier_desc_filtered = np.zeros_like(fourier_desc)
        fourier_desc_filtered[:keep//2] = fourier_desc[:keep//2]
        fourier_desc_filtered[-keep//2:] = fourier_desc[-keep//2:]
        
        # Reconstruct the smoothed contour
        smoothed_complex = np.fft.ifft(fourier_desc_filtered)
        smoothed_points = np.column_stack((smoothed_complex.real, smoothed_complex.imag))
        
        # Convert back to proper contour format
        return smoothed_points.reshape(-1, 1, 2).astype(np.int32)
        
    def export_to_svg(self, contours_dict, filename):
        """Export contours to SVG vector format for perfect scaling."""
        if not self.config["VECTOR_EXPORT"]:
            return
            
        # Determine image size from current image if available
        if self.current_image:
            img_height, img_width = self.current_image["rgb"].shape[:2]
        else:
            # Find max bounds from contours
            all_points = []
            for method, contour_list in contours_dict.items():
                if contour_list:
                    for contour in contour_list:
                        if contour is not None and len(contour) > 0:
                            all_points.extend(contour.reshape(-1, 2))
            
            if all_points:
                all_points = np.array(all_points)
                min_x, min_y = np.min(all_points, axis=0)
                max_x, max_y = np.max(all_points, axis=0)
                img_width, img_height = max_x - min_x + 20, max_y - min_y + 20  # Add padding
            else:
                img_width, img_height = 800, 600  # Default size
                
        # Create SVG drawing
        dwg = svgwrite.Drawing(filename, size=(img_width, img_height))
        
        # Add background
        dwg.add(dwg.rect(insert=(0, 0), size=(img_width, img_height), fill='white'))
        
        # Add each contour type with appropriate styling
        color_scheme = self.COLOR_SCHEMES[self.config["COLOR_SCHEME"]]
        
        for method_name, contour_list in contours_dict.items():
            if not contour_list:
                continue
                
            # Get color for this method
            if method_name in color_scheme:
                rgb_color = color_scheme[method_name]
                svg_color = f'rgb({rgb_color[0]},{rgb_color[1]},{rgb_color[2]})'
            else:
                svg_color = 'black'  # Default
                
            for contour in contour_list:
                if contour is None or len(contour) < 3:
                    continue
                
                # Convert contour to SVG path format
                points = contour.reshape(-1, 2)
                path_data = f"M {points[0][0]},{points[0][1]}"
                for point in points[1:]:
                    path_data += f" L {point[0]},{point[1]}"
                path_data += " Z"  # Close path
                
                # Create path with styling
                path = dwg.path(d=path_data, fill='none', stroke=svg_color, 
                               stroke_width=self.config["CONTOUR_THICKNESS"])
                dwg.add(path)
                
        # Save SVG file
        dwg.save()
        return filename
                
    def get_contours_stats(self, contours_list):
        """Calculate total points and size for a list of contours."""
        total_points = 0
        total_size_bytes = 0
        if not contours_list:
            return 0, 0
        for contour in contours_list:
            if contour is not None and len(contour) > 0:
                total_points += contour.shape[0]
                total_size_bytes += contour.nbytes
        return total_points, total_size_bytes

    def calculate_savings(self, new_size, original_size):
        """Calculate the percentage of data reduction."""
        if original_size == 0:
            return 0.0
        savings = 1.0 - (new_size / original_size)
        return 100.0 * savings
        
    def format_size(self, size_bytes):
        """Format byte sizes in a human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.2f} KB"
        else:
            return f"{size_bytes/(1024*1024):.2f} MB"
    
    def process_single_image(self, image_path):
        """Process a single image and return all contours and image data."""
        start_time = time.time()
        
        try:
            img_rgb, img_gray = self.load_image(image_path)
            print(f"Processing image: {os.path.basename(image_path)} ({img_gray.shape[0]}x{img_gray.shape[1]})")
            
            # Find contours in the whole image
            all_contours = self.find_contours_in_image(img_gray)
            
            if not all_contours:
                print(f"No significant contours found in {os.path.basename(image_path)}")
                return None
            
            # Process each contour
            original_contours = []
            hull_contours = []
            dp_contours = []
            circle_contours = []
            custom_contours = []
            
            # Animation frames if enabled
            animation_frames = []
            
            for contour in all_contours:
                # Apply approximations directly to each contour
                hull_contour = self.convex_hull_approximation(contour)
                dp_contour = self.douglas_peucker_approximation(contour)
                circle_contour = self.ransac_circle_fit(contour)
                custom_contour = self.custom_approximation(contour)
                
                # Generate animation frames if enabled
                if self.config["ENABLE_ANIMATION"]:
                    animation_frames.append(self.progressive_dp_approximation(
                        contour, steps=self.config["ANIMATION_FRAMES"]))
                
                # Store results
                original_contours.append(contour)
                hull_contours.append(hull_contour)
                dp_contours.append(dp_contour)
                if circle_contour is not None:
                    circle_contours.append(circle_contour)
                custom_contours.append(custom_contour)
            
            elapsed_time = time.time() - start_time
            
            results = {
                "image": img_rgb,
                "original_contours": original_contours,
                "hull_contours": hull_contours,
                "dp_contours": dp_contours,
                "circle_contours": circle_contours,
                "custom_contours": custom_contours,
                "image_size": img_rgb.nbytes,
                "image_shape": img_rgb.shape[:2],  # height, width
                "animation_frames": animation_frames if self.config["ENABLE_ANIMATION"] else None,
                "processing_time_ms": int(elapsed_time * 1000)
            }
            
            # Update the stats
            self.processing_stats["images_processed"] += 1
            self.processing_stats["total_contours"] += len(original_contours)
            self.processing_stats["runtime_ms"] += results["processing_time_ms"]
            
            # Save current results for visualization
            self.current_results = results
            
            return results
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_image_list(self, image_paths):
        """Process a list of images with optional parallel processing."""
        if self.config["USE_PARALLEL"] and len(image_paths) > 1:
            max_workers = min(cpu_count(), len(image_paths))
            print(f"Using {max_workers} parallel workers for processing {len(image_paths)} images")
            
            with Pool(processes=max_workers) as pool:
                results = list(tqdm(pool.imap(self.process_single_image, image_paths), 
                                   total=len(image_paths), desc="Processing Images"))
        else:
            results = []
            for img_path in tqdm(image_paths, desc="Processing Images"):
                result = self.process_single_image(img_path)
                if result:
                    results.append(result)
                    
        # Filter out None results
        results = [r for r in results if r is not None]
        
        return results
        
    def visualize_results(self, results, output_basename=None, show_plots=True):
        """Create comprehensive visualization of results."""
        if not results:
            print("No valid results to visualize!")
            return
            
        color_scheme = self.COLOR_SCHEMES[self.config["COLOR_SCHEME"]]
        
        # Process each image result
        for i, result in enumerate(results):
            # Setup figure with dark mode support
            fig_bg_color = '#1f1f1f' if self.config["DARK_MODE"] else '#f5f5f5'
            text_color = 'white' if self.config["DARK_MODE"] else 'black'
            
            # Create a figure with 2x2 layout
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.patch.set_facecolor(fig_bg_color)
            fig.suptitle(f"Shape Approximation Comparison", fontsize=18, fontweight='bold', y=0.98, color=text_color)
            
            img_rgb = result["image"]
            
            # Flatten the axes for easier indexing
            axs = axs.flatten()
            
            # Create the plots in a 2x2 grid
            methods = [
                {"name": "Original", "contours": result["original_contours"], "color": color_scheme["original"],
                 "desc": f"{len(result['original_contours'])} contours, {self.get_contours_stats(result['original_contours'])[0]} points"},
                {"name": "Convex Hull", "contours": result["hull_contours"], "color": color_scheme["hull"],
                 "desc": f"Savings: {self.calculate_savings(self.get_contours_stats(result['hull_contours'])[1], self.get_contours_stats(result['original_contours'])[1]):.1f}%"},
                {"name": "Douglas-Peucker", "contours": result["dp_contours"], "color": color_scheme["dp"],
                 "desc": f"Tolerance: {self.config['DP_TOLERANCE']}, Savings: {self.calculate_savings(self.get_contours_stats(result['dp_contours'])[1], self.get_contours_stats(result['original_contours'])[1]):.1f}%"}
            ]
            
            # Fourth plot - either custom method or circles
            if self.config["CUSTOM_METHOD_ENABLED"]:
                methods.append(
                    {"name": "Fourier Smoothing", "contours": result["custom_contours"], "color": color_scheme["custom"],
                     "desc": f"Smoothing: {self.config['CURVE_SMOOTHING']}, Savings: {self.calculate_savings(self.get_contours_stats(result['custom_contours'])[1], self.get_contours_stats(result['original_contours'])[1]):.1f}%"}
                )
            else:
                methods.append(
                    {"name": "RANSAC Circles", "contours": result["circle_contours"], "color": color_scheme["circle"],
                     "desc": f"{len(result['circle_contours'])} circles, Savings: {self.calculate_savings(self.get_contours_stats(result['circle_contours'])[1], self.get_contours_stats(result['original_contours'])[1]):.1f}%"}
                )
            
            # Draw each method
            for j, method in enumerate(methods):
                img_plot = img_rgb.copy()
                ax = axs[j]
                
                cv2.drawContours(img_plot, method["contours"], -1, method["color"], self.config["CONTOUR_THICKNESS"])
                ax.imshow(img_plot)
                
                # Add styled title and description with dark mode support
                ax.set_title(method["name"], fontsize=14, fontweight='bold', pad=10, color=text_color)
                
                # Box color based on dark mode
                box_color = '#2a2a2a' if self.config["DARK_MODE"] else 'white'
                box_text_color = 'white' if self.config["DARK_MODE"] else 'black'
                
                ax.text(0.5, -0.1, method["desc"], horizontalalignment='center', 
                        verticalalignment='center', transform=ax.transAxes, 
                        fontsize=10, color=box_text_color,
                        bbox=dict(facecolor=box_color, alpha=0.8, boxstyle='round,pad=0.5'))
                
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                # Set background color for each subplot
                ax_bg_color = '#2a2a2a' if self.config["DARK_MODE"] else '#f8f9fa'
                ax.set_facecolor(ax_bg_color)
            
            # Get stats for all methods
            orig_points, orig_size = self.get_contours_stats(result["original_contours"])
            hull_points, hull_size = self.get_contours_stats(result["hull_contours"])  
            dp_points, dp_size = self.get_contours_stats(result["dp_contours"])
            circle_points, circle_size = self.get_contours_stats(result["circle_contours"])
            custom_points, custom_size = self.get_contours_stats(result["custom_contours"])
            
            # Calculate savings
            hull_savings = self.calculate_savings(hull_size, orig_size)
            dp_savings = self.calculate_savings(dp_size, orig_size)
            circle_savings = self.calculate_savings(circle_size, orig_size)
            custom_savings = self.calculate_savings(custom_size, orig_size)
            
            # Summary box with all stats
            summary_text = (
                f"Image: {os.path.basename(self.current_image['path'])}\n"
                f"Original: {orig_points} points\n"
                f"Hull: {hull_points} points ({hull_savings:.1f}% savings)\n"
                f"D-P: {dp_points} points ({dp_savings:.1f}% savings)\n"
                f"Circles: {circle_points} points ({circle_savings:.1f}% savings)\n"
                f"Custom: {custom_points} points ({custom_savings:.1f}% savings)\n"
                f"Processing time: {result['processing_time_ms']} ms"
            )
            
            # Box color based on dark mode
            box_color = '#2a2a2a' if self.config["DARK_MODE"] else 'white'
            
            # Add summary below the plots
            fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12, color=text_color,
                     bbox=dict(facecolor=box_color, alpha=0.9, boxstyle='round,pad=0.7'))
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            # Add timestamp and metadata
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            fig.text(0.98, 0.01, f"Generated: {timestamp}", ha='right', fontsize=8, color=text_color)
            
            # Save comparison image
            if output_basename:
                output_name = f"{output_basename}_{i+1}.png"
            else:
                output_name = os.path.join(self.config["OUTPUT_DIR"], f"comparison_{i+1}.png")
                
            plt.savefig(output_name, dpi=300, bbox_inches='tight')
            print(f"Comparison results saved to {output_name}")
            
            # Also export to SVG if enabled
            if self.config["VECTOR_EXPORT"]:
                svg_filename = os.path.splitext(output_name)[0] + ".svg"
                svg_file = self.export_to_svg({
                    "original": result["original_contours"],
                    "hull": result["hull_contours"],
                    "dp": result["dp_contours"], 
                    "circle": result["circle_contours"],
                    "custom": result["custom_contours"]
                }, svg_filename)
                print(f"Vector graphics exported to {svg_filename}")
            
            # Create contours-only visualization
            fig_outline, axs_outline = plt.subplots(2, 2, figsize=(12, 10))
            fig_outline.suptitle(f"Contours Only - No Background", fontsize=18, fontweight='bold', y=0.98, color=text_color)
            fig_outline.patch.set_facecolor(fig_bg_color)
            
            # Flatten the axes for easier indexing
            axs_outline = axs_outline.flatten()
            
            # Get image dimensions
            h, w = result["image_shape"]
            
            # Draw each method on white background
            bg_color = (30, 30, 30) if self.config["DARK_MODE"] else (255, 255, 255)
            for j, method in enumerate(methods):
                ax = axs_outline[j]
                
                # Create background image
                blank_img = np.ones((h, w, 3), dtype=np.uint8) * bg_color
                
                # Draw contours
                cv2.drawContours(blank_img, method["contours"], -1, method["color"], self.config["CONTOUR_THICKNESS"])
                ax.imshow(blank_img)
                
                # Add styled title
                ax.set_title(method["name"], fontsize=14, fontweight='bold', color=text_color)
                ax.text(0.5, -0.1, method["desc"], horizontalalignment='center', 
                        verticalalignment='center', transform=ax.transAxes, 
                        fontsize=10, color=box_text_color,
                        bbox=dict(facecolor=box_color, alpha=0.8, boxstyle='round,pad=0.5'))
                
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                # Set background color for each subplot
                ax_bg_color = '#2a2a2a' if self.config["DARK_MODE"] else '#f8f9fa'
                ax.set_facecolor(ax_bg_color)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save contours-only image
            if output_basename:
                outlines_name = f"{os.path.splitext(output_basename)[0]}_contours_{i+1}.png"
            else:
                outlines_name = os.path.join(self.config["OUTPUT_DIR"], f"contours_only_{i+1}.png")
                
            plt.savefig(outlines_name, dpi=300, bbox_inches='tight')
            print(f"Contours-only image saved to {outlines_name}")
            
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
                plt.close(fig_outline)
                
        # If animation is enabled, create animation frames
        if self.config["ENABLE_ANIMATION"] and result["animation_frames"]:
            self.create_animation(result, output_basename)
            
    def create_animation(self, result, output_basename=None):
        """Create animated GIF of progressive contour approximation."""
        try:
            import imageio
            
            if not result["animation_frames"]:
                return
                
            frames = []
            h, w = result["image_shape"]
            img_rgb = result["image"]
            
            # Create a frame for each step in the approximation
            for step_idx in range(self.config["ANIMATION_FRAMES"]):
                # Create a new image for this frame
                frame_img = img_rgb.copy()
                
                # Get the DP approximation for each contour at this step
                for contour_idx, contour_frames in enumerate(result["animation_frames"]):
                    if step_idx < len(contour_frames):
                        dp_contour = contour_frames[step_idx]
                        cv2.drawContours(frame_img, [dp_contour], -1, self.COLOR_SCHEMES[self.config["COLOR_SCHEME"]]["dp"], 
                                       self.config["CONTOUR_THICKNESS"])
                
                frames.append(frame_img)
                
            # Save the animation
            if output_basename:
                anim_filename = f"{os.path.splitext(output_basename)[0]}_animation.gif"
            else:
                anim_filename = os.path.join(self.config["OUTPUT_DIR"], "dp_animation.gif")
                
            imageio.mimsave(anim_filename, frames, fps=4)
            print(f"Animation saved to {anim_filename}")
            
        except ImportError:
            print("Animation creation requires imageio. Install with: pip install imageio")
    
    def save_processing_stats(self, output_json=None):
        """Save processing statistics to a JSON file for future reference."""
        if not output_json:
            output_json = os.path.join(self.config["OUTPUT_DIR"], f"stats_{time.strftime('%Y%m%d_%H%M%S')}.json")
            
        # Add timestamp
        self.processing_stats["timestamp"] = datetime.now().isoformat()
        self.processing_stats["config"] = {k: v for k, v in self.config.items() if type(v) in (str, int, float, bool)}
        
        with open(output_json, 'w') as f:
            json.dump(self.processing_stats, f, indent=2)
            
        print(f"Processing statistics saved to {output_json}")
        return output_json
        
    def create_gui(self):
        """Create an interactive GUI for visualizing and tweaking shape approximation."""
        # Setup the main window
        root = tk.Tk()
        root.title("Shape Approximation Tool")
        root.geometry("1200x800")
        
        # Dark mode setup
        bg_color = "#333333" if self.config["DARK_MODE"] else "#f0f0f0"
        text_color = "white" if self.config["DARK_MODE"] else "black"
        root.configure(bg=bg_color)
        
        # Create frames
        control_frame = Frame(root, bg=bg_color)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        viz_frame = Frame(root, bg=bg_color)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Variables for controls
        self.gui_vars = {
            "threshold": IntVar(value=self.config["SEGMENT_THRESHOLD"]),
            "min_area": IntVar(value=self.config["SEGMENT_MIN_AREA"]),
            "dp_tolerance": DoubleVar(value=self.config["DP_TOLERANCE"]),
            "ransac_thresh": DoubleVar(value=self.config["RANSAC_THRESH"]),
            "color_scheme": StringVar(value=self.config["COLOR_SCHEME"]),
            "custom_enabled": IntVar(value=int(self.config["CUSTOM_METHOD_ENABLED"])),
            "smoothing": DoubleVar(value=self.config["CURVE_SMOOTHING"]),
        }
        
        # Function to handle control changes
        def update_config(event=None):
            self.config["SEGMENT_THRESHOLD"] = self.gui_vars["threshold"].get()
            self.config["SEGMENT_MIN_AREA"] = self.gui_vars["min_area"].get()
            self.config["DP_TOLERANCE"] = self.gui_vars["dp_tolerance"].get()
            self.config["RANSAC_THRESH"] = self.gui_vars["ransac_thresh"].get()
            self.config["COLOR_SCHEME"] = self.gui_vars["color_scheme"].get()
            self.config["CUSTOM_METHOD_ENABLED"] = bool(self.gui_vars["custom_enabled"].get())
            self.config["CURVE_SMOOTHING"] = self.gui_vars["smoothing"].get()
            
            # If we have an image loaded, reprocess it
            if self.current_image:
                self.process_and_update_display()
                
        # Add controls
        Label(control_frame, text="Shape Approximation Controls", font=("Arial", 14, "bold"), 
             bg=bg_color, fg=text_color).pack(pady=10)
        
        # Slider for threshold
        Label(control_frame, text="Threshold:", bg=bg_color, fg=text_color).pack(anchor="w")
        Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.gui_vars["threshold"],
             command=update_config, length=200, bg=bg_color, fg=text_color).pack(fill=tk.X)
        
        # Slider for min area
        Label(control_frame, text="Min Contour Area:", bg=bg_color, fg=text_color).pack(anchor="w")
        Scale(control_frame, from_=10, to=1000, orient=tk.HORIZONTAL, variable=self.gui_vars["min_area"],
             command=update_config, length=200, bg=bg_color, fg=text_color).pack(fill=tk.X)
        
        # Slider for DP tolerance
        Label(control_frame, text="Douglas-Peucker Tolerance:", bg=bg_color, fg=text_color).pack(anchor="w")
        Scale(control_frame, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL,
             variable=self.gui_vars["dp_tolerance"], command=update_config, length=200, 
             bg=bg_color, fg=text_color).pack(fill=tk.X)
        
        # Slider for RANSAC threshold
        Label(control_frame, text="RANSAC Threshold:", bg=bg_color, fg=text_color).pack(anchor="w")
        Scale(control_frame, from_=1.0, to=20.0, resolution=0.5, orient=tk.HORIZONTAL,
             variable=self.gui_vars["ransac_thresh"], command=update_config, length=200,
             bg=bg_color, fg=text_color).pack(fill=tk.X)
        
        # Checkbox for custom method
        tk.Checkbutton(control_frame, text="Enable Custom Method", variable=self.gui_vars["custom_enabled"],
                      command=update_config, bg=bg_color, fg=text_color, 
                      selectcolor=bg_color).pack(anchor="w", pady=5)
        
        # Slider for smoothing if custom method enabled
        Label(control_frame, text="Curve Smoothing:", bg=bg_color, fg=text_color).pack(anchor="w")
        Scale(control_frame, from_=0.0, to=0.9, resolution=0.05, orient=tk.HORIZONTAL,
             variable=self.gui_vars["smoothing"], command=update_config, length=200,
             bg=bg_color, fg=text_color).pack(fill=tk.X)
        
        # Color scheme selector
        Label(control_frame, text="Color Scheme:", bg=bg_color, fg=text_color).pack(anchor="w")
        for scheme in self.COLOR_SCHEMES.keys():
            tk.Radiobutton(control_frame, text=scheme.capitalize(), variable=self.gui_vars["color_scheme"],
                          value=scheme, command=update_config, bg=bg_color, fg=text_color,
                          selectcolor=bg_color).pack(anchor="w")
        
        # Button to load image
        Button(control_frame, text="Load Image", command=self.load_image_dialog, 
              bg="#4CAF50", fg="white").pack(fill=tk.X, pady=10)
        
        # Button to save results
        Button(control_frame, text="Save Results", command=self.save_results, 
              bg="#2196F3", fg="white").pack(fill=tk.X, pady=5)
        
        # Button to export vector
        Button(control_frame, text="Export SVG", command=self.export_svg_from_gui, 
              bg="#9C27B0", fg="white").pack(fill=tk.X, pady=5)
        
        # Create matplotlib figure for visualization
        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor(bg_color)
        
        # Flatten axes for easier access
        self.axes = self.axes.flatten()
        
        # Embed the matplotlib figure in tkinter
        canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status text
        self.status_text = tk.StringVar(value="Ready. Load an image to begin.")
        Label(viz_frame, textvariable=self.status_text, bg=bg_color, fg=text_color).pack(pady=5)
        
        # Start the GUI loop
        root.mainloop()
        
    def load_image_dialog(self):
        """GUI function to load an image file via dialog."""
        filetypes = (
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Open an image file',
            initialdir='.',
            filetypes=filetypes
        )
        
        if filename:
            self.status_text.set(f"Loading image: {os.path.basename(filename)}")
            try:
                # Process the image and update display
                self.current_image = None  # Reset current image
                self.process_single_image(filename)
                self.process_and_update_display()
            except Exception as e:
                self.status_text.set(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    def process_and_update_display(self):
        """Process current image with current settings and update GUI display."""
        if not self.current_image:
            self.status_text.set("No image loaded!")
            return
            
        try:
            # Reprocess the current image
            result = self.process_single_image(self.current_image["path"])
            
            if not result:
                self.status_text.set("Failed to process image")
                return
                
            # Get color scheme
            color_scheme = self.COLOR_SCHEMES[self.config["COLOR_SCHEME"]]
            
            # Clear previous plots
            for ax in self.axes:
                ax.clear()
                
            # Setup the plots
            methods = [
                {"name": "Original", "contours": result["original_contours"], "color": color_scheme["original"]},
                {"name": "Convex Hull", "contours": result["hull_contours"], "color": color_scheme["hull"]},
                {"name": "Douglas-Peucker", "contours": result["dp_contours"], "color": color_scheme["dp"]}
            ]
            
            # Fourth plot - either custom method or circles
            if self.config["CUSTOM_METHOD_ENABLED"]:
                methods.append({
                    "name": "Fourier Smoothing", 
                    "contours": result["custom_contours"], 
                    "color": color_scheme["custom"]
                })
            else:
                methods.append({
                    "name": "RANSAC Circles", 
                    "contours": result["circle_contours"], 
                    "color": color_scheme["circle"]
                })
            
            # Draw each method
            for j, method in enumerate(methods):
                img_plot = result["image"].copy()
                ax = self.axes[j]
                
                # Draw contours
                cv2.drawContours(img_plot, method["contours"], -1, method["color"], self.config["CONTOUR_THICKNESS"])
                ax.imshow(img_plot)
                ax.set_title(method["name"])
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Update the figure
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            
            # Update status
            orig_pts = self.get_contours_stats(result["original_contours"])[0]
            self.status_text.set(
                f"Processed: {os.path.basename(self.current_image['path'])} | "
                f"Found {len(result['original_contours'])} contours with {orig_pts} points | "
                f"Time: {result['processing_time_ms']} ms"
            )
            
        except Exception as e:
            self.status_text.set(f"Error updating display: {e}")
            import traceback
            traceback.print_exc()
    
    def save_results(self):
        """Save current visualization from the GUI."""
        if not self.current_results or not self.current_image:
            self.status_text.set("No results to save!")
            return
            
        # Get output filename
        output_dir = self.config["OUTPUT_DIR"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        basename = os.path.splitext(os.path.basename(self.current_image["path"]))[0]
        output_file = os.path.join(output_dir, f"{basename}_processed.png")
        
        # Save the figure
        self.fig.savefig(output_file, dpi=300, bbox_inches='tight')
        self.status_text.set(f"Results saved to {output_file}")
    
    def export_svg_from_gui(self):
        """Export current results to SVG from GUI."""
        if not self.current_results or not self.current_image:
            self.status_text.set("No results to export!")
            return
            
        # Get output filename
        output_dir = self.config["OUTPUT_DIR"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        basename = os.path.splitext(os.path.basename(self.current_image["path"]))[0]
        svg_file = os.path.join(output_dir, f"{basename}_vector.svg")
        
        # Export to SVG
        try:
            result = self.current_results
            self.export_to_svg({
                "original": result["original_contours"],
                "hull": result["hull_contours"],
                "dp": result["dp_contours"],
                "circle": result["circle_contours"],
                "custom": result["custom_contours"]
            }, svg_file)
            
            self.status_text.set(f"SVG exported to {svg_file}")
            
            # Ask if user wants to open the SVG
            if tk.messagebox.askyesno("Open SVG", "Would you like to open the SVG file?"):
                webbrowser.open(svg_file)
                
        except Exception as e:
            self.status_text.set(f"Error exporting SVG: {e}")
            import traceback
            traceback.print_exc()


# --- Main Entry Point ---
def main():
    """Main entry point for the application."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Advanced Shape Approximation Tool')
    parser.add_argument('--input', '-i', type=str, default="*.jpg",
                       help='Input image path or pattern')
    parser.add_argument('--threshold', '-t', type=int, default=10,
                       help='Threshold for contour detection')
    parser.add_argument('--min-area', '-m', type=int, default=100,
                       help='Minimum contour area to process')
    parser.add_argument('--dp-tolerance', '-d', type=float, default=3.0,
                       help='Tolerance for Douglas-Peucker approximation')
    parser.add_argument('--ransac-iter', '-r', type=int, default=100,
                       help='Iterations for RANSAC circle fitting')
    parser.add_argument('--output-dir', '-o', type=str, default="output",
                       help='Output directory for results')
    parser.add_argument('--color-scheme', '-c', type=str, default="vibrant",
                       choices=["vibrant", "pastel", "neon", "monochrome"],
                       help='Color scheme for visualization')
    parser.add_argument('--no-animation', action='store_true',
                       help='Disable animation generation')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--dark-mode', action='store_true',
                       help='Use dark mode for visualization')
    parser.add_argument('--gui', action='store_true',
                       help='Launch interactive GUI')
    
    args = parser.parse_args()
    
    # Configure the approximator
    config = {
        "SEGMENT_THRESHOLD": args.threshold,
        "SEGMENT_MIN_AREA": args.min_area,
        "DP_TOLERANCE": args.dp_tolerance,
        "RANSAC_ITER": args.ransac_iter,
        "OUTPUT_DIR": args.output_dir,
        "COLOR_SCHEME": args.color_scheme,
        "ENABLE_ANIMATION": not args.no_animation,
        "USE_PARALLEL": not args.no_parallel,
        "DARK_MODE": args.dark_mode
    }
    
    approximator = ShapeApproximator(config)
    
    # Start GUI or process images
    if args.gui:
        approximator.create_gui()
    else:
        # Find all matching image files
        image_files = glob.glob(args.input)
        
        if not image_files:
            print(f"No image files found matching pattern: {args.input}")
            return
            
        print(f"Found {len(image_files)} image(s) to process")
        
        # Process the images
        results = approximator.process_image_list(image_files)
        
        if not results:
            print("No valid results from any images")
            return
            
        # Visualize results
        approximator.visualize_results(results)
        
        # Save processing stats
        approximator.save_processing_stats()
        
        print("\nProcessing Summary:")
        print(f"- Images processed: {approximator.processing_stats['images_processed']}")
        print(f"- Total contours: {approximator.processing_stats['total_contours']}")
        print(f"- Total runtime: {approximator.processing_stats['runtime_ms'] / 1000:.2f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

