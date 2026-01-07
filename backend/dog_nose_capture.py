#!/usr/bin/env python3
"""
Dog Nose Auto-Capture System with Advanced Nose Detection
Detects and outlines the nose region in real-time like the reference images
"""

import cv2
import numpy as np
import os
from datetime import datetime
import time
import argparse
from pathlib import Path


class DogNoseCaptureSystem:
    def __init__(self, output_dir="dog_nose_dataset", blur_threshold=100.0, 
                 auto_capture=True, capture_interval=2.0, min_sharpness=150.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.blur_threshold = blur_threshold
        self.auto_capture = auto_capture
        self.capture_interval = capture_interval
        self.min_sharpness = min_sharpness
        self.last_capture_time = 0
        self.capture_count = 0
        
        # Visualization mode
        self.viz_mode = 0  # 0=normal, 1=edges, 2=threshold, 3=wrinkles, 4=outline
        
        # Create subdirectories
        self.sharp_dir = self.output_dir / "sharp"
        self.rejected_dir = self.output_dir / "rejected"
        self.viz_dir = self.output_dir / "visualizations"
        self.sharp_dir.mkdir(exist_ok=True)
        self.rejected_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Sharpness threshold: {self.min_sharpness}")
        print(f"⏱️  Auto-capture interval: {self.capture_interval}s")
        
    def calculate_sharpness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance
    
    def calculate_edge_density(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density
    
    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def detect_focus_region(self, image):
        h, w = image.shape[:2]
        center_region = image[h//3:2*h//3, w//3:2*w//3]
        return self.calculate_sharpness(center_region)
    
    def detect_nose_region(self, image):
        """
        Enhanced nose detection using color segmentation and morphology
        Dog noses are typically darker and in the center
        """
        h, w = image.shape[:2]
        
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Focus on center region where nose usually is
        center_y1, center_y2 = h//4, 3*h//4
        center_x1, center_x2 = w//4, 3*w//4
        
        # Create mask for darker regions (dog noses are typically dark)
        # Invert so dark regions are white
        _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Also detect using Otsu's method for adaptive thresholding
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combine masks
        combined_mask = cv2.bitwise_and(dark_mask, otsu_mask)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        nose_mask = np.zeros_like(gray)
        nose_contour = None
        
        if contours:
            # Filter contours by area and position (should be in center)
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    # Check if contour is in center region
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Prefer contours near center
                        if center_x1 < cx < center_x2 and center_y1 < cy < center_y2:
                            valid_contours.append((contour, area))
            
            if valid_contours:
                # Get the largest valid contour
                nose_contour = max(valid_contours, key=lambda x: x[1])[0]
                
                # Smooth the contour
                epsilon = 0.005 * cv2.arcLength(nose_contour, True)
                nose_contour = cv2.approxPolyDP(nose_contour, epsilon, True)
                
                cv2.drawContours(nose_mask, [nose_contour], -1, 255, -1)
        
        return nose_mask, nose_contour
    
    def detect_wrinkles(self, image):
        """
        Enhanced wrinkle detection using multiple techniques
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges while smoothing
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive threshold for wrinkle details
        adaptive_thresh = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Canny edge detection
        edges = cv2.Canny(bilateral, 30, 100)
        
        # Combine both
        wrinkles = cv2.bitwise_or(adaptive_thresh, edges)
        
        return wrinkles
    
    def create_nose_outline_visualization(self, image):
        """
        Create visualization with nose outline like the reference images
        """
        # Detect nose region
        nose_mask, nose_contour = self.detect_nose_region(image)
        
        # Create outline image
        outline_image = image.copy()
        
        if nose_contour is not None:
            # Draw thick outline around nose (like in reference images)
            cv2.drawContours(outline_image, [nose_contour], -1, (255, 0, 255), 3)  # Purple outline
            
            # Optionally fill with semi-transparent color
            overlay = image.copy()
            cv2.drawContours(overlay, [nose_contour], -1, (255, 0, 255), -1)
            outline_image = cv2.addWeighted(outline_image, 0.9, overlay, 0.1, 0)
        
        return outline_image, nose_mask, nose_contour
    
    def create_wrinkle_visualization(self, image):
        """
        Create comprehensive wrinkle visualization for ML training
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Wrinkle detection
        wrinkles = self.detect_wrinkles(image)
        
        # Nose outline
        outline_image, nose_mask, nose_contour = self.create_nose_outline_visualization(image)
        
        # Create colorized visualization
        viz = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Overlay wrinkles in red
        viz[wrinkles > 0] = [0, 0, 255]
        
        # Overlay nose outline in green
        if nose_contour is not None:
            cv2.drawContours(viz, [nose_contour], -1, (0, 255, 0), 2)
        
        # Create edges
        edges = cv2.Canny(enhanced, 50, 150)
        
        return viz, wrinkles, nose_mask, edges, nose_contour
    
    def get_visualization(self, image):
        """
        Get different visualization modes
        """
        if self.viz_mode == 0:
            # Normal view with nose outline
            outline_image, _, _ = self.create_nose_outline_visualization(image)
            return outline_image
        
        elif self.viz_mode == 1:
            # Edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        elif self.viz_mode == 2:
            # Adaptive threshold
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        elif self.viz_mode == 3:
            # Wrinkle detection with nose outline
            wrinkles = self.detect_wrinkles(image)
            wrinkle_viz = cv2.cvtColor(wrinkles, cv2.COLOR_GRAY2BGR)
            
            # Add nose outline
            outline_image, _, nose_contour = self.create_nose_outline_visualization(image)
            
            # Overlay wrinkles on original with outline
            alpha = 0.6
            result = cv2.addWeighted(outline_image, alpha, wrinkle_viz, 1-alpha, 0)
            return result
        
        elif self.viz_mode == 4:
            # Full visualization with outline
            viz, wrinkles, nose_mask, edges, nose_contour = self.create_wrinkle_visualization(image)
            return viz
        
        return image
    
    def analyze_image_quality(self, image):
        metrics = {
            'sharpness': self.calculate_sharpness(image),
            'center_sharpness': self.detect_focus_region(image),
            'edge_density': self.calculate_edge_density(image),
            'brightness': np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        }
        
        metrics['quality_score'] = (
            metrics['sharpness'] * 0.5 + 
            metrics['center_sharpness'] * 0.3 + 
            metrics['edge_density'] * 200 * 0.2
        )
        
        return metrics
    
    def save_image(self, image, metrics, is_sharp=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_dir = self.sharp_dir if is_sharp else self.rejected_dir
        
        filename = f"dog_nose_{timestamp}_s{int(metrics['sharpness'])}.jpg"
        filepath = save_dir / filename
        
        # Save original image
        cv2.imwrite(str(filepath), image)
        
        if is_sharp:
            # Save enhanced version
            enhanced = self.enhance_contrast(image)
            enhanced_path = save_dir / f"enhanced_{filename}"
            cv2.imwrite(str(enhanced_path), enhanced)
            
            # Save nose outline visualization
            outline_image, nose_mask, nose_contour = self.create_nose_outline_visualization(image)
            outline_path = self.viz_dir / f"outline_{filename}"
            cv2.imwrite(str(outline_path), outline_image)
            
            # Save full wrinkle visualization
            viz, wrinkles, nose_mask_full, edges, _ = self.create_wrinkle_visualization(image)
            viz_path = self.viz_dir / f"viz_{filename}"
            cv2.imwrite(str(viz_path), viz)
            
            # Save wrinkle map
            wrinkle_path = self.viz_dir / f"wrinkles_{filename}"
            cv2.imwrite(str(wrinkle_path), wrinkles)
            
            # Save nose mask
            mask_path = self.viz_dir / f"nose_mask_{filename}"
            cv2.imwrite(str(mask_path), nose_mask)
            
            # Save edge map
            edge_path = self.viz_dir / f"edges_{filename}"
            cv2.imwrite(str(edge_path), edges)
        
        # Save metadata
        metadata_path = save_dir / f"{filename}.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Sharpness: {metrics['sharpness']:.2f}\n")
            f.write(f"Center Sharpness: {metrics['center_sharpness']:.2f}\n")
            f.write(f"Edge Density: {metrics['edge_density']:.4f}\n")
            f.write(f"Brightness: {metrics['brightness']:.2f}\n")
            f.write(f"Quality Score: {metrics['quality_score']:.2f}\n")
        
        return filepath
    
    def draw_overlay(self, frame, metrics, is_sharp):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw center focus region
        cv2.rectangle(overlay, (w//3, h//3), (2*w//3, 2*h//3), 
                     (0, 255, 0) if is_sharp else (0, 165, 255), 2)
        
        # Status panel
        panel_height = 240
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        
        # Status text
        status = "SHARP - AUTO CAPTURING" if is_sharp else "BLURRY - REFOCUS"
        color = (0, 255, 0) if is_sharp else (0, 165, 255)
        
        cv2.putText(overlay, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Display metrics
        y_offset = 70
        cv2.putText(overlay, f"Sharpness: {metrics['sharpness']:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f"Center: {metrics['center_sharpness']:.1f}", 
                   (10, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f"Edge Density: {metrics['edge_density']:.3f}", 
                   (10, y_offset+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f"Captures: {self.capture_count}", 
                   (10, y_offset+90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Visualization mode
        viz_modes = ["Nose Outline", "Edges", "Threshold", "Wrinkles+Outline", "Full Viz"]
        cv2.putText(overlay, f"View: {viz_modes[self.viz_mode]} (V to change)", 
                   (10, y_offset+120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Controls
        cv2.putText(overlay, "SPACE: Capture | Q: Quit | A: Auto | V: View Mode", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return overlay
    
    def run(self, source=0, resolution=(1920, 1080)):
        print("\n🎥 Starting Dog Nose Capture with Real-time Nose Detection...")
        print("\n📋 Controls:")
        print("  SPACE - Manual capture")
        print("  A     - Toggle auto-capture")
        print("  V     - Change visualization mode")
        print("  Q     - Quit")
        print("\n" + "="*60 + "\n")
        
        # Determine source type
        is_url = isinstance(source, str) and (source.startswith('http://') or source.startswith('https://') or source.startswith('rtsp://'))
        
        if is_url:
            print(f"📡 Connecting to: {source}")
        else:
            print(f"📷 Opening camera: {source}")
        
        # Initialize camera/stream
        cap = cv2.VideoCapture(source)
        
        if not is_url:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        if not cap.isOpened():
            print("❌ Error: Could not open video source")
            return
        
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"✅ Connected! Resolution: {int(actual_width)}x{int(actual_height)}")
        print(f"🎯 Auto-capture: {'Enabled' if self.auto_capture else 'Disabled'}")
        print("\n" + "="*60 + "\n")
        
        frame_count = 0
        last_print_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to grab frame")
                if is_url:
                    print("   Reconnecting...")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(source)
                    if cap.isOpened():
                        print("✅ Reconnected!")
                        continue
                break
            
            frame_count += 1
            
            # Analyze image quality
            metrics = self.analyze_image_quality(frame)
            is_sharp = metrics['sharpness'] >= self.min_sharpness
            
            # Auto-capture logic
            current_time = time.time()
            if (self.auto_capture and is_sharp and 
                current_time - self.last_capture_time >= self.capture_interval):
                
                filepath = self.save_image(frame, metrics, is_sharp=True)
                self.capture_count += 1
                self.last_capture_time = current_time
                print(f"✅ Auto-captured #{self.capture_count}: {filepath.name} "
                      f"(sharpness: {metrics['sharpness']:.1f})")
            
            # Apply visualization
            viz_frame = self.get_visualization(frame)
            
            # Draw overlay
            display_frame = self.draw_overlay(viz_frame, metrics, is_sharp)
            
            # Display
            cv2.imshow('Dog Nose Capture - Real-time Detection', display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n🛑 Quitting...")
                break
            elif key == ord(' '):
                filepath = self.save_image(frame, metrics, is_sharp=is_sharp)
                self.capture_count += 1
                status = "sharp" if is_sharp else "blurry"
                print(f"📸 Manual capture #{self.capture_count}: {filepath.name} "
                      f"(sharpness: {metrics['sharpness']:.1f}, {status})")
            elif key == ord('a'):
                self.auto_capture = not self.auto_capture
                status = "enabled" if self.auto_capture else "disabled"
                print(f"🔄 Auto-capture {status}")
            elif key == ord('v'):
                self.viz_mode = (self.viz_mode + 1) % 5
                viz_modes = ["Nose Outline", "Edges", "Threshold", "Wrinkles+Outline", "Full Visualization"]
                print(f"👁️  View mode: {viz_modes[self.viz_mode]}")
            
            # Print stats
            if current_time - last_print_time >= 5.0 and frame_count > 0:
                fps = frame_count / (current_time - last_print_time)
                print(f"📊 FPS: {fps:.1f} | Sharpness: {metrics['sharpness']:.1f}")
                frame_count = 0
                last_print_time = current_time
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"✅ Session complete!")
        print(f"📊 Total captures: {self.capture_count}")
        print(f"📁 Saved to: {self.output_dir}")
        print(f"📁 Visualizations: {self.viz_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Dog Nose Auto-Capture with Real-time Nose Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dog_nose_capture.py --source http://100.105.10.78:8080/video
  python dog_nose_capture.py --source 0 --sharpness 200
  python dog_nose_capture.py --source http://192.168.1.100:8080/video --interval 1.5
        """
    )
    
    parser.add_argument('--source', '-s', default=0,
                       help='Video source: camera ID or URL')
    parser.add_argument('--output', '-o', default='dog_nose_dataset',
                       help='Output directory')
    parser.add_argument('--sharpness', '-t', type=float, default=150.0,
                       help='Minimum sharpness threshold')
    parser.add_argument('--interval', '-i', type=float, default=2.0,
                       help='Auto-capture interval (seconds)')
    parser.add_argument('--resolution', '-r', default='1920x1080',
                       help='Camera resolution')
    parser.add_argument('--no-auto', action='store_true',
                       help='Disable auto-capture')
    
    args = parser.parse_args()
    
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        resolution = (1920, 1080)
    
    system = DogNoseCaptureSystem(
        output_dir=args.output,
        min_sharpness=args.sharpness,
        auto_capture=not args.no_auto,
        capture_interval=args.interval
    )
    
    try:
        system.run(source=source, resolution=resolution)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print(f"📊 Total captures: {system.capture_count}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
