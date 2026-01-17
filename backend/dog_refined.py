import cv2
import numpy as np
import os
from datetime import datetime
import time
import argparse
from pathlib import Path


class DogNoseCaptureSystemRefined:
    def __init__(self, output_dir="dog_nose_dataset_refined", blur_threshold=100.0, 
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
        print(f"🔍 REFINED: Improved center flexibility & contour filtering")
        
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
        REFINED: Improved nose detection with better center flexibility
        and stronger morphological filtering
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # IMPROVED: Wider center region (30-70% instead of 25-75%)
        center_y1, center_y2 = int(h*0.3), int(h*0.7)
        center_x1, center_x2 = int(w*0.3), int(w*0.7)
        
        # IMPROVED: Lower threshold to 80 for darker noses
        _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Otsu's method
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combine masks
        combined_mask = cv2.bitwise_and(dark_mask, otsu_mask)
        
        # IMPROVED: Stronger morphological operations (5-7 kernel size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Additional noise reduction with larger kernel
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        nose_mask = np.zeros_like(gray)
        nose_contour = None
        
        if contours:
            # IMPROVED: Stronger filtering with shape metrics
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                # IMPROVED: Minimum area 1500-2000
                if area < 1500:
                    continue
                
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                    
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # IMPROVED: Accept slightly outside center, but prefer inside
                if center_x1-50 < cx < center_x2+50 and center_y1-50 < cy < center_y2+50:
                    # Calculate shape metrics
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        # IMPROVED: Require high solidity (≥0.6)
                        if solidity >= 0.6:
                            x, y, w, h = cv2.boundingRect(contour)
                            if h > 0:
                                aspect = w / float(h)
                                # Dog noses are roughly square (0.6-2.0)
                                if 0.6 <= aspect <= 2.0:
                                    # Calculate score: area * solidity * position_bonus
                                    in_strict_center = center_x1 <= cx <= center_x2 and center_y1 <= cy <= center_y2
                                    position_bonus = 1.2 if in_strict_center else 1.0
                                    score = area * solidity * position_bonus
                                    valid_contours.append((contour, score))
            
            if valid_contours:
                # Get the best contour by score
                nose_contour = max(valid_contours, key=lambda x: x[1])[0]
                
                # IMPROVED: Stronger smoothing (0.008 instead of 0.005)
                epsilon = 0.008 * cv2.arcLength(nose_contour, True)
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
        Create visualization with nose outline
        """
        nose_mask, nose_contour = self.detect_nose_region(image)
        
        outline_image = image.copy()
        
        if nose_contour is not None:
            # Draw green outline (more visible than purple)
            cv2.drawContours(outline_image, [nose_contour], -1, (0, 255, 0), 3)
            
            # Semi-transparent fill
            overlay = image.copy()
            cv2.drawContours(overlay, [nose_contour], -1, (0, 255, 0), -1)
            outline_image = cv2.addWeighted(outline_image, 0.85, overlay, 0.15, 0)
        
        return outline_image, nose_mask, nose_contour
    
    def create_wrinkle_visualization(self, image):
        """
        Create comprehensive wrinkle visualization for ML training
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        wrinkles = self.detect_wrinkles(image)
        
        outline_image, nose_mask, nose_contour = self.create_nose_outline_visualization(image)
        
        viz = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        viz[wrinkles > 0] = [0, 0, 255]
        
        if nose_contour is not None:
            cv2.drawContours(viz, [nose_contour], -1, (0, 255, 0), 2)
        
        edges = cv2.Canny(enhanced, 50, 150)
        
        return viz, wrinkles, nose_mask, edges, nose_contour
    
    def get_visualization(self, image):
        """
        Get different visualization modes
        """
        if self.viz_mode == 0:
            outline_image, _, _ = self.create_nose_outline_visualization(image)
            return outline_image
        
        elif self.viz_mode == 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        elif self.viz_mode == 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        elif self.viz_mode == 3:
            wrinkles = self.detect_wrinkles(image)
            wrinkle_viz = cv2.cvtColor(wrinkles, cv2.COLOR_GRAY2BGR)
            
            outline_image, _, nose_contour = self.create_nose_outline_visualization(image)
            
            alpha = 0.6
            result = cv2.addWeighted(outline_image, alpha, wrinkle_viz, 1-alpha, 0)
            return result
        
        elif self.viz_mode == 4:
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
        
        cv2.imwrite(str(filepath), image)
        
        if is_sharp:
            enhanced = self.enhance_contrast(image)
            enhanced_path = save_dir / f"enhanced_{filename}"
            cv2.imwrite(str(enhanced_path), enhanced)
            
            outline_image, nose_mask, nose_contour = self.create_nose_outline_visualization(image)
            outline_path = self.viz_dir / f"outline_{filename}"
            cv2.imwrite(str(outline_path), outline_image)
            
            viz, wrinkles, nose_mask_full, edges, _ = self.create_wrinkle_visualization(image)
            viz_path = self.viz_dir / f"viz_{filename}"
            cv2.imwrite(str(viz_path), viz)
            
            wrinkle_path = self.viz_dir / f"wrinkles_{filename}"
            cv2.imwrite(str(wrinkle_path), wrinkles)
            
            mask_path = self.viz_dir / f"nose_mask_{filename}"
            cv2.imwrite(str(mask_path), nose_mask)
            
            edge_path = self.viz_dir / f"edges_{filename}"
            cv2.imwrite(str(edge_path), edges)
        
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
        
        cv2.rectangle(overlay, (w//3, h//3), (2*w//3, 2*h//3), 
                     (0, 255, 0) if is_sharp else (0, 165, 255), 2)
        
        panel_height = 240
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        
        status = "SHARP - AUTO CAPTURING" if is_sharp else "BLURRY - REFOCUS"
        color = (0, 255, 0) if is_sharp else (0, 165, 255)
        
        cv2.putText(overlay, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        y_offset = 70
        cv2.putText(overlay, f"Sharpness: {metrics['sharpness']:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f"Center: {metrics['center_sharpness']:.1f}", 
                   (10, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f"Edge Density: {metrics['edge_density']:.3f}", 
                   (10, y_offset+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f"Captures: {self.capture_count}", 
                   (10, y_offset+90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        viz_modes = ["Nose Outline", "Edges", "Threshold", "Wrinkles+Outline", "Full Viz"]
        cv2.putText(overlay, f"View: {viz_modes[self.viz_mode]} (V to change)", 
                   (10, y_offset+120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        cv2.putText(overlay, "SPACE: Capture | Q: Quit | A: Auto | V: View Mode", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return overlay
    
    def run(self, source=0, resolution=(1920, 1080)):
        print("\n🎥 Starting Dog Nose Capture - REFINED VERSION...")
        print("\n📋 Controls:")
        print("  SPACE - Manual capture")
        print("  A     - Toggle auto-capture")
        print("  V     - Change visualization mode")
        print("  Q     - Quit")
        print("\n" + "="*60 + "\n")
        
        is_url = isinstance(source, str) and (source.startswith('http://') or source.startswith('https://') or source.startswith('rtsp://'))
        
        if is_url:
            print(f"📡 Connecting to: {source}")
        else:
            print(f"📷 Opening camera: {source}")
        
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
            
            metrics = self.analyze_image_quality(frame)
            is_sharp = metrics['sharpness'] >= self.min_sharpness
            
            current_time = time.time()
            if (self.auto_capture and is_sharp and 
                current_time - self.last_capture_time >= self.capture_interval):
                
                filepath = self.save_image(frame, metrics, is_sharp=True)
                self.capture_count += 1
                self.last_capture_time = current_time
                print(f"✅ Auto-captured #{self.capture_count}: {filepath.name} "
                      f"(sharpness: {metrics['sharpness']:.1f})")
            
            viz_frame = self.get_visualization(frame)
            display_frame = self.draw_overlay(viz_frame, metrics, is_sharp)
            
            cv2.imshow('Dog Nose Capture - REFINED', display_frame)
            
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
        description='Dog Nose Auto-Capture - REFINED with Improved Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dog_nose_capture_refined.py --source http://192.168.1.100:8080/video
  python dog_nose_capture_refined.py --source 0 --sharpness 200
  python dog_nose_capture_refined.py --source http://192.168.1.100:8080/video --interval 1.5
        """
    )
    
    parser.add_argument('--source', '-s', default=0,
                       help='Video source: camera ID or URL')
    parser.add_argument('--output', '-o', default='dog_nose_dataset_refined',
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
    
    system = DogNoseCaptureSystemRefined(
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