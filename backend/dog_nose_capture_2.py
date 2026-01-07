#!/usr/bin/env python3
"""
Dog Nose Auto-Capture System with Fixed Detection
Improved detection algorithm with better confidence scoring
"""

import cv2
import numpy as np
import os
from datetime import datetime
import time
import argparse
from pathlib import Path


class AccurateDogNoseDetector:
    def __init__(self, output_dir="dog_nose_dataset", min_sharpness=100.0, 
                 auto_capture=True, capture_interval=1.5, min_confidence=0.60):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_capture = auto_capture
        self.capture_interval = capture_interval
        self.min_sharpness = min_sharpness
        self.min_confidence = min_confidence
        self.last_capture_time = 0
        self.capture_count = 0
        
        self.viz_mode = 0
        
        # Create subdirectories
        self.sharp_dir = self.output_dir / "sharp"
        self.rejected_dir = self.output_dir / "rejected"
        self.viz_dir = self.output_dir / "visualizations"
        self.sharp_dir.mkdir(exist_ok=True)
        self.rejected_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)
        
        print(f"📁 Output: {self.output_dir}")
        print(f"🎯 Sharpness threshold: {self.min_sharpness}")
        print(f"🎯 Confidence threshold: {self.min_confidence * 100}%")
        print(f"⏱️  Interval: {self.capture_interval}s")
        
    def calculate_sharpness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def detect_nose_improved(self, image):
        """
        Improved nose detection with relaxed constraints
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Center region where nose should be
        center_margin = 0.25
        center_x1 = int(w * center_margin)
        center_x2 = int(w * (1 - center_margin))
        center_y1 = int(h * center_margin)
        center_y2 = int(h * (1 - center_margin))
        
        # Step 1: Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Step 2: Multiple thresholding methods
        # Method A: Otsu thresholding (good for dark noses)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Method B: Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 8
        )
        
        # Method C: Simple threshold for very dark regions
        _, simple = cv2.threshold(enhanced, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Combine methods - if any method detects it, include it
        combined = cv2.bitwise_or(otsu, adaptive)
        combined = cv2.bitwise_or(combined, simple)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Fill small holes
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_large)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(gray), None, 0.0
        
        # Find best contour
        best_contour = None
        best_confidence = 0.0
        
        for contour in contours:
            confidence = self.score_contour(contour, h, w, center_x1, center_x2, center_y1, center_y2)
            if confidence > best_confidence:
                best_confidence = confidence
                best_contour = contour
        
        # Create mask
        nose_mask = np.zeros_like(gray)
        if best_contour is not None and best_confidence > 0.3:  # Lower threshold for detection
            # Smooth the contour
            epsilon = 0.005 * cv2.arcLength(best_contour, True)
            best_contour = cv2.approxPolyDP(best_contour, epsilon, True)
            cv2.drawContours(nose_mask, [best_contour], -1, 255, -1)
        else:
            best_contour = None
            best_confidence = 0.0
        
        return nose_mask, best_contour, best_confidence
    
    def score_contour(self, contour, h, w, cx1, cx2, cy1, cy2):
        """
        Score contour with more lenient criteria
        """
        score = 0.0
        
        # Get properties
        area = cv2.contourArea(contour)
        if area < 500:  # Too small
            return 0.0
        
        total_area = h * w
        area_ratio = area / total_area
        
        # Check if too large or too small
        if area_ratio > 0.7:  # Takes up more than 70% of image
            return 0.0
        
        # Score 1: Area (15 points) - More lenient
        if 0.03 <= area_ratio <= 0.5:
            score += 0.15
        elif 0.01 <= area_ratio <= 0.7:
            score += 0.08
        else:
            score += 0.03
        
        # Get centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0.0
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Score 2: Position (30 points) - Must be in center region
        in_center = cx1 <= cx <= cx2 and cy1 <= cy <= cy2
        
        if in_center:
            # Calculate how close to center
            center_x, center_y = w // 2, h // 2
            dist_x = abs(cx - center_x) / (w / 2)
            dist_y = abs(cy - center_y) / (h / 2)
            avg_dist = (dist_x + dist_y) / 2
            
            # Closer to center = higher score
            position_score = 1.0 - (avg_dist * 0.5)  # Max penalty 50%
            score += 0.30 * position_score
        else:
            # Outside center, but give some points if close
            score += 0.05
        
        # Score 3: Aspect ratio (15 points)
        x, y, width, height = cv2.boundingRect(contour)
        if height > 0:
            aspect = width / float(height)
            # Dog noses are roughly square to slightly wider
            if 0.6 <= aspect <= 2.0:
                score += 0.15
            elif 0.4 <= aspect <= 3.0:
                score += 0.08
            else:
                score += 0.03
        
        # Score 4: Solidity (15 points)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if 0.50 <= solidity <= 0.99:
                score += 0.15
            elif 0.30 <= solidity:
                score += 0.08
        
        # Score 5: Extent (10 points)
        rect_area = width * height
        if rect_area > 0:
            extent = area / rect_area
            if 0.40 <= extent <= 0.95:
                score += 0.10
            elif 0.25 <= extent:
                score += 0.05
        
        # Score 6: Circularity (15 points)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if 0.25 <= circularity <= 0.90:
                score += 0.15
            elif 0.15 <= circularity:
                score += 0.08
        
        return min(score, 1.0)
    
    def detect_wrinkles(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        adaptive_thresh = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        edges = cv2.Canny(bilateral, 25, 80)
        wrinkles = cv2.bitwise_or(adaptive_thresh, edges)
        
        return wrinkles
    
    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def create_visualization(self, image, nose_mask, nose_contour, confidence):
        if self.viz_mode == 0:
            # Nose outline with confidence
            viz = image.copy()
            if nose_contour is not None:
                # Color based on confidence
                if confidence >= 0.75:
                    color = (0, 255, 0)  # Green - excellent
                elif confidence >= 0.60:
                    color = (0, 255, 255)  # Yellow - good
                elif confidence >= 0.40:
                    color = (0, 165, 255)  # Orange - fair
                else:
                    color = (0, 100, 255)  # Red-orange - poor
                
                cv2.drawContours(viz, [nose_contour], -1, color, 4)
                
                # Semi-transparent fill
                overlay = image.copy()
                cv2.drawContours(overlay, [nose_contour], -1, color, -1)
                viz = cv2.addWeighted(viz, 0.90, overlay, 0.10, 0)
            return viz
        
        elif self.viz_mode == 1:
            # Wrinkles
            wrinkles = self.detect_wrinkles(image)
            return cv2.cvtColor(wrinkles, cv2.COLOR_GRAY2BGR)
        
        elif self.viz_mode == 2:
            # Wrinkles with outline
            viz = image.copy()
            wrinkles = self.detect_wrinkles(image)
            viz[wrinkles > 0] = [0, 0, 255]
            
            if nose_contour is not None:
                cv2.drawContours(viz, [nose_contour], -1, (0, 255, 0), 2)
            
            return viz
        
        elif self.viz_mode == 3:
            # Full viz with enhanced contrast
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            wrinkles = self.detect_wrinkles(image)
            viz = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            viz[wrinkles > 0] = [0, 0, 255]
            
            if nose_contour is not None:
                cv2.drawContours(viz, [nose_contour], -1, (0, 255, 0), 3)
            
            return viz
        
        return image
    
    def save_image(self, image, sharpness, confidence, nose_mask, nose_contour):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        filename = f"nose_{timestamp}_s{int(sharpness)}_c{int(confidence*100)}.jpg"
        filepath = self.sharp_dir / filename
        
        cv2.imwrite(str(filepath), image)
        
        enhanced = self.enhance_contrast(image)
        cv2.imwrite(str(self.sharp_dir / f"enhanced_{filename}"), enhanced)
        
        wrinkles = self.detect_wrinkles(image)
        cv2.imwrite(str(self.viz_dir / f"wrinkles_{filename}"), wrinkles)
        
        cv2.imwrite(str(self.viz_dir / f"mask_{filename}"), nose_mask)
        
        outline_viz = image.copy()
        if nose_contour is not None:
            cv2.drawContours(outline_viz, [nose_contour], -1, (0, 255, 0), 3)
        cv2.imwrite(str(self.viz_dir / f"outline_{filename}"), outline_viz)
        
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        viz[wrinkles > 0] = [0, 0, 255]
        if nose_contour is not None:
            cv2.drawContours(viz, [nose_contour], -1, (0, 255, 0), 2)
        cv2.imwrite(str(self.viz_dir / f"viz_{filename}"), viz)
        
        with open(self.sharp_dir / f"{filename}.txt", 'w') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Sharpness: {sharpness:.2f}\n")
            f.write(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)\n")
            f.write(f"Status: {'VALID' if confidence >= self.min_confidence else 'LOW_CONF'}\n")
        
        return filepath
    
    def draw_overlay(self, frame, sharpness, confidence, nose_detected):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        panel_height = 300
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        
        # Status
        if nose_detected and confidence >= self.min_confidence:
            status = "NOSE DETECTED - CAPTURING"
            color = (0, 255, 0)
        elif nose_detected and confidence >= 0.40:
            status = "NOSE FOUND - LOW CONFIDENCE"
            color = (0, 255, 255)
        elif nose_detected:
            status = "POSSIBLE NOSE - VERY LOW CONFIDENCE"
            color = (0, 165, 255)
        else:
            status = "NO NOSE - ADJUST POSITION"
            color = (0, 0, 255)
        
        cv2.putText(overlay, status, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        
        # Metrics
        y = 80
        cv2.putText(overlay, f"Sharpness: {sharpness:.1f} (min: {self.min_sharpness})", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Confidence with bar
        cv2.putText(overlay, f"Confidence: {confidence*100:.1f}%", 
                   (10, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Confidence bar
        bar_x, bar_y = 10, y+55
        bar_w, bar_h = 300, 25
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (80, 80, 80), -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (150, 150, 150), 2)
        
        filled_w = int(bar_w * min(confidence, 1.0))
        if confidence >= 0.75:
            bar_color = (0, 255, 0)
        elif confidence >= 0.60:
            bar_color = (0, 255, 255)
        elif confidence >= 0.40:
            bar_color = (0, 165, 255)
        else:
            bar_color = (0, 100, 255)
        
        if filled_w > 0:
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x+filled_w, bar_y+bar_h), bar_color, -1)
        
        # Threshold marker
        threshold_x = bar_x + int(bar_w * self.min_confidence)
        cv2.line(overlay, (threshold_x, bar_y), (threshold_x, bar_y+bar_h), (255, 255, 255), 3)
        cv2.putText(overlay, "MIN", (threshold_x-15, bar_y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # More info
        cv2.putText(overlay, f"Threshold: {self.min_confidence*100:.0f}% | Captures: {self.capture_count}", 
                   (10, y+95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        modes = ["Outline", "Wrinkles", "Wrinkles+Outline", "Full Viz"]
        cv2.putText(overlay, f"View Mode: {modes[self.viz_mode]} (Press V to change)", 
                   (10, y+125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(overlay, "TIP: Position dog nose in CENTER of frame", 
                   (10, y+155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)
        
        cv2.putText(overlay, "SPACE: Manual | A: Auto | V: View | Q: Quit", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        
        return overlay
    
    def run(self, source=0, resolution=(1280, 720)):
        print("\n🎥 Dog Nose Detection System - IMPROVED DETECTION")
        print("\n✅ System will capture when:")
        print(f"   - Sharpness > {self.min_sharpness}")
        print(f"   - Confidence > {self.min_confidence*100}%")
        print(f"   - Nose properly detected in center")
        print("\n📋 Controls:")
        print("  SPACE - Manual capture")
        print("  A     - Toggle auto-capture")
        print("  V     - Change view mode")
        print("  Q     - Quit")
        print("\n" + "="*60 + "\n")
        
        is_url = isinstance(source, str) and source.startswith(('http://', 'https://', 'rtsp://'))
        
        cap = cv2.VideoCapture(source)
        
        if not is_url:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        if not cap.isOpened():
            print("❌ Could not open video source")
            return
        
        print(f"✅ Connected! Detecting noses...\n")
        
        frame_count = 0
        last_print = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_url:
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(source)
                    if cap.isOpened():
                        continue
                break
            
            frame_count += 1
            
            # Detect nose
            nose_mask, nose_contour, confidence = self.detect_nose_improved(frame)
            sharpness = self.calculate_sharpness(frame)
            
            nose_detected = nose_contour is not None
            meets_confidence = confidence >= self.min_confidence
            is_sharp = sharpness >= self.min_sharpness
            
            # Auto-capture
            current_time = time.time()
            if (self.auto_capture and meets_confidence and is_sharp and nose_detected and
                current_time - self.last_capture_time >= self.capture_interval):
                
                filepath = self.save_image(frame, sharpness, confidence, nose_mask, nose_contour)
                self.capture_count += 1
                self.last_capture_time = current_time
                print(f"✅ CAPTURED #{self.capture_count}: {filepath.name}")
                print(f"   Sharp: {sharpness:.1f} | Conf: {confidence*100:.1f}%\n")
            
            # Visualization
            viz_frame = self.create_visualization(frame, nose_mask, nose_contour, confidence)
            display_frame = self.draw_overlay(viz_frame, sharpness, confidence, nose_detected)
            
            cv2.imshow('Dog Nose Detection - IMPROVED', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                if meets_confidence and is_sharp and nose_detected:
                    filepath = self.save_image(frame, sharpness, confidence, nose_mask, nose_contour)
                    self.capture_count += 1
                    print(f"📸 MANUAL #{self.capture_count}: {filepath.name}")
                    print(f"   Sharp: {sharpness:.1f} | Conf: {confidence*100:.1f}%\n")
                else:
                    print(f"⚠️  Low confidence ({confidence*100:.1f}%) or sharpness ({sharpness:.1f})")
            elif key == ord('a'):
                self.auto_capture = not self.auto_capture
                print(f"🔄 Auto: {'ON' if self.auto_capture else 'OFF'}")
            elif key == ord('v'):
                self.viz_mode = (self.viz_mode + 1) % 4
                modes = ["Outline", "Wrinkles", "Wrinkles+Outline", "Full Viz"]
                print(f"👁️  View: {modes[self.viz_mode]}")
            
            if current_time - last_print >= 3.0:
                fps = frame_count / (current_time - last_print)
                print(f"📊 FPS: {fps:.1f} | Sharp: {sharpness:.1f} | Conf: {confidence*100:.1f}% {'✓' if meets_confidence else '✗'}")
                frame_count = 0
                last_print = current_time
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"✅ Done! Captured: {self.capture_count}")
        print(f"📁 Location: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Dog Nose Detection - IMPROVED')
    
    parser.add_argument('--source', '-s', default=0, help='Video source')
    parser.add_argument('--output', '-o', default='dog_nose_dataset', help='Output dir')
    parser.add_argument('--sharpness', '-t', type=float, default=100.0, help='Min sharpness (default: 100)')
    parser.add_argument('--confidence', '-c', type=float, default=0.60, help='Min confidence 0-1 (default: 0.60)')
    parser.add_argument('--interval', '-i', type=float, default=1.5, help='Capture interval (default: 1.5s)')
    parser.add_argument('--resolution', '-r', default='1280x720', help='Resolution')
    parser.add_argument('--no-auto', action='store_true', help='Disable auto-capture')
    
    args = parser.parse_args()
    
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        resolution = (1280, 720)
    
    system = AccurateDogNoseDetector(
        output_dir=args.output,
        min_sharpness=args.sharpness,
        min_confidence=args.confidence,
        auto_capture=not args.no_auto,
        capture_interval=args.interval
    )
    
    try:
        system.run(source=source, resolution=resolution)
    except KeyboardInterrupt:
        print(f"\n⚠️  Stopped - {system.capture_count} captures")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
