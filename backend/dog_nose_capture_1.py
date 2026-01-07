#!/usr/bin/env python3
"""
Dog Nose Auto-Capture System with High-Accuracy Detection
Only captures when nose is detected with high confidence
Uses multiple detection techniques combined
"""

import cv2
import numpy as np
import os
from datetime import datetime
import time
import argparse
from pathlib import Path


class AccurateDogNoseDetector:
    def __init__(self, output_dir="dog_nose_dataset", min_sharpness=150.0, 
                 auto_capture=True, capture_interval=2.0, min_confidence=0.75):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_capture = auto_capture
        self.capture_interval = capture_interval
        self.min_sharpness = min_sharpness
        self.min_confidence = min_confidence  # Confidence threshold (0-1)
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
    
    def detect_nose_multi_method(self, image):
        """
        Ultra-accurate nose detection using multiple methods
        Returns: nose_mask, nose_contour, confidence_score
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Define center region (nose should be here)
        center_y1, center_y2 = h//4, 3*h//4
        center_x1, center_x2 = w//4, 3*w//4
        
        confidence_scores = []
        
        # METHOD 1: Dark region detection (noses are dark)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, dark_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # METHOD 2: Texture analysis (noses have wrinkle texture)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        sobel_mag = np.uint8(sobel_mag / sobel_mag.max() * 255)
        _, texture_mask = cv2.threshold(sobel_mag, 30, 255, cv2.THRESH_BINARY)
        
        # METHOD 3: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 15, 5)
        
        # Combine all methods
        combined = cv2.bitwise_and(dark_mask, adaptive)
        combined = cv2.bitwise_and(combined, texture_mask)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_confidence = 0.0
        nose_mask = np.zeros_like(gray)
        
        if contours:
            for contour in contours:
                confidence = self.validate_nose_contour(contour, h, w, center_x1, center_x2, 
                                                         center_y1, center_y2, gray)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_contour = contour
        
        if best_contour is not None and best_confidence >= self.min_confidence:
            # Smooth contour
            epsilon = 0.003 * cv2.arcLength(best_contour, True)
            best_contour = cv2.approxPolyDP(best_contour, epsilon, True)
            cv2.drawContours(nose_mask, [best_contour], -1, 255, -1)
        else:
            best_contour = None
            best_confidence = 0.0
        
        return nose_mask, best_contour, best_confidence
    
    def validate_nose_contour(self, contour, h, w, cx1, cx2, cy1, cy2, gray):
        """
        Validate if contour is actually a dog nose
        Returns confidence score 0-1
        """
        confidence = 0.0
        
        # Check 1: Area (nose should be reasonable size)
        area = cv2.contourArea(contour)
        total_area = h * w
        area_ratio = area / total_area
        
        if area < 2000 or area > total_area * 0.6:
            return 0.0  # Too small or too large
        
        # Area score (ideal is 5-30% of image)
        if 0.05 <= area_ratio <= 0.3:
            confidence += 0.25
        elif 0.03 <= area_ratio <= 0.5:
            confidence += 0.15
        
        # Check 2: Center position
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0.0
        
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        
        # Nose should be in center region
        if cx1 < centroid_x < cx2 and cy1 < centroid_y < cy2:
            # Calculate distance from center
            center_x, center_y = w // 2, h // 2
            distance = np.sqrt((centroid_x - center_x)**2 + (centroid_y - center_y)**2)
            max_distance = np.sqrt((w//4)**2 + (h//4)**2)
            
            position_score = 1.0 - (distance / max_distance)
            confidence += 0.25 * position_score
        else:
            return 0.0  # Not in center region
        
        # Check 3: Aspect ratio (nose is roughly square or slightly wider)
        x, y, width, height = cv2.boundingRect(contour)
        if height == 0:
            return 0.0
        
        aspect_ratio = width / float(height)
        
        # Ideal aspect ratio is 0.8 - 1.5
        if 0.7 <= aspect_ratio <= 1.8:
            confidence += 0.15
        elif 0.5 <= aspect_ratio <= 2.5:
            confidence += 0.08
        
        # Check 4: Solidity (how solid vs hollow)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            
            # Nose should be fairly solid (0.7-0.95)
            if 0.65 <= solidity <= 0.98:
                confidence += 0.15
            elif 0.5 <= solidity:
                confidence += 0.08
        
        # Check 5: Extent (ratio of contour area to bounding box)
        rect_area = width * height
        if rect_area > 0:
            extent = area / rect_area
            
            # Should be reasonably filled (0.5-0.9)
            if 0.5 <= extent <= 0.92:
                confidence += 0.10
            elif 0.4 <= extent:
                confidence += 0.05
        
        # Check 6: Shape complexity (perimeter vs area)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Nose is somewhat round but not perfectly circular
            if 0.3 <= circularity <= 0.85:
                confidence += 0.10
            elif 0.2 <= circularity:
                confidence += 0.05
        
        return min(confidence, 1.0)
    
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
        """Create visualization based on mode"""
        if self.viz_mode == 0:
            # Nose outline with confidence
            viz = image.copy()
            if nose_contour is not None and confidence >= self.min_confidence:
                # Color based on confidence (green=high, yellow=medium, red=low)
                if confidence >= 0.85:
                    color = (0, 255, 0)  # Green
                elif confidence >= 0.75:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 165, 255)  # Orange
                
                cv2.drawContours(viz, [nose_contour], -1, color, 4)
                
                # Semi-transparent fill
                overlay = image.copy()
                cv2.drawContours(overlay, [nose_contour], -1, color, -1)
                viz = cv2.addWeighted(viz, 0.92, overlay, 0.08, 0)
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
            
            if nose_contour is not None and confidence >= self.min_confidence:
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
            
            if nose_contour is not None and confidence >= self.min_confidence:
                cv2.drawContours(viz, [nose_contour], -1, (0, 255, 0), 3)
            
            return viz
        
        return image
    
    def save_image(self, image, sharpness, confidence, nose_mask, nose_contour):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        filename = f"nose_{timestamp}_s{int(sharpness)}_c{int(confidence*100)}.jpg"
        filepath = self.sharp_dir / filename
        
        # Save original
        cv2.imwrite(str(filepath), image)
        
        # Save enhanced
        enhanced = self.enhance_contrast(image)
        enhanced_path = self.sharp_dir / f"enhanced_{filename}"
        cv2.imwrite(str(enhanced_path), enhanced)
        
        # Save wrinkles
        wrinkles = self.detect_wrinkles(image)
        wrinkle_path = self.viz_dir / f"wrinkles_{filename}"
        cv2.imwrite(str(wrinkle_path), wrinkles)
        
        # Save nose mask
        mask_path = self.viz_dir / f"mask_{filename}"
        cv2.imwrite(str(mask_path), nose_mask)
        
        # Save outline visualization
        outline_viz = image.copy()
        if nose_contour is not None:
            cv2.drawContours(outline_viz, [nose_contour], -1, (0, 255, 0), 3)
        outline_path = self.viz_dir / f"outline_{filename}"
        cv2.imwrite(str(outline_path), outline_viz)
        
        # Save full viz
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        viz[wrinkles > 0] = [0, 0, 255]
        if nose_contour is not None:
            cv2.drawContours(viz, [nose_contour], -1, (0, 255, 0), 2)
        viz_path = self.viz_dir / f"viz_{filename}"
        cv2.imwrite(str(viz_path), viz)
        
        # Save metadata
        meta_path = self.sharp_dir / f"{filename}.txt"
        with open(meta_path, 'w') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Sharpness: {sharpness:.2f}\n")
            f.write(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)\n")
            f.write(f"Detection Status: {'VALID' if confidence >= self.min_confidence else 'REJECTED'}\n")
        
        return filepath
    
    def draw_overlay(self, frame, sharpness, confidence, nose_detected):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Status panel
        panel_height = 280
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        
        # Detection status
        if nose_detected and confidence >= self.min_confidence:
            status = "NOSE DETECTED - READY TO CAPTURE"
            color = (0, 255, 0)  # Green
        elif nose_detected:
            status = "NOSE FOUND - LOW CONFIDENCE"
            color = (0, 255, 255)  # Yellow
        else:
            status = "NO NOSE DETECTED - ADJUST POSITION"
            color = (0, 0, 255)  # Red
        
        cv2.putText(overlay, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Metrics
        y = 70
        cv2.putText(overlay, f"Sharpness: {sharpness:.1f}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Confidence bar
        cv2.putText(overlay, f"Confidence: {confidence*100:.1f}%", 
                   (10, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw confidence bar
        bar_x, bar_y = 150, y+20
        bar_w, bar_h = 200, 20
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (100, 100, 100), 2)
        
        filled_w = int(bar_w * confidence)
        if confidence >= 0.85:
            bar_color = (0, 255, 0)
        elif confidence >= 0.75:
            bar_color = (0, 255, 255)
        elif confidence >= 0.5:
            bar_color = (0, 165, 255)
        else:
            bar_color = (0, 0, 255)
        
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x+filled_w, bar_y+bar_h), bar_color, -1)
        
        # Threshold line
        threshold_x = bar_x + int(bar_w * self.min_confidence)
        cv2.line(overlay, (threshold_x, bar_y-5), (threshold_x, bar_y+bar_h+5), (255, 255, 255), 2)
        
        cv2.putText(overlay, f"Threshold: {self.min_confidence*100:.0f}%", 
                   (10, y+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(overlay, f"Captures: {self.capture_count}", 
                   (10, y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # View mode
        modes = ["Outline", "Wrinkles", "Wrinkles+Outline", "Full Viz"]
        cv2.putText(overlay, f"View: {modes[self.viz_mode]} (V)", 
                   (10, y+135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Instructions
        cv2.putText(overlay, "Position dog nose in center until GREEN status", 
                   (10, y+170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls
        cv2.putText(overlay, "SPACE: Manual | A: Auto | V: View | Q: Quit", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return overlay
    
    def run(self, source=0, resolution=(1920, 1080)):
        print("\n🎥 High-Accuracy Dog Nose Detection System")
        print("\n⚠️  IMPORTANT: System will ONLY capture when:")
        print(f"   1. Sharpness > {self.min_sharpness}")
        print(f"   2. Confidence > {self.min_confidence*100}%")
        print(f"   3. Nose is properly detected and validated")
        print("\n📋 Controls:")
        print("  SPACE - Manual capture (if confidence is high)")
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
        
        print(f"✅ Connected!")
        print(f"🎯 Waiting for high-confidence nose detection...\n")
        
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
            nose_mask, nose_contour, confidence = self.detect_nose_multi_method(frame)
            sharpness = self.calculate_sharpness(frame)
            
            nose_detected = nose_contour is not None
            high_confidence = confidence >= self.min_confidence
            is_sharp = sharpness >= self.min_sharpness
            
            # Auto-capture ONLY if all conditions met
            current_time = time.time()
            if (self.auto_capture and high_confidence and is_sharp and nose_detected and
                current_time - self.last_capture_time >= self.capture_interval):
                
                filepath = self.save_image(frame, sharpness, confidence, nose_mask, nose_contour)
                self.capture_count += 1
                self.last_capture_time = current_time
                print(f"✅ AUTO-CAPTURED #{self.capture_count}: {filepath.name}")
                print(f"   Sharpness: {sharpness:.1f} | Confidence: {confidence*100:.1f}%\n")
            
            # Create visualization
            viz_frame = self.create_visualization(frame, nose_mask, nose_contour, confidence)
            display_frame = self.draw_overlay(viz_frame, sharpness, confidence, nose_detected)
            
            cv2.imshow('High-Accuracy Dog Nose Detection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                if high_confidence and is_sharp and nose_detected:
                    filepath = self.save_image(frame, sharpness, confidence, nose_mask, nose_contour)
                    self.capture_count += 1
                    print(f"📸 MANUAL CAPTURE #{self.capture_count}: {filepath.name}")
                    print(f"   Sharpness: {sharpness:.1f} | Confidence: {confidence*100:.1f}%\n")
                else:
                    print(f"⚠️  Cannot capture - Confidence too low ({confidence*100:.1f}%)")
            elif key == ord('a'):
                self.auto_capture = not self.auto_capture
                print(f"🔄 Auto-capture: {'ENABLED' if self.auto_capture else 'DISABLED'}")
            elif key == ord('v'):
                self.viz_mode = (self.viz_mode + 1) % 4
                modes = ["Outline", "Wrinkles", "Wrinkles+Outline", "Full Viz"]
                print(f"👁️  View: {modes[self.viz_mode]}")
            
            # Stats
            if current_time - last_print >= 5.0:
                fps = frame_count / (current_time - last_print)
                print(f"📊 FPS: {fps:.1f} | Sharpness: {sharpness:.1f} | Confidence: {confidence*100:.1f}%")
                frame_count = 0
                last_print = current_time
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"✅ Session complete!")
        print(f"📊 Total valid captures: {self.capture_count}")
        print(f"📁 Saved to: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='High-Accuracy Dog Nose Detection and Capture System'
    )
    
    parser.add_argument('--source', '-s', default=0,
                       help='Video source: camera ID or URL')
    parser.add_argument('--output', '-o', default='dog_nose_dataset',
                       help='Output directory')
    parser.add_argument('--sharpness', '-t', type=float, default=150.0,
                       help='Minimum sharpness (default: 150)')
    parser.add_argument('--confidence', '-c', type=float, default=0.75,
                       help='Minimum confidence 0-1 (default: 0.75)')
    parser.add_argument('--interval', '-i', type=float, default=2.0,
                       help='Capture interval seconds (default: 2.0)')
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
        print(f"\n⚠️  Interrupted - {system.capture_count} captures saved")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
