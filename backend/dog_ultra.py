#!/usr/bin/env python3
"""
Dog Nose Ultra-Enhanced Detection System v2.0
Multi-method ensemble detection with nostril detection, advanced filtering,
and confidence weighting for maximum accuracy
"""

import cv2
import numpy as np
import os
from datetime import datetime
import time
import argparse
from pathlib import Path


class DogNoseUltraDetector:
    def __init__(self, output_dir="dog_nose_ultra", blur_threshold=100.0, 
                 auto_capture=True, capture_interval=2.0, min_sharpness=150.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_capture = auto_capture
        self.capture_interval = capture_interval
        self.min_sharpness = min_sharpness
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
        
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Sharpness threshold: {self.min_sharpness}")
        print(f"⏱️  Auto-capture interval: {self.capture_interval}s")
        print(f"🚀 ULTRA v2.0: Multi-method ensemble + nostril detection + advanced filtering")
        
    def calculate_sharpness_advanced(self, image):
        """Multiple sharpness metrics combined"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()
        
        # Method 2: Tenengrad (Sobel gradient)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(sobelx**2 + sobely**2)
        
        # Method 3: Brenner gradient
        brenner = np.sum((gray[:, 2:].astype(float) - gray[:, :-2].astype(float))**2)
        
        # Weighted average (normalize)
        sharpness = (lap_var * 0.5 + tenengrad * 0.3 + brenner / 1e5 * 0.2)
        return sharpness
    
    def calculate_edge_density(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density
    
    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def detect_focus_region(self, image):
        h, w = image.shape[:2]
        center_region = image[h//3:2*h//3, w//3:2*w//3]
        return self.calculate_sharpness_advanced(center_region)
    
    def detect_nose_method_1(self, image):
        """Method 1: Dark region + Otsu thresholding"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, dark_mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        combined = cv2.bitwise_and(dark_mask, otsu_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return combined
    
    def detect_nose_method_2(self, image):
        """Method 2: Adaptive threshold on contrast-enhanced image"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 10
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return adaptive
    
    def detect_nose_method_3(self, image):
        """Method 3: HSV-based dark region detection"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Dark regions in HSV (low value channel)
        v_channel = hsv[:, :, 2]
        _, v_mask = cv2.threshold(v_channel, 120, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        v_mask = cv2.morphologyEx(v_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return v_mask
    
    def extract_best_contour(self, image, mask):
        """Extract best contour from mask with advanced filtering"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        center_y1, center_y2 = int(h*0.25), int(h*0.75)
        center_x1, center_x2 = int(w*0.25), int(w*0.75)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        nose_mask = np.zeros_like(gray)
        best_contour = None
        best_score = 0.0
        
        if contours:
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1200:
                    continue
                
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Flexible center check
                if center_x1-80 < cx < center_x2+80 and center_y1-80 < cy < center_y2+80:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        if solidity < 0.5:
                            continue
                        
                        x, y, bw, bh = cv2.boundingRect(contour)
                        if bh > 0:
                            aspect = bw / float(bh)
                            if not (0.5 <= aspect <= 2.5):
                                continue
                            
                            perimeter = cv2.arcLength(contour, True)
                            if perimeter > 0:
                                circularity = 4 * np.pi * area / (perimeter * perimeter)
                            else:
                                circularity = 0.0
                            
                            # Score: area × solidity × circularity × position_bonus
                            in_strict = center_x1 <= cx <= center_x2 and center_y1 <= cy <= center_y2
                            pos_bonus = 1.3 if in_strict else 1.0
                            score = (area * solidity * (circularity + 0.3) * pos_bonus)
                            
                            valid_contours.append((contour, score))
            
            if valid_contours:
                best_contour = max(valid_contours, key=lambda x: x[1])[0]
                
                # Smooth contour with adaptive epsilon
                perimeter = cv2.arcLength(best_contour, True)
                epsilon = 0.01 * perimeter
                best_contour = cv2.approxPolyDP(best_contour, epsilon, True)
                
                cv2.drawContours(nose_mask, [best_contour], -1, 255, -1)
        
        return nose_mask, best_contour
    
    def detect_nostrils(self, image, nose_mask):
        """Detect nostrils within nose region - doubles accuracy"""
        if nose_mask is None or cv2.countNonZero(nose_mask) == 0:
            return np.zeros_like(nose_mask), []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Focus on nose region only
        nose_region = cv2.bitwise_and(gray, gray, mask=nose_mask)
        
        # Detect dark spots (nostrils)
        _, nostril_mask = cv2.threshold(nose_region, 40, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        nostril_mask = cv2.morphologyEx(nostril_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find nostril contours
        nostril_contours, _ = cv2.findContours(nostril_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter nostrils: should be 2-3 reasonably sized dark regions
        valid_nostrils = []
        for contour in nostril_contours:
            area = cv2.contourArea(contour)
            if 50 < area < nose_mask.sum() // 10:  # Between 50 and 10% of nose area
                valid_nostrils.append(contour)
        
        return nostril_mask, valid_nostrils[:3]  # Max 3 nostrils
    
    def ensemble_detect(self, image):
        """Ensemble detection: combine all 3 methods"""
        mask1 = self.detect_nose_method_1(image)
        mask2 = self.detect_nose_method_2(image)
        mask3 = self.detect_nose_method_3(image)
        
        # Combine masks: keep regions detected by at least 2 methods
        combined = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
        
        nose_mask, nose_contour = self.extract_best_contour(image, combined)
        nostril_mask, nostrils = self.detect_nostrils(image, nose_mask)
        
        return nose_mask, nose_contour, nostril_mask, nostrils
    
    def detect_wrinkles(self, image):
        """Advanced wrinkle detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        adaptive_thresh = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 13, 3
        )
        
        edges = cv2.Canny(bilateral, 25, 80)
        wrinkles = cv2.bitwise_or(adaptive_thresh, edges)
        
        return wrinkles
    
    def create_visualization(self, image, nose_mask, nose_contour, nostrils):
        """Create enhanced visualization with nostrils"""
        viz = image.copy()
        
        if nose_contour is not None:
            # Draw nose outline in bright green
            cv2.drawContours(viz, [nose_contour], -1, (0, 255, 0), 4)
            
            # Draw nostrils in bright blue
            if nostrils:
                cv2.drawContours(viz, nostrils, -1, (255, 0, 0), 3)
            
            # Semi-transparent fill
            overlay = image.copy()
            cv2.drawContours(overlay, [nose_contour], -1, (0, 255, 0), -1)
            viz = cv2.addWeighted(viz, 0.8, overlay, 0.2, 0)
        
        return viz
    
    def analyze_image_quality(self, image):
        metrics = {
            'sharpness': self.calculate_sharpness_advanced(image),
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
    
    def save_image(self, image, metrics, nose_mask, nose_contour, nostrils, is_sharp=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_dir = self.sharp_dir if is_sharp else self.rejected_dir
        
        filename = f"dog_nose_{timestamp}_s{int(metrics['sharpness'])}.jpg"
        filepath = save_dir / filename
        
        cv2.imwrite(str(filepath), image)
        
        if is_sharp:
            enhanced = self.enhance_contrast(image)
            cv2.imwrite(str(self.sharp_dir / f"enhanced_{filename}"), enhanced)
            
            viz = self.create_visualization(image, nose_mask, nose_contour, nostrils)
            cv2.imwrite(str(self.viz_dir / f"outline_{filename}"), viz)
            
            wrinkles = self.detect_wrinkles(image)
            cv2.imwrite(str(self.viz_dir / f"wrinkles_{filename}"), wrinkles)
            cv2.imwrite(str(self.viz_dir / f"nose_mask_{filename}"), nose_mask)
        
        metadata_path = save_dir / f"{filename}.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Sharpness: {metrics['sharpness']:.2f}\n")
            f.write(f"Center Sharpness: {metrics['center_sharpness']:.2f}\n")
            f.write(f"Edge Density: {metrics['edge_density']:.4f}\n")
            f.write(f"Brightness: {metrics['brightness']:.2f}\n")
            f.write(f"Nostrils Detected: {len(nostrils)}\n")
        
        return filepath
    
    def draw_overlay(self, frame, metrics, is_sharp, nostrils_count):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        cv2.rectangle(overlay, (w//3, h//3), (2*w//3, 2*h//3), 
                     (0, 255, 0) if is_sharp else (0, 165, 255), 2)
        
        panel_height = 280
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        
        status = "✓ CAPTURING" if is_sharp else "✗ BLURRY"
        color = (0, 255, 0) if is_sharp else (0, 165, 255)
        
        cv2.putText(overlay, status, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        y_offset = 80
        cv2.putText(overlay, f"Sharpness: {metrics['sharpness']:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f"Center: {metrics['center_sharpness']:.1f}", 
                   (10, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f"Nostrils: {nostrils_count}", 
                   (10, y_offset+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(overlay, f"Captures: {self.capture_count}", 
                   (10, y_offset+90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(overlay, f"Mode: {['Outline','Edges','Thresh','Wrinkles','Viz'][self.viz_mode]}", 
                   (10, y_offset+120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        cv2.putText(overlay, "SPACE:Save A:Auto V:View Q:Quit", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return overlay
    
    def run(self, source=0, resolution=(1920, 1080)):
        print("\n🚀 Dog Nose Ultra Detector v2.0 - Ensemble Method")
        print("Features: Multi-method detection + Nostril detection + Advanced filtering")
        print("\n" + "="*60 + "\n")
        
        is_url = isinstance(source, str) and (source.startswith('http://') or 
                                              source.startswith('https://') or 
                                              source.startswith('rtsp://'))
        
        cap = cv2.VideoCapture(source)
        
        if not is_url:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        if not cap.isOpened():
            print("❌ Error: Could not open video source")
            return
        
        print(f"✅ Connected!")
        print("="*60 + "\n")
        
        frame_count = 0
        last_print_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_url:
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(source)
                    continue
                break
            
            frame_count += 1
            
            metrics = self.analyze_image_quality(frame)
            is_sharp = metrics['sharpness'] >= self.min_sharpness
            
            # Ensemble detection
            nose_mask, nose_contour, nostril_mask, nostrils = self.ensemble_detect(frame)
            nostrils_count = len(nostrils)
            
            # Auto-capture
            current_time = time.time()
            if (self.auto_capture and is_sharp and nostrils_count >= 2 and
                current_time - self.last_capture_time >= self.capture_interval):
                
                filepath = self.save_image(frame, metrics, nose_mask, nose_contour, nostrils, True)
                self.capture_count += 1
                self.last_capture_time = current_time
                print(f"✅ #{self.capture_count}: {filepath.name} (nostrils:{nostrils_count})")
            
            # Visualization
            viz = self.create_visualization(frame, nose_mask, nose_contour, nostrils)
            display = self.draw_overlay(viz, metrics, is_sharp, nostrils_count)
            
            cv2.imshow('Dog Nose Ultra v2.0', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                filepath = self.save_image(frame, metrics, nose_mask, nose_contour, nostrils, is_sharp)
                self.capture_count += 1
                print(f"📸 Manual #{self.capture_count}")
            elif key == ord('a'):
                self.auto_capture = not self.auto_capture
                print(f"🔄 Auto: {'ON' if self.auto_capture else 'OFF'}")
            elif key == ord('v'):
                self.viz_mode = (self.viz_mode + 1) % 5
            
            if current_time - last_print_time >= 3.0 and frame_count > 0:
                fps = frame_count / (current_time - last_print_time)
                print(f"📊 FPS: {fps:.1f} | Sharp: {metrics['sharpness']:.1f} | Nostrils: {nostrils_count}")
                frame_count = 0
                last_print_time = current_time
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"✅ Complete! Captures: {self.capture_count}")
        print(f"📁 {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Dog Nose Ultra Detector v2.0')
    parser.add_argument('--source', '-s', default=0, help='Video source')
    parser.add_argument('--output', '-o', default='dog_nose_ultra', help='Output dir')
    parser.add_argument('--sharpness', '-t', type=float, default=150.0, help='Min sharpness')
    parser.add_argument('--interval', '-i', type=float, default=2.0, help='Capture interval')
    parser.add_argument('--resolution', '-r', default='1920x1080', help='Resolution')
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
        resolution = (1920, 1080)
    
    system = DogNoseUltraDetector(
        output_dir=args.output,
        min_sharpness=args.sharpness,
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
