"""Enhanced live camera tool for dog nose-print dataset collection with auto-capture.

Features:
- Multi-metric sharpness detection (Laplacian + Tenengrad + Brenner)
- Auto-capture when quality conditions met (stable for N frames)
- Real-time quality feedback (sharpness, contrast, brightness, focus)
- Multiple preprocessing modes (edges, CLAHE, adaptive threshold, morphology)
- Automatic dataset organization with metadata
- Countdown timer before auto-capture
- Batch capture mode for multi-angle dataset collection

Controls:
- space: toggle edge overlay modes (off → edges → enhanced → threshold → morph)
- s: force save current frame
- a: toggle auto-capture mode
- r: reset capture counter
- q: quit

Source:
- Default webcam (index 0) or IP stream:
    python main.py --source http://<phone-ip>:8080/video

Options:
- --roi-fraction: square ROI as fraction of shorter side (default 0.5)
- --target-count: number of images to auto-capture (default 5)
- --stable-frames: frames that must pass quality check before capture (default 5)
- --min-sharpness: minimum Laplacian variance (default 150.0)
- --cooldown: seconds between auto-captures (default 2.5)
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np


# ============================================================================
# QUALITY METRICS
# ============================================================================

def compute_laplacian_variance(gray: np.ndarray) -> float:
    """Variance of Laplacian - standard sharpness metric."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_tenengrad(gray: np.ndarray) -> float:
    """Tenengrad (Sobel-based) sharpness - better for edge detection."""
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return float(np.mean(gradient_magnitude))


def compute_brenner_gradient(gray: np.ndarray) -> float:
    """Brenner gradient - fast sharpness alternative."""
    h, w = gray.shape
    if w < 3:
        return 0.0
    brenner = np.sum((gray[:, 2:].astype(float) - gray[:, :-2].astype(float))**2)
    return float(brenner / (h * w))


def compute_local_contrast(gray: np.ndarray) -> float:
    """Local contrast using standard deviation."""
    return float(np.std(gray))


def compute_brightness(gray: np.ndarray) -> float:
    """Average brightness."""
    return float(np.mean(gray))


def check_quality(
    gray: np.ndarray,
    min_laplacian: float = 150.0,
    min_tenengrad: float = 200.0,
    min_contrast: float = 60.0,
    min_brightness: float = 50.0,
    max_brightness: float = 200.0
) -> Dict[str, any]:
    """Comprehensive quality check with multiple metrics."""
    
    laplacian = compute_laplacian_variance(gray)
    tenengrad = compute_tenengrad(gray)
    brenner = compute_brenner_gradient(gray)
    contrast = compute_local_contrast(gray)
    brightness = compute_brightness(gray)
    
    checks = {
        'laplacian': laplacian >= min_laplacian,
        'tenengrad': tenengrad >= min_tenengrad,
        'contrast': contrast >= min_contrast,
        'brightness': min_brightness <= brightness <= max_brightness,
    }
    
    return {
        'passed': all(checks.values()),
        'laplacian': laplacian,
        'tenengrad': tenengrad,
        'brenner': brenner,
        'contrast': contrast,
        'brightness': brightness,
        'checks': checks
    }


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def enhance_clahe(gray: np.ndarray) -> np.ndarray:
    """CLAHE enhancement for contrast."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def enhance_adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Adaptive thresholding to highlight nose patterns."""
    enhanced = enhance_clahe(gray)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    return thresh


def enhance_morphology(gray: np.ndarray) -> np.ndarray:
    """Morphological operations to enhance ridge patterns."""
    thresh = enhance_adaptive_threshold(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    return morph


def process_frame(
    frame: np.ndarray,
    mode: str = "edges"
) -> Tuple[np.ndarray, np.ndarray]:
    """Process frame with selected enhancement mode.
    
    Args:
        frame: Input BGR frame
        mode: One of ["edges", "enhanced", "threshold", "morph", "off"]
    
    Returns:
        (annotated_frame, processed_grayscale)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    annotated = frame.copy()
    
    if mode == "off":
        return annotated, gray
    
    elif mode == "edges":
        # Original edge detection with improved preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        smooth = cv2.bilateralFilter(equalized, d=5, sigmaColor=75, sigmaSpace=75)
        edges = cv2.Canny(smooth, threshold1=40, threshold2=120)
        annotated[edges > 0] = (0, 255, 255)  # Yellow grooves
        return annotated, edges
    
    elif mode == "enhanced":
        # CLAHE enhancement
        enhanced = enhance_clahe(gray)
        edges = cv2.Canny(enhanced, threshold1=50, threshold2=150)
        annotated[edges > 0] = (0, 255, 0)  # Green edges
        return annotated, enhanced
    
    elif mode == "threshold":
        # Adaptive threshold
        thresh = enhance_adaptive_threshold(gray)
        annotated = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return annotated, thresh
    
    elif mode == "morph":
        # Morphological enhancement
        morph = enhance_morphology(gray)
        annotated = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        return annotated, morph
    
    return annotated, gray


# ============================================================================
# AUTO-CAPTURE LOGIC
# ============================================================================

class AutoCaptureState:
    """Manages auto-capture state and timing."""
    
    def __init__(
        self,
        target_count: int = 5,
        stable_frames: int = 5,
        cooldown_sec: float = 2.5
    ):
        self.target_count = target_count
        self.stable_frames = stable_frames
        self.cooldown_sec = cooldown_sec
        
        self.capture_count = 0
        self.sharp_frame_count = 0
        self.last_capture_time = 0.0
        self.enabled = False
    
    def reset(self):
        """Reset capture counter."""
        self.capture_count = 0
        self.sharp_frame_count = 0
    
    def update(self, quality_passed: bool) -> bool:
        """Update state and return True if should capture.
        
        Args:
            quality_passed: Whether current frame passes quality checks
        
        Returns:
            True if conditions met for auto-capture
        """
        if not self.enabled:
            return False
        
        if self.capture_count >= self.target_count:
            return False
        
        current_time = time.time()
        
        if quality_passed:
            self.sharp_frame_count += 1
        else:
            self.sharp_frame_count = 0
        
        # Check if ready to capture
        if (self.sharp_frame_count >= self.stable_frames and
            current_time - self.last_capture_time > self.cooldown_sec):
            self.last_capture_time = current_time
            self.capture_count += 1
            self.sharp_frame_count = 0
            return True
        
        return False
    
    def get_countdown(self) -> Optional[int]:
        """Get countdown number if approaching capture."""
        if self.sharp_frame_count >= self.stable_frames - 3:
            remaining = self.stable_frames - self.sharp_frame_count
            return remaining if remaining > 0 else None
        return None


# ============================================================================
# SAVING & DATASET MANAGEMENT
# ============================================================================

def save_capture(
    frame: np.ndarray,
    roi_coords: Tuple[int, int, int, int],
    quality_metrics: Dict,
    output_dir: Path,
    capture_id: int
) -> Path:
    """Save captured frame with all enhancement versions and metadata.
    
    Args:
        frame: Original BGR frame
        roi_coords: (x1, y1, x2, y2) ROI coordinates
        quality_metrics: Quality metrics dictionary
        output_dir: Base output directory
        capture_id: Capture sequence number
    
    Returns:
        Path to saved original image
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    x1, y1, x2, y2 = roi_coords
    
    # Extract ROI from original frame (no overlays)
    roi = frame[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Create organized directory structure
    dirs = {
        'original': output_dir / 'original',
        'enhanced': output_dir / 'enhanced',
        'threshold': output_dir / 'threshold',
        'morph': output_dir / 'morph',
        'metadata': output_dir / 'metadata',
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # Save original (most important for training)
    original_path = dirs['original'] / f"nose_{capture_id:03d}_{timestamp}.jpg"
    cv2.imwrite(str(original_path), roi, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    # Save enhanced versions
    enhanced = enhance_clahe(roi_gray)
    thresh = enhance_adaptive_threshold(roi_gray)
    morph = enhance_morphology(roi_gray)
    
    cv2.imwrite(str(dirs['enhanced'] / f"nose_{capture_id:03d}_{timestamp}.jpg"), enhanced)
    cv2.imwrite(str(dirs['threshold'] / f"nose_{capture_id:03d}_{timestamp}.jpg"), thresh)
    cv2.imwrite(str(dirs['morph'] / f"nose_{capture_id:03d}_{timestamp}.jpg"), morph)
    
    # Save metadata as JSON
    metadata = {
        'capture_id': capture_id,
        'timestamp': timestamp,
        'quality_metrics': {
            'laplacian': round(quality_metrics['laplacian'], 2),
            'tenengrad': round(quality_metrics['tenengrad'], 2),
            'brenner': round(quality_metrics['brenner'], 2),
            'contrast': round(quality_metrics['contrast'], 2),
            'brightness': round(quality_metrics['brightness'], 2),
        },
        'quality_passed': quality_metrics['passed'],
        'roi_coords': roi_coords,
        'roi_size': (x2 - x1, y2 - y1),
    }
    
    metadata_path = dirs['metadata'] / f"nose_{capture_id:03d}_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return original_path


# ============================================================================
# UI RENDERING
# ============================================================================

def draw_ui(
    display: np.ndarray,
    roi_coords: Tuple[int, int, int, int],
    quality: Dict,
    auto_capture: AutoCaptureState,
    mode: str
) -> None:
    """Draw UI overlays on display frame (in-place)."""
    h, w = display.shape[:2]
    x1, y1, x2, y2 = roi_coords
    
    # Draw ROI rectangle
    roi_color = (0, 255, 0) if quality['passed'] else (0, 0, 255)
    cv2.rectangle(display, (x1, y1), (x2, y2), roi_color, 3)
    
    # Quality metrics panel (top-left)
    panel_y = 30
    text_color = (0, 255, 0) if quality['passed'] else (0, 0, 255)
    
    metrics = [
        f"Laplacian: {quality['laplacian']:.1f} {'✓' if quality['checks']['laplacian'] else '✗'}",
        f"Tenengrad: {quality['tenengrad']:.1f} {'✓' if quality['checks']['tenengrad'] else '✗'}",
        f"Contrast: {quality['contrast']:.1f} {'✓' if quality['checks']['contrast'] else '✗'}",
        f"Brightness: {quality['brightness']:.1f} {'✓' if quality['checks']['brightness'] else '✗'}",
    ]
    
    for metric in metrics:
        cv2.putText(display, metric, (10, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        panel_y += 25
    
    # Stable frame indicator
    if auto_capture.enabled and quality['passed']:
        progress = auto_capture.sharp_frame_count
        total = auto_capture.stable_frames
        cv2.putText(display, f"Stable: {progress}/{total}", 
                   (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw progress bar
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = h - 70
        filled = int(bar_width * progress / total)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height), 
                     (0, 255, 0), -1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
    
    # Countdown
    countdown = auto_capture.get_countdown()
    if countdown is not None and auto_capture.enabled:
        cv2.putText(display, str(countdown), (w//2 - 50, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
    
    # Capture counter
    counter_text = f"Captured: {auto_capture.capture_count}/{auto_capture.target_count}"
    auto_status = "[AUTO]" if auto_capture.enabled else "[MANUAL]"
    cv2.putText(display, f"{counter_text} {auto_status}", 
               (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Mode indicator
    cv2.putText(display, f"Mode: {mode.upper()}", 
               (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Controls help
    help_text = "space:mode  a:auto  s:save  r:reset  q:quit"
    cv2.putText(display, help_text, 
               (10, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


# ============================================================================
# MAIN LOOP
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhanced dog nose-print dataset collection tool"
    )
    parser.add_argument(
        "--source", 
        default="0", 
        help="Camera index or IP stream URL"
    )
    parser.add_argument(
        "--roi-fraction",
        type=float,
        default=0.5,
        help="Square ROI fraction of shorter image side (0-1, default 0.5)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=5,
        help="Number of images to auto-capture (default 5)",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=5,
        help="Frames that must pass quality check before capture (default 5)",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=150.0,
        help="Minimum Laplacian variance for sharpness (default 150.0)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.5,
        help="Seconds between auto-captures (default 2.5)",
    )
    args = parser.parse_args()
    
    # Initialize camera
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera source: {source}")
    
    # Set camera properties for best quality
    capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Setup output directory
    output_dir = Path(__file__).resolve().parent / "dataset"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize state
    auto_capture_state = AutoCaptureState(
        target_count=args.target_count,
        stable_frames=args.stable_frames,
        cooldown_sec=args.cooldown
    )
    
    modes = ["off", "edges", "enhanced", "threshold", "morph"]
    current_mode_idx = 1  # Start with edges
    roi_fraction = max(0.1, min(1.0, args.roi_fraction))
    
    print("🐕 Enhanced Dog Nose Dataset Collection")
    print(f"📁 Saving to: {output_dir}")
    print(f"🎯 Target: {args.target_count} captures")
    print(f"📊 Quality threshold: Laplacian > {args.min_sharpness}")
    print("\nControls:")
    print("  space - cycle through visualization modes")
    print("  a     - toggle auto-capture")
    print("  s     - force save current frame")
    print("  r     - reset capture counter")
    print("  q     - quit")
    
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        
        # Calculate ROI coordinates
        h, w = frame.shape[:2]
        side = int(min(h, w) * roi_fraction)
        side = max(1, side)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        roi_coords = (x1, y1, x2, y2)
        
        # Extract ROI for quality check
        roi = frame[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Check quality
        quality = check_quality(
            roi_gray,
            min_laplacian=args.min_sharpness,
            min_tenengrad=200.0,
            min_contrast=60.0,
            min_brightness=50.0,
            max_brightness=200.0
        )
        
        # Update auto-capture state
        should_capture = auto_capture_state.update(quality['passed'])
        
        # Process frame with selected mode
        current_mode = modes[current_mode_idx]
        annotated, processed = process_frame(frame, mode=current_mode)
        
        # Apply ROI mask to visualization
        if current_mode != "off":
            display = frame.copy()
            display[y1:y2, x1:x2] = annotated[y1:y2, x1:x2]
        else:
            display = frame.copy()
        
        # Draw UI overlays
        draw_ui(display, roi_coords, quality, auto_capture_state, current_mode)
        
        # Auto-capture
        if should_capture:
            save_path = save_capture(
                frame, roi_coords, quality, output_dir, 
                auto_capture_state.capture_count
            )
            print(f"✅ Auto-captured #{auto_capture_state.capture_count}: {save_path.name}")
            print(f"   Laplacian: {quality['laplacian']:.1f}, "
                  f"Tenengrad: {quality['tenengrad']:.1f}")
            
            # Flash effect
            display[:] = (255, 255, 255)
            cv2.imshow("Dog Nose Dataset Collection", display)
            cv2.waitKey(100)
        
        # Display
        cv2.imshow("Dog Nose Dataset Collection", display)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        elif key == ord(" "):
            # Cycle through modes
            current_mode_idx = (current_mode_idx + 1) % len(modes)
            print(f"📸 Mode: {modes[current_mode_idx]}")
        elif key == ord("a"):
            # Toggle auto-capture
            auto_capture_state.enabled = not auto_capture_state.enabled
            status = "ENABLED" if auto_capture_state.enabled else "DISABLED"
            print(f"🤖 Auto-capture: {status}")
        elif key == ord("s"):
            # Force save
            auto_capture_state.capture_count += 1
            save_path = save_capture(
                frame, roi_coords, quality, output_dir,
                auto_capture_state.capture_count
            )
            print(f"💾 Manual save #{auto_capture_state.capture_count}: {save_path.name}")
        elif key == ord("r"):
            # Reset counter
            auto_capture_state.reset()
            print("🔄 Capture counter reset")
        
        # Auto-exit when target reached
        if auto_capture_state.capture_count >= auto_capture_state.target_count:
            print(f"\n✅ Target reached! Captured {auto_capture_state.target_count} images")
            print(f"📁 Dataset saved to: {output_dir}")
            time.sleep(2)
            break
    
    capture.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Final count: {auto_capture_state.capture_count}/{auto_capture_state.target_count}")
    print(f"📁 Dataset location: {output_dir}")


if __name__ == "__main__":
    main()
