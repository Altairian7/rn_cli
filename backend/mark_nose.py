"""
Batch Nose Marking Tool - Process multiple dog nose images with threshold visualization.

Features:
- Batch process folder of dog nose images
- Apply CLAHE + adaptive threshold preprocessing
- Detect nose contours automatically
- Mark with purple polygon overlay (matching your dataset style)
- Save processed images in organized output folder
- Generate dataset labels file

Usage:
    python mark_noses.py --input dataset/original --output dataset/marked

Requirements:
    pip install opencv-python numpy
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def enhance_clahe(gray: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def apply_adaptive_threshold(gray: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    """Apply adaptive thresholding to highlight nose patterns."""
    enhanced = enhance_clahe(gray)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=C
    )
    return thresh


def apply_morphology(thresh: np.ndarray) -> np.ndarray:
    """Apply morphological operations to clean up noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    return morph


# ============================================================================
# NOSE CONTOUR DETECTION
# ============================================================================

def detect_nose_contour(image: np.ndarray, min_area: int = 5000) -> Optional[np.ndarray]:
    """
    Detect the main nose contour from image.
    
    Args:
        image: Input BGR image
        min_area: Minimum contour area to consider
    
    Returns:
        Largest contour polygon or None if not found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing
    thresh = apply_adaptive_threshold(gray)
    morph = apply_morphology(thresh)
    
    # Invert if needed (we want nose region to be white)
    # Check which color is dominant in center
    h, w = morph.shape
    center_region = morph[h//3:2*h//3, w//3:2*w//3]
    if np.mean(center_region) < 128:
        morph = cv2.bitwise_not(morph)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter by area and select largest
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if not valid_contours:
        return None
    
    # Get largest contour (main nose region)
    nose_contour = max(valid_contours, key=cv2.contourArea)
    
    # Approximate polygon
    epsilon = 0.01 * cv2.arcLength(nose_contour, True)
    approx = cv2.approxPolyDP(nose_contour, epsilon, True)
    
    return approx


def create_ellipse_mask(image_shape: Tuple[int, int], center: Tuple[int, int], 
                        axes: Tuple[int, int]) -> np.ndarray:
    """Create elliptical mask for nose region."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def detect_nose_ellipse(image: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Detect nose region and return ellipse parameters.
    
    Returns:
        (center, axes) for ellipse
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Default ellipse (centered, covering 60% of image)
    center = (w // 2, h // 2)
    axes = (int(w * 0.35), int(h * 0.4))
    
    # Try to refine based on edge detection
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Fit ellipse to largest contour
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5:  # Need at least 5 points to fit ellipse
            try:
                ellipse = cv2.fitEllipse(largest)
                center = tuple(map(int, ellipse[0]))
                axes = tuple(map(lambda x: int(x / 2), ellipse[1]))
            except:
                pass
    
    return center, axes


# ============================================================================
# MARKING AND VISUALIZATION
# ============================================================================

def mark_nose_with_polygon(image: np.ndarray, contour: np.ndarray, 
                          color: Tuple[int, int, int] = (128, 0, 128),
                          thickness: int = 3) -> np.ndarray:
    """
    Mark nose region with polygon overlay (purple outline matching dataset style).
    
    Args:
        image: Input BGR image
        contour: Nose contour polygon
        color: BGR color for marking (default: purple)
        thickness: Line thickness
    
    Returns:
        Image with marked nose
    """
    marked = image.copy()
    
    # Draw polygon outline
    cv2.drawContours(marked, [contour], -1, color, thickness)
    
    return marked


def create_threshold_overlay(image: np.ndarray, threshold_image: np.ndarray,
                            alpha: float = 0.6) -> np.ndarray:
    """
    Create overlay with threshold visualization (matching your dataset style).
    
    Args:
        image: Original BGR image
        threshold_image: Binary threshold image
        alpha: Overlay transparency
    
    Returns:
        Image with threshold overlay
    """
    # Convert threshold to colored overlay
    overlay = image.copy()
    
    # Create yellow/green overlay for threshold regions
    threshold_colored = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR)
    threshold_colored[threshold_image > 0] = [0, 255, 255]  # Yellow for grooves
    
    # Blend with original
    result = cv2.addWeighted(overlay, 1 - alpha, threshold_colored, alpha, 0)
    
    return result


def create_dataset_style_visualization(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Create visualization matching your dataset style:
    - Purple polygon outline
    - Threshold/edge overlay
    - Clean, professional look
    """
    h, w = image.shape[:2]
    
    # Create threshold visualization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = apply_adaptive_threshold(gray)
    
    # Start with original image
    result = image.copy()
    
    # Apply threshold overlay inside contour region
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Create yellow/cyan edge overlay
    edges = cv2.Canny(thresh, 50, 150)
    result[edges > 0] = [0, 255, 255]  # Yellow grooves
    
    # Draw purple polygon outline
    cv2.drawContours(result, [contour], -1, (128, 0, 128), 3)
    
    return result


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_single_image(
    image_path: Path,
    output_dir: Path,
    visualization_mode: str = "polygon"
) -> Optional[dict]:
    """
    Process a single image and save marked version.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output
        visualization_mode: "polygon", "threshold", or "dataset"
    
    Returns:
        Metadata dict or None if processing failed
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠️  Failed to load: {image_path.name}")
        return None
    
    h, w = image.shape[:2]
    
    # Detect nose contour
    contour = detect_nose_contour(image)
    
    if contour is None:
        # Fallback: use ellipse
        print(f"⚠️  No contour detected for {image_path.name}, using ellipse fallback")
        center, axes = detect_nose_ellipse(image)
        
        # Create ellipse contour
        ellipse_points = cv2.ellipse2Poly(center, axes, 0, 0, 360, 10)
        contour = ellipse_points.reshape(-1, 1, 2)
    
    # Create visualization based on mode
    if visualization_mode == "polygon":
        marked = mark_nose_with_polygon(image, contour)
    elif visualization_mode == "threshold":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = apply_adaptive_threshold(gray)
        marked = create_threshold_overlay(image, thresh)
        cv2.drawContours(marked, [contour], -1, (128, 0, 128), 3)
    elif visualization_mode == "dataset":
        marked = create_dataset_style_visualization(image, contour)
    else:
        marked = image.copy()
    
    # Save marked image
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), marked, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Generate metadata
    contour_points = contour.squeeze().tolist()
    metadata = {
        "filename": image_path.name,
        "image_size": {"width": w, "height": h},
        "nose_contour": contour_points,
        "contour_area": int(cv2.contourArea(contour)),
        "visualization_mode": visualization_mode
    }
    
    return metadata


def batch_process_folder(
    input_dir: Path,
    output_dir: Path,
    visualization_mode: str = "dataset",
    save_metadata: bool = True
) -> None:
    """
    Batch process all images in a folder.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save marked images
        visualization_mode: Visualization style
        save_metadata: Whether to save labels.json file
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"🔍 Found {len(image_files)} images to process")
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🎨 Mode: {visualization_mode}")
    print("-" * 60)
    
    # Process each image
    metadata_list = []
    success_count = 0
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}...", end=" ")
        
        metadata = process_single_image(image_path, output_dir, visualization_mode)
        
        if metadata:
            metadata_list.append(metadata)
            success_count += 1
            print("✅")
        else:
            print("❌")
    
    print("-" * 60)
    print(f"✅ Successfully processed: {success_count}/{len(image_files)}")
    
    # Save metadata file
    if save_metadata and metadata_list:
        labels_path = output_dir / "labels.json"
        with open(labels_path, 'w') as f:
            json.dump({
                "total_images": len(metadata_list),
                "visualization_mode": visualization_mode,
                "images": metadata_list
            }, f, indent=2)
        print(f"📄 Labels saved to: {labels_path}")


# ============================================================================
# INTERACTIVE SINGLE IMAGE MARKER
# ============================================================================

def interactive_mark_single_image(image_path: Path, output_path: Optional[Path] = None) -> None:
    """
    Interactive tool to mark a single image with keyboard controls.
    
    Controls:
        1-4: Switch visualization modes
        s: Save marked image
        q: Quit
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Cannot load image: {image_path}")
        return
    
    modes = ["polygon", "threshold", "dataset", "original"]
    mode_names = {
        "polygon": "Purple Polygon Only",
        "threshold": "Threshold Overlay",
        "dataset": "Dataset Style (Threshold + Polygon)",
        "original": "Original Image"
    }
    current_mode = 0
    
    # Detect contour once
    contour = detect_nose_contour(image)
    if contour is None:
        print("⚠️  No contour detected, using ellipse fallback")
        center, axes = detect_nose_ellipse(image)
        ellipse_points = cv2.ellipse2Poly(center, axes, 0, 0, 360, 10)
        contour = ellipse_points.reshape(-1, 1, 2)
    
    print("🎨 Interactive Nose Marker")
    print("Controls:")
    print("  1-4: Switch visualization modes")
    print("  s: Save current visualization")
    print("  q: Quit")
    
    while True:
        mode = modes[current_mode]
        
        # Create visualization
        if mode == "polygon":
            display = mark_nose_with_polygon(image, contour)
        elif mode == "threshold":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = apply_adaptive_threshold(gray)
            display = create_threshold_overlay(image, thresh)
            cv2.drawContours(display, [contour], -1, (128, 0, 128), 3)
        elif mode == "dataset":
            display = create_dataset_style_visualization(image, contour)
        else:  # original
            display = image.copy()
        
        # Add mode indicator
        cv2.putText(display, f"Mode: {mode_names[mode]}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Press 1-4 to switch | s:save | q:quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Nose Marker", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_mode = 0
        elif key == ord('2'):
            current_mode = 1
        elif key == ord('3'):
            current_mode = 2
        elif key == ord('4'):
            current_mode = 3
        elif key == ord('s'):
            if output_path is None:
                output_path = image_path.parent / f"marked_{image_path.name}"
            cv2.imwrite(str(output_path), display)
            print(f"💾 Saved to: {output_path}")
    
    cv2.destroyAllWindows()


# ============================================================================
# MAIN
# ============================================================================

def capture_and_process_video(
    source: str,
    output_dir: Path,
    visualization_mode: str = "dataset",
    frame_interval: int = 5,
    max_frames: int = 100
) -> None:
    """
    Capture frames from video source and process them.
    
    Args:
        source: Video file path or URL (e.g., http://ip:port/video)
        output_dir: Directory to save marked frames
        visualization_mode: Visualization style
        frame_interval: Process every Nth frame
        max_frames: Maximum number of frames to capture
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎥 Capturing video from: {source}")
    print(f"📁 Output: {output_dir}")
    print(f"🎨 Mode: {visualization_mode}")
    print(f"⏱️  Frame interval: {frame_interval}, Max frames: {max_frames}")
    print("-" * 60)
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video source: {source}")
        return
    
    frame_count = 0
    processed_count = 0
    metadata_list = []
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                print("📹 End of video stream reached")
                break
            
            frame_count += 1
            
            # Process every Nth frame
            if frame_count % frame_interval != 0:
                continue
            
            processed_count += 1
            frame_name = f"frame_{frame_count:06d}.jpg"
            
            print(f"[{processed_count}] Processing frame {frame_count}...", end=" ")
            
            # Create temporary path for processing
            temp_path = output_dir / frame_name
            
            # Save frame temporarily
            cv2.imwrite(str(temp_path), frame)
            
            # Detect and mark nose
            contour = detect_nose_contour(frame)
            
            if contour is None:
                print("⚠️  (no contour, using ellipse)")
                center, axes = detect_nose_ellipse(frame)
                ellipse_points = cv2.ellipse2Poly(center, axes, 0, 0, 360, 10)
                contour = ellipse_points.reshape(-1, 1, 2)
            else:
                print("✅")
            
            # Create visualization
            if visualization_mode == "polygon":
                marked = mark_nose_with_polygon(frame, contour)
            elif visualization_mode == "threshold":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                thresh = apply_adaptive_threshold(gray)
                marked = create_threshold_overlay(frame, thresh)
                cv2.drawContours(marked, [contour], -1, (128, 0, 128), 3)
            elif visualization_mode == "dataset":
                marked = create_dataset_style_visualization(frame, contour)
            else:
                marked = frame.copy()
            
            # Save marked frame
            cv2.imwrite(str(temp_path), marked, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Generate metadata
            h, w = frame.shape[:2]
            contour_points = contour.squeeze().tolist()
            metadata = {
                "filename": frame_name,
                "frame_number": frame_count,
                "image_size": {"width": w, "height": h},
                "nose_contour": contour_points,
                "contour_area": int(cv2.contourArea(contour)),
                "visualization_mode": visualization_mode
            }
            metadata_list.append(metadata)
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
    
    finally:
        cap.release()
    
    print("-" * 60)
    print(f"✅ Captured {frame_count} total frames")
    print(f"✅ Processed {processed_count} frames")
    
    # Save metadata
    if metadata_list:
        labels_path = output_dir / "labels.json"
        with open(labels_path, 'w') as f:
            json.dump({
                "total_frames_captured": frame_count,
                "total_frames_processed": processed_count,
                "frame_interval": frame_interval,
                "visualization_mode": visualization_mode,
                "frames": metadata_list
            }, f, indent=2)
        print(f"📄 Labels saved to: {labels_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Mark dog nose regions with polygon overlay and threshold visualization"
    )
    
    # Input source (mutually exclusive: input or source)
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    input_group.add_argument(
        '--input', '-i',
        type=str,
        help='Input directory or single image file'
    )
    
    input_group.add_argument(
        '--source', '-s',
        type=str,
        help='Video source (file path or URL like http://ip:port/video)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory (default: input_dir/marked or source_dir/marked)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['polygon', 'threshold', 'dataset'],
        default='dataset',
        help='Visualization mode (default: dataset)'
    )
    
    parser.add_argument(
        '--interactive', '-int',
        action='store_true',
        help='Launch interactive mode for single image'
    )
    
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Skip saving labels.json metadata file'
    )
    
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=5,
        help='Process every Nth frame from video (default: 5)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=100,
        help='Maximum number of frames to capture from video (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Handle video source
    if args.source:
        output_dir = Path(args.output) if args.output else Path('./video_marked')
        capture_and_process_video(
            args.source,
            output_dir,
            visualization_mode=args.mode,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames
        )
    
    # Handle input (directory or image)
    elif args.input:
        input_path = Path(args.input)
        
        if not input_path.exists():
            print(f"❌ Input path does not exist: {input_path}")
            return
        
        # Single image processing
        if input_path.is_file():
            if args.interactive:
                output_path = Path(args.output) if args.output else None
                interactive_mark_single_image(input_path, output_path)
            else:
                output_dir = Path(args.output) if args.output else input_path.parent / 'marked'
                output_dir.mkdir(parents=True, exist_ok=True)
                metadata = process_single_image(input_path, output_dir, args.mode)
                if metadata:
                    print(f"✅ Processed: {input_path.name}")
                    print(f"📁 Saved to: {output_dir / input_path.name}")
        
        # Batch folder processing
        elif input_path.is_dir():
            output_dir = Path(args.output) if args.output else input_path / 'marked'
            batch_process_folder(
                input_path,
                output_dir,
                visualization_mode=args.mode,
                save_metadata=not args.no_metadata
            )
        
        else:
            print(f"❌ Invalid input path: {input_path}")
    
    else:
        print("❌ Either --input or --source must be specified")


if __name__ == "__main__":
    main()
