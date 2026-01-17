"""
Dog Nose Pattern Fingerprint Extractor

Extracts ONLY the nose leather (rhinarium) and its unique ridge/crease patterns
for use as a biometric fingerprint. Like human fingerprints, the pattern of
lines, ridges, and creases on a dog's nose are completely unique.

Requirements:
    pip install torch torchvision opencv-python numpy pillow scikit-learn scipy scikit-image

Usage:
    # Extract nose pattern fingerprint
    python nose_fingerprint.py --input dog_nose.jpg --output nose_pattern.jpg
    
    # Enroll a dog with their nose pattern
    python nose_fingerprint.py --enroll --input dog_nose.jpg --name "Max" --db nose_db.json
    
    # Identify a dog from nose pattern
    python nose_fingerprint.py --identify --input unknown_nose.jpg --db nose_db.json
"""

import cv2
import numpy as np
import torch
import argparse
import json
from pathlib import Path
from datetime import datetime
import hashlib
from scipy.spatial.distance import cosine, euclidean
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.morphology import skeletonize
from sklearn.preprocessing import normalize
import warnings
import time
import threading
warnings.filterwarnings('ignore')


class NosePatternExtractor:
    """
    Extracts the unique ridge and crease pattern from a dog's nose leather.
    
    The rhinarium (nose leather) has unique patterns similar to fingerprints:
    - Ridge patterns and orientations
    - Crease depths and directions
    - Texture discontinuities
    - Pore patterns
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load segmentation model
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 
                                     'deeplabv3_resnet101', 
                                     pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
    def isolate_nose_leather(self, image):
        """
        Precisely isolate ONLY the nose leather (rhinarium) - the dark, moist part
        This is the area with the unique ridge patterns
        """
        h, w = image.shape[:2]
        
        # Convert to multiple color spaces for better segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Nose leather is typically:
        # 1. Dark (low value in HSV)
        # 2. Low saturation (grayish/black)
        # 3. In the upper-center region
        
        # Create region of interest - upper center where nose is
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[int(h*0.15):int(h*0.7), int(w*0.2):int(w*0.8)] = 255
        
        # Multi-stage segmentation
        
        # Stage 1: Darkness threshold (nose leather is dark)
        _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Stage 2: Color-based segmentation (nose leather is desaturated)
        lower_nose = np.array([0, 0, 0])
        upper_nose = np.array([180, 80, 100])
        color_mask = cv2.inRange(hsv, lower_nose, upper_nose)
        
        # Stage 3: Texture-based (nose leather has specific texture)
        # Use Laplacian to find textured regions
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_score = np.abs(laplacian)
        texture_score = (texture_score / texture_score.max() * 255).astype(np.uint8)
        _, texture_mask = cv2.threshold(texture_score, 20, 255, cv2.THRESH_BINARY)
        
        # Combine all masks
        combined_mask = cv2.bitwise_and(dark_mask, color_mask)
        combined_mask = cv2.bitwise_and(combined_mask, texture_mask)
        combined_mask = cv2.bitwise_and(combined_mask, roi_mask)
        
        # Morphological operations to clean up and connect regions
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        nose_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
        nose_mask = cv2.morphologyEx(nose_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Fill holes
        contours, _ = cv2.findContours(nose_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get largest contour (the nose leather)
            largest_contour = max(contours, key=cv2.contourArea)
            nose_mask = np.zeros_like(nose_mask)
            cv2.drawContours(nose_mask, [largest_contour], -1, 255, -1)
            
            # Exclude nostrils (they are too dark and circular)
            nostril_mask = self.detect_and_exclude_nostrils(gray, nose_mask)
            nose_mask = cv2.bitwise_and(nose_mask, cv2.bitwise_not(nostril_mask))
        
        return nose_mask
    
    def detect_and_exclude_nostrils(self, gray, nose_mask):
        """Detect nostrils to exclude them from the pattern area"""
        # Nostrils are extremely dark circular regions
        _, very_dark = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)
        nostril_candidates = cv2.bitwise_and(very_dark, nose_mask)
        
        # Find circular regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        nostril_candidates = cv2.morphologyEx(nostril_candidates, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(nostril_candidates, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        nostril_mask = np.zeros_like(gray)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 3000:  # Reasonable nostril size
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # Fairly circular
                        # Expand the nostril region slightly
                        cv2.drawContours(nostril_mask, [cnt], -1, 255, -1)
                        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        nostril_mask = cv2.dilate(nostril_mask, kernel_dilate)
        
        return nostril_mask
    
    def extract_nose_region(self, image, mask):
        """Extract the nose region and normalize it"""
        # Find bounding box with padding
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # Add padding
        pad = 30
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2*pad)
        h = min(image.shape[0] - y, h + 2*pad)
        
        # Extract and normalize
        nose_roi = image[y:y+h, x:x+w].copy()
        mask_roi = mask[y:y+h, x:x+w].copy()
        
        # Resize to standard size for consistent feature extraction
        target_size = (400, 400)
        nose_normalized = cv2.resize(nose_roi, target_size, interpolation=cv2.INTER_CUBIC)
        mask_normalized = cv2.resize(mask_roi, target_size, interpolation=cv2.INTER_NEAREST)
        
        return nose_normalized, mask_normalized, (x, y, w, h)
    
    def enhance_ridge_patterns(self, nose_image, mask):
        """
        Enhance the ridge and crease patterns on the nose leather
        This is critical for fingerprint-like identification
        """
        gray = cv2.cvtColor(nose_image, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Histogram equalization for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(masked)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        
        # Unsharp masking to enhance fine details
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        unsharp = cv2.addWeighted(enhanced, 2.0, gaussian, -1.0, 0)
        unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
        unsharp = cv2.bitwise_and(unsharp, unsharp, mask=mask)
        
        # Bilateral filter to preserve edges while smoothing
        bilateral = cv2.bilateralFilter(unsharp, 9, 75, 75)
        bilateral = cv2.bitwise_and(bilateral, bilateral, mask=mask)
        
        return bilateral
    
    def extract_ridge_orientation_map(self, enhanced_gray, mask):
        """
        Extract orientation field of ridges - like in fingerprint analysis
        """
        # Compute gradients
        sobelx = cv2.Sobel(enhanced_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(enhanced_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute orientation
        orientation = np.arctan2(sobely, sobelx)
        
        # Smooth orientation field
        orientation = gaussian_filter(orientation, sigma=3)
        
        # Apply mask
        orientation = orientation * (mask > 0)
        
        return orientation
    
    def extract_ridge_frequency_map(self, enhanced_gray, mask, block_size=16):
        """
        Extract ridge frequency map - spacing between ridges
        """
        h, w = enhanced_gray.shape
        freq_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = enhanced_gray[i:i+block_size, j:j+block_size]
                mask_block = mask[i:i+block_size, j:j+block_size]
                
                if np.sum(mask_block) < (block_size * block_size * 0.5):
                    continue
                
                # FFT to find dominant frequency
                fft = np.fft.fft2(block)
                fft_shifted = np.fft.fftshift(fft)
                magnitude = np.abs(fft_shifted)
                
                # Find peak frequency (ridge spacing)
                center = block_size // 2
                magnitude[center-2:center+2, center-2:center+2] = 0  # Remove DC component
                
                if magnitude.max() > 0:
                    peak_idx = np.unravel_index(magnitude.argmax(), magnitude.shape)
                    freq = np.sqrt((peak_idx[0] - center)**2 + (peak_idx[1] - center)**2)
                    freq_map[i:i+block_size, j:j+block_size] = freq
        
        return freq_map
    
    def extract_minutiae_points(self, enhanced_gray, mask):
        """
        Extract minutiae points (ridge endings and bifurcations)
        Like in fingerprint matching
        """
        # Binarize
        _, binary = cv2.threshold(enhanced_gray, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_and(binary, binary, mask=mask)
        
        # Skeletonize to get ridge structure
        skeleton = skeletonize(binary > 0).astype(np.uint8) * 255
        
        # Detect minutiae using morphological operations
        # Ridge endings: points with only 1 neighbor
        # Bifurcations: points with 3 neighbors
        
        minutiae = []
        h, w = skeleton.shape
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                if skeleton[i, j] == 255 and mask[i, j] > 0:
                    # Count neighbors
                    neighbors = [
                        skeleton[i-1, j-1], skeleton[i-1, j], skeleton[i-1, j+1],
                        skeleton[i, j-1],                       skeleton[i, j+1],
                        skeleton[i+1, j-1], skeleton[i+1, j], skeleton[i+1, j+1]
                    ]
                    neighbor_count = sum(n > 0 for n in neighbors)
                    
                    # Ridge ending or bifurcation
                    if neighbor_count == 1 or neighbor_count == 3:
                        minutiae.append({
                            'position': (j, i),
                            'type': 'ending' if neighbor_count == 1 else 'bifurcation'
                        })
        
        return minutiae, skeleton
    
    def extract_pattern_features(self, enhanced_gray, mask):
        """
        Extract comprehensive pattern features from the nose leather
        """
        # 1. Multi-scale LBP for texture at different scales
        lbp_features = []
        for radius in [1, 2, 3]:
            n_points = 8 * radius
            lbp = local_binary_pattern(enhanced_gray, n_points, radius, method='uniform')
            lbp_masked = lbp[mask > 0]
            
            if len(lbp_masked) > 0:
                hist, _ = np.histogram(lbp_masked, bins=n_points+2, 
                                     range=(0, n_points+2), density=True)
                lbp_features.extend(hist)
        
        # 2. GLCM (Gray Level Co-occurrence Matrix) for texture relationships
        glcm = graycomatrix(enhanced_gray, distances=[1, 2, 3], 
                           angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=256, symmetric=True, normed=True)
        
        glcm_features = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            glcm_features.extend(graycoprops(glcm, prop).flatten())
        
        # 3. Orientation histogram
        orientation = self.extract_ridge_orientation_map(enhanced_gray, mask)
        orientation_masked = orientation[mask > 0]
        
        if len(orientation_masked) > 0:
            orientation_hist, _ = np.histogram(orientation_masked, bins=36, 
                                             range=(-np.pi, np.pi), density=True)
        else:
            orientation_hist = np.zeros(36)
        
        # 4. Frequency statistics
        freq_map = self.extract_ridge_frequency_map(enhanced_gray, mask)
        freq_masked = freq_map[mask > 0]
        
        if len(freq_masked) > 0:
            freq_features = [
                np.mean(freq_masked),
                np.std(freq_masked),
                np.percentile(freq_masked, 25),
                np.percentile(freq_masked, 50),
                np.percentile(freq_masked, 75)
            ]
        else:
            freq_features = [0] * 5
        
        # 5. Minutiae density and distribution
        minutiae, skeleton = self.extract_minutiae_points(enhanced_gray, mask)
        
        minutiae_features = [
            len(minutiae),  # Total count
            sum(1 for m in minutiae if m['type'] == 'ending'),
            sum(1 for m in minutiae if m['type'] == 'bifurcation')
        ]
        
        # Spatial distribution of minutiae
        if minutiae:
            positions = np.array([m['position'] for m in minutiae])
            minutiae_features.extend([
                np.mean(positions[:, 0]),  # Mean x
                np.std(positions[:, 0]),   # Std x
                np.mean(positions[:, 1]),  # Mean y
                np.std(positions[:, 1])    # Std y
            ])
        else:
            minutiae_features.extend([0, 0, 0, 0])
        
        # Combine all features
        all_features = np.concatenate([
            lbp_features,
            glcm_features,
            orientation_hist,
            freq_features,
            minutiae_features
        ])
        
        return {
            'feature_vector': all_features,
            'minutiae': minutiae,
            'skeleton': skeleton,
            'orientation_map': orientation,
            'frequency_map': freq_map
        }
    
    def create_pattern_fingerprint(self, image_path):
        """
        Main pipeline: Extract pattern fingerprint from nose image
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"\n{'='*60}")
        print(f"EXTRACTING NOSE PATTERN FINGERPRINT")
        print(f"{'='*60}")
        print(f"Image: {image_path}")
        
        # 1. Isolate nose leather only
        print("Step 1/4: Isolating nose leather (rhinarium)...")
        nose_mask = self.isolate_nose_leather(image)
        
        # 2. Extract and normalize
        print("Step 2/4: Extracting and normalizing region...")
        result = self.extract_nose_region(image, nose_mask)
        if result is None:
            raise ValueError("Could not extract nose region")
        
        nose_image, mask_norm, bbox = result
        
        # 3. Enhance ridge patterns
        print("Step 3/4: Enhancing ridge patterns...")
        enhanced = self.enhance_ridge_patterns(nose_image, mask_norm)
        
        # 4. Extract pattern features
        print("Step 4/4: Extracting pattern features...")
        features = self.extract_pattern_features(enhanced, mask_norm)
        
        print(f"\n✓ Extraction complete!")
        print(f"  - Feature vector size: {len(features['feature_vector'])}")
        print(f"  - Minutiae points: {len(features['minutiae'])}")
        print(f"  - Endings: {sum(1 for m in features['minutiae'] if m['type'] == 'ending')}")
        print(f"  - Bifurcations: {sum(1 for m in features['minutiae'] if m['type'] == 'bifurcation')}")
        
        # Normalize feature vector
        feature_vector = normalize(features['feature_vector'].reshape(1, -1))[0]
        
        return {
            'fingerprint': feature_vector.tolist(),
            'minutiae': features['minutiae'],
            'nose_image': nose_image,
            'enhanced_image': enhanced,
            'mask': mask_norm,
            'skeleton': features['skeleton'],
            'orientation_map': features['orientation_map'],
            'bbox': bbox,
            'timestamp': datetime.now().isoformat()
        }
    
    def visualize_pattern(self, image_path, output_path):
        """
        Create detailed visualization of the nose pattern extraction
        """
        image = cv2.imread(str(image_path))
        data = self.create_pattern_fingerprint(image_path)
        
        # Create multi-panel visualization
        nose_img = data['nose_image']
        enhanced = data['enhanced_image']
        mask = data['mask']
        skeleton = data['skeleton']
        minutiae = data['minutiae']
        
        # Panel 1: Original nose region with outline
        panel1 = nose_img.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel1, contours, -1, (0, 255, 0), 3)
        cv2.putText(panel1, "NOSE LEATHER", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Panel 2: Enhanced pattern
        panel2 = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel2, "ENHANCED PATTERN", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Panel 3: Skeleton with minutiae
        panel3 = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        for m in minutiae:
            x, y = m['position']
            color = (0, 0, 255) if m['type'] == 'ending' else (255, 0, 0)
            cv2.circle(panel3, (x, y), 3, color, -1)
        cv2.putText(panel3, "MINUTIAE POINTS", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel3, f"Endings: {sum(1 for m in minutiae if m['type'] == 'ending')}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(panel3, f"Bifurcations: {sum(1 for m in minutiae if m['type'] == 'bifurcation')}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Panel 4: Orientation field
        orientation = data['orientation_map']
        orientation_vis = ((orientation + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        orientation_vis = cv2.applyColorMap(orientation_vis, cv2.COLORMAP_HSV)
        orientation_vis = cv2.bitwise_and(orientation_vis, orientation_vis, 
                                          mask=mask)
        cv2.putText(orientation_vis, "RIDGE ORIENTATION", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combine panels
        top_row = np.hstack([panel1, panel2])
        bottom_row = np.hstack([panel3, orientation_vis])
        combined = np.vstack([top_row, bottom_row])
        
        # Save
        cv2.imwrite(str(output_path), combined)
        print(f"\n✓ Visualization saved to: {output_path}\n")


class NoseFingerprintDatabase:
    """Database for storing and matching nose pattern fingerprints"""
    
    def __init__(self, db_path='nose_db.json'):
        self.db_path = Path(db_path)
        self.db = self.load_db()
    
    def load_db(self):
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {'dogs': [], 'metadata': {'created': datetime.now().isoformat()}}
    
    def save_db(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.db, f, indent=2)
    
    def enroll(self, name, fingerprint_data, image_path):
        dog_id = hashlib.sha256(f"{name}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        entry = {
            'id': dog_id,
            'name': name,
            'fingerprint': fingerprint_data['fingerprint'],
            'minutiae_count': len(fingerprint_data['minutiae']),
            'enrolled_date': datetime.now().isoformat(),
            'image_path': str(image_path)
        }
        
        self.db['dogs'].append(entry)
        self.save_db()
        
        print(f"\n{'='*60}")
        print(f"✓ DOG ENROLLED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Name: {name}")
        print(f"ID: {dog_id}")
        print(f"Minutiae points: {len(fingerprint_data['minutiae'])}")
        print(f"Total dogs in database: {len(self.db['dogs'])}")
        print(f"{'='*60}\n")
        
        return dog_id
    
    def identify(self, fingerprint, threshold=0.88):
        if not self.db['dogs']:
            return None, 0.0, "Database is empty"
        
        fingerprint_np = np.array(fingerprint)
        
        best_match = None
        best_similarity = 0.0
        
        results = []
        
        for dog in self.db['dogs']:
            stored_print = np.array(dog['fingerprint'])
            
            # Multiple similarity metrics for robust matching
            cosine_sim = 1 - cosine(fingerprint_np, stored_print)
            euclidean_dist = euclidean(fingerprint_np, stored_print)
            
            # Normalize euclidean distance to 0-1 range
            euclidean_sim = 1 / (1 + euclidean_dist)
            
            # Combined similarity
            similarity = (cosine_sim * 0.7 + euclidean_sim * 0.3)
            
            results.append((dog, similarity))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = dog
        
        # Sort results
        results.sort(key=lambda x: x[1], reverse=True)
        
        if best_similarity >= threshold:
            return best_match, best_similarity, "Match found", results[:3]
        else:
            return None, best_similarity, f"No match (best: {best_similarity:.2%}, threshold: {threshold:.2%})", results[:3]


class IPWebcamCapture:
    """Capture frames from IP webcam"""
    
    def __init__(self, url, output_dir='webcam_snaps'):
        self.url = url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.capture_running = False
        
    def connect(self):
        """Test connection to IP webcam"""
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            raise ConnectionError(f"Failed to connect to IP webcam at {self.url}")
        cap.release()
        print(f"✓ Connected to IP webcam: {self.url}")
        return True
    
    def capture_snaps(self, num_snaps=5, interval=0.5, display=True):
        """
        Capture multiple snaps from IP webcam
        
        Args:
            num_snaps: Number of frames to capture
            interval: Time interval between captures (seconds)
            display: Whether to display frames during capture
        """
        print(f"\n{'='*60}")
        print(f"CAPTURING {num_snaps} SNAPS FROM IP WEBCAM")
        print(f"{'='*60}")
        print(f"URL: {self.url}")
        print(f"Interval: {interval}s")
        
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            raise ConnectionError(f"Failed to connect to IP webcam")
        
        self.frames = []
        captured_count = 0
        
        try:
            while captured_count < num_snaps:
                ret, frame = cap.read()
                
                if not ret:
                    print("Warning: Failed to read frame, retrying...")
                    continue
                
                # Resize for consistency
                frame = cv2.resize(frame, (800, 600))
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"snap_{captured_count+1:03d}_{timestamp}.jpg"
                filepath = self.output_dir / filename
                
                cv2.imwrite(str(filepath), frame)
                self.frames.append({
                    'path': filepath,
                    'filename': filename,
                    'timestamp': datetime.now().isoformat(),
                    'index': captured_count + 1
                })
                
                captured_count += 1
                print(f"  [{captured_count}/{num_snaps}] Captured: {filename}")
                
                # Display frame if requested
                if display:
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"Snap {captured_count}/{num_snaps}", (20, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('IP Webcam Capture', display_frame)
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        print("Capture interrupted by user")
                        break
                
                # Wait before next capture
                if captured_count < num_snaps:
                    time.sleep(interval)
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
        
        print(f"\n✓ Captured {len(self.frames)} snaps successfully")
        print(f"Output directory: {self.output_dir}")
        return self.frames


def main():
    parser = argparse.ArgumentParser(description='Dog Nose Pattern Fingerprint System')
    parser.add_argument('--input', type=str, help='Input image path')
    parser.add_argument('--output', type=str, help='Output visualization path')
    parser.add_argument('--batch', type=str, help='Batch process directory of images')
    parser.add_argument('--batch-output', type=str, default='nose_pattern_visualizations', help='Output directory for batch processing')
    parser.add_argument('--enroll', action='store_true', help='Enroll a new dog')
    parser.add_argument('--identify', action='store_true', help='Identify a dog')
    parser.add_argument('--visualize', action='store_true', help='Visualize pattern extraction')
    parser.add_argument('--webcam', type=str, help='IP webcam URL (e.g., http://192.168.1.100:8080/video)')
    parser.add_argument('--snaps', type=int, default=5, help='Number of snaps to capture from webcam')
    parser.add_argument('--interval', type=float, default=0.5, help='Interval between snaps (seconds)')
    parser.add_argument('--snap-output', type=str, default='webcam_snaps', help='Output directory for snaps')
    parser.add_argument('--process-snaps', action='store_true', help='Process all captured snaps for fingerprinting')
    parser.add_argument('--name', type=str, help='Dog name (for enrollment)')
    parser.add_argument('--db', type=str, default='nose_db.json', help='Database file')
    parser.add_argument('--threshold', type=float, default=0.88, help='Match threshold')
    
    args = parser.parse_args()
    
    extractor = NosePatternExtractor()
    database = NoseFingerprintDatabase(args.db)
    
    # Handle batch processing
    if args.batch:
        batch_dir = Path(args.batch)
        if not batch_dir.exists():
            print(f"Error: Batch directory not found: {batch_dir}")
            return
        
        output_dir = Path(args.batch_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING NOSE PATTERN FINGERPRINTS")
        print(f"{'='*70}")
        print(f"Input directory: {batch_dir}")
        print(f"Output directory: {output_dir}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in batch_dir.rglob('*') if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images to process\n")
        
        if not image_files:
            print("No image files found!")
            return
        
        results = []
        processed_count = 0
        failed_count = 0
        
        for idx, image_file in enumerate(sorted(image_files), 1):
            try:
                print(f"[{idx}/{len(image_files)}] Processing: {image_file.name}")
                
                # Extract fingerprint
                data = extractor.create_pattern_fingerprint(image_file)
                
                # Create output filename
                base_name = image_file.stem
                output_filename = f"{base_name}_pattern_viz.jpg"
                output_path = output_dir / output_filename
                
                # Generate and save visualization
                nose_img = data['nose_image']
                enhanced = data['enhanced_image']
                mask = data['mask']
                skeleton = data['skeleton']
                minutiae = data['minutiae']
                
                # Panel 1: Original nose region with outline
                panel1 = nose_img.copy()
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel1, contours, -1, (0, 255, 0), 3)
                cv2.putText(panel1, "NOSE LEATHER", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Panel 2: Enhanced pattern
                panel2 = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                cv2.putText(panel2, "ENHANCED PATTERN", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Panel 3: Skeleton with minutiae
                panel3 = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
                for m in minutiae:
                    x, y = m['position']
                    color = (0, 0, 255) if m['type'] == 'ending' else (255, 0, 0)
                    cv2.circle(panel3, (x, y), 3, color, -1)
                cv2.putText(panel3, "MINUTIAE POINTS", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(panel3, f"Endings: {sum(1 for m in minutiae if m['type'] == 'ending')}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(panel3, f"Bifurcations: {sum(1 for m in minutiae if m['type'] == 'bifurcation')}", 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Panel 4: Orientation field
                orientation = data['orientation_map']
                orientation_vis = ((orientation + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
                orientation_vis = cv2.applyColorMap(orientation_vis, cv2.COLORMAP_HSV)
                orientation_vis = cv2.bitwise_and(orientation_vis, orientation_vis, mask=mask)
                cv2.putText(orientation_vis, "RIDGE ORIENTATION", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Combine panels
                top_row = np.hstack([panel1, panel2])
                bottom_row = np.hstack([panel3, orientation_vis])
                combined = np.vstack([top_row, bottom_row])
                
                # Save visualization
                cv2.imwrite(str(output_path), combined)
                
                # Save metadata
                meta_filename = f"{base_name}_pattern_metadata.json"
                meta_path = output_dir / meta_filename
                
                metadata = {
                    'source_image': image_file.name,
                    'timestamp': datetime.now().isoformat(),
                    'visualization': output_filename,
                    'fingerprint': data['fingerprint'],
                    'minutiae_count': len(minutiae),
                    'minutiae_endings': sum(1 for m in minutiae if m['type'] == 'ending'),
                    'minutiae_bifurcations': sum(1 for m in minutiae if m['type'] == 'bifurcation'),
                    'feature_vector_size': len(data['fingerprint'])
                }
                
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                results.append({
                    'source': image_file.name,
                    'output': output_filename,
                    'status': 'success',
                    'minutiae': len(minutiae)
                })
                
                processed_count += 1
                print(f"  ✓ Saved: {output_filename} ({len(minutiae)} minutiae points)")
                
            except Exception as e:
                failed_count += 1
                error_msg = str(e)
                results.append({
                    'source': image_file.name,
                    'status': 'failed',
                    'error': error_msg
                })
                print(f"  ✗ Error: {error_msg}")
        
        # Save batch summary
        summary_path = output_dir / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'input_directory': str(batch_dir),
                'total_images': len(image_files),
                'processed': processed_count,
                'failed': failed_count,
                'results': results
            }, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Processed: {processed_count}/{len(image_files)}")
        print(f"Failed: {failed_count}")
        print(f"Output directory: {output_dir}")
        print(f"Summary: {summary_path}")
        print(f"{'='*70}\n")
    

        webcam = IPWebcamCapture(args.webcam, args.snap_output)
        
        try:
            webcam.connect()
        except ConnectionError as e:
            print(f"Error: {e}")
            return
        
        # Capture snaps
        snaps = webcam.capture_snaps(args.snaps, args.interval)
        
        # Process snaps if requested
        if args.process_snaps:
            print(f"\n{'='*60}")
            print(f"PROCESSING {len(snaps)} CAPTURED SNAPS")
            print(f"{'='*60}\n")
            
            results = []
            
            for snap in snaps:
                print(f"\nProcessing: {snap['filename']}")
                try:
                    # Extract fingerprint
                    data = extractor.create_pattern_fingerprint(snap['path'])
                    
                    # Create visualization
                    viz_output = snap['path'].parent / f"viz_{snap['filename']}"
                    extractor.visualize_pattern(snap['path'], viz_output)
                    
                    result = {
                        'snap_file': snap['filename'],
                        'timestamp': snap['timestamp'],
                        'fingerprint': data['fingerprint'],
                        'minutiae_count': len(data['minutiae']),
                        'visualization': str(viz_output),
                        'endings': sum(1 for m in data['minutiae'] if m['type'] == 'ending'),
                        'bifurcations': sum(1 for m in data['minutiae'] if m['type'] == 'bifurcation')
                    }
                    
                    # Try to identify
                    match, similarity, message, top_matches = database.identify(
                        data['fingerprint'], args.threshold)
                    
                    if match:
                        result['match'] = {
                            'name': match['name'],
                            'id': match['id'],
                            'confidence': similarity
                        }
                    else:
                        result['match'] = None
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"  Error processing snap: {e}")
                    results.append({
                        'snap_file': snap['filename'],
                        'error': str(e)
                    })
            
            # Save results summary
            summary_file = Path(args.snap_output) / 'processing_summary.json'
            with open(summary_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_snaps': len(snaps),
                    'results': results
                }, f, indent=2)
            
            print(f"\n{'='*60}")
            print(f"✓ PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Summary saved to: {summary_file}")
    
    elif args.visualize:
        if not args.input or not args.output:
            print("Error: --input and --output required")
            return
        extractor.visualize_pattern(args.input, args.output)
    
    elif args.enroll:
        if not args.input or not args.name:
            print("Error: --input and --name required")
            return
        data = extractor.create_pattern_fingerprint(args.input)
        database.enroll(args.name, data, args.input)
    
    elif args.identify:
        if not args.input:
            print("Error: --input required")
            return
        data = extractor.create_pattern_fingerprint(args.input)
        match, similarity, message, top_matches = database.identify(
            data['fingerprint'], args.threshold)
        
        print(f"\n{'='*60}")
        print("IDENTIFICATION RESULT")
        print(f"{'='*60}")
        
        if match:
            print(f"✓ MATCH FOUND!")
            print(f"  Name: {match['name']}")
            print(f"  ID: {match['id']}")
            print(f"  Confidence: {similarity:.2%}")
            print(f"  Enrolled: {match['enrolled_date']}")
        else:
            print(f"✗ No confident match")
            print(f"  {message}")
        
        print(f"\nTop 3 candidates:")
        for i, (dog, sim) in enumerate(top_matches, 1):
            print(f"  {i}. {dog['name']} - {sim:.2%}")
        
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()