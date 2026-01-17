"""
Enhanced Dog Nose Pattern Fingerprint System v2

Improved version with:
- Better nose leather isolation (excludes nostrils properly)
- High-quality minutiae detection with filtering
- Advanced ridge enhancement
- Cleaner orientation fields
- Multi-resolution feature extraction

Requirements:
    pip install torch torchvision opencv-python numpy pillow scikit-learn scipy scikit-image

Usage:
    python nose_fingerprint.py --visualize --input dog_nose.jpg --output pattern_viz.jpg
    python nose_fingerprint.py --enroll --input dog_nose.jpg --name "Max" --db nose_db.json
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
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.morphology import skeletonize, remove_small_objects, thin
from skimage.filters import frangi, meijering, sato
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')


class EnhancedNosePatternExtractor:
    """
    Enhanced extractor with improved segmentation and feature extraction
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 
                                     'deeplabv3_resnet101', 
                                     pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
    def isolate_nose_leather(self, image):
        """
        Improved nose leather isolation that properly excludes nostrils
        """
        h, w = image.shape[:2]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # ROI - upper center region
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[int(h*0.1):int(h*0.7), int(w*0.15):int(w*0.85)] = 255
        
        # Multi-stage segmentation
        
        # Stage 1: Find dark regions (nose leather is dark but not extreme)
        # Use adaptive thresholding for better handling of lighting variations
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )
        dark_mask = cv2.bitwise_and(adaptive_thresh, roi_mask)
        
        # Stage 2: Color-based segmentation (nose is low saturation)
        lower_nose = np.array([0, 0, 10])
        upper_nose = np.array([180, 100, 120])
        color_mask = cv2.inRange(hsv, lower_nose, upper_nose)
        
        # Stage 3: Texture analysis - nose leather has specific texture
        # Use Gabor filters to detect textured regions
        texture_score = self.compute_texture_score(gray)
        _, texture_mask = cv2.threshold(texture_score, 30, 255, cv2.THRESH_BINARY)
        
        # Combine masks
        combined = cv2.bitwise_and(dark_mask, color_mask)
        combined = cv2.bitwise_and(combined, texture_mask)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Get main nose region
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(gray)
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        nose_mask = np.zeros_like(gray)
        cv2.drawContours(nose_mask, [largest], -1, 255, -1)
        
        # CRITICAL: Detect and EXCLUDE nostrils
        nostril_mask = self.detect_nostrils_precise(gray, nose_mask)
        
        # Remove nostrils from nose mask
        nose_leather_only = cv2.bitwise_and(nose_mask, cv2.bitwise_not(nostril_mask))
        
        # Clean up edges
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        nose_leather_only = cv2.erode(nose_leather_only, kernel_erode, iterations=1)
        nose_leather_only = cv2.dilate(nose_leather_only, kernel_erode, iterations=1)
        
        # Remove small holes
        contours, _ = cv2.findContours(nose_leather_only, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            nose_leather_only = np.zeros_like(nose_leather_only)
            cv2.drawContours(nose_leather_only, [largest], -1, 255, -1)
        
        return nose_leather_only
    
    def compute_texture_score(self, gray):
        """Compute texture score using multiple Gabor filters"""
        texture_responses = []
        
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((15, 15), 3.0, theta, 8.0, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            texture_responses.append(np.abs(filtered))
        
        # Combine responses
        texture_score = np.mean(texture_responses, axis=0)
        texture_score = (texture_score / texture_score.max() * 255).astype(np.uint8)
        
        return texture_score
    
    def detect_nostrils_precise(self, gray, nose_mask):
        """
        Precise nostril detection to exclude from nose leather
        Nostrils are very dark, roughly circular, and appear as cavities
        """
        # Nostrils are EXTREMELY dark (darker than nose leather)
        _, extreme_dark = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        
        # Only within nose region
        nostril_candidates = cv2.bitwise_and(extreme_dark, nose_mask)
        
        # Morphological operations to separate nostrils
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        nostril_candidates = cv2.morphologyEx(nostril_candidates, cv2.MORPH_OPEN, 
                                             kernel_open, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(nostril_candidates, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        nostril_mask = np.zeros_like(gray)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Reasonable nostril size
            if 50 < area < 5000:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    # Check if roughly circular/elliptical
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.3:  # Somewhat circular
                        # Draw nostril with dilation to ensure complete exclusion
                        cv2.drawContours(nostril_mask, [cnt], -1, 255, -1)
        
        # Dilate nostril mask to ensure complete exclusion
        nostril_mask = cv2.dilate(nostril_mask, kernel_dilate, iterations=2)
        
        return nostril_mask
    
    def extract_nose_region(self, image, mask):
        """Extract and normalize the nose region"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # Add padding
        pad = 40
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2*pad)
        h = min(image.shape[0] - y, h + 2*pad)
        
        nose_roi = image[y:y+h, x:x+w].copy()
        mask_roi = mask[y:y+h, x:x+w].copy()
        
        # Normalize to fixed size
        target_size = (512, 512)  # Larger for better detail
        nose_normalized = cv2.resize(nose_roi, target_size, 
                                     interpolation=cv2.INTER_CUBIC)
        mask_normalized = cv2.resize(mask_roi, target_size, 
                                     interpolation=cv2.INTER_NEAREST)
        
        return nose_normalized, mask_normalized, (x, y, w, h)
    
    def enhance_ridge_patterns(self, nose_image, mask):
        """
        Advanced ridge enhancement using multiple techniques
        """
        gray = cv2.cvtColor(nose_image, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Step 1: Contrast enhancement with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(masked)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        
        # Step 2: Ridge enhancement using Frangi filter (vessel/ridge enhancement)
        # This is specifically designed for ridge-like structures
        enhanced_float = enhanced.astype(float) / 255.0
        frangi_enhanced = frangi(enhanced_float, sigmas=range(1, 5, 1), 
                                black_ridges=False)
        frangi_enhanced = (frangi_enhanced * 255).astype(np.uint8)
        
        # Step 3: Combine original with Frangi
        combined = cv2.addWeighted(enhanced, 0.6, frangi_enhanced, 0.4, 0)
        
        # Step 4: Bilateral filter to preserve edges while smoothing
        bilateral = cv2.bilateralFilter(combined, 7, 50, 50)
        
        # Step 5: Unsharp masking for final sharpening
        gaussian = cv2.GaussianBlur(bilateral, (0, 0), 1.5)
        unsharp = cv2.addWeighted(bilateral, 1.8, gaussian, -0.8, 0)
        unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
        
        # Apply mask
        final = cv2.bitwise_and(unsharp, unsharp, mask=mask)
        
        return final
    
    def extract_ridge_orientation_map(self, enhanced_gray, mask, block_size=16):
        """
        Extract smooth ridge orientation field using block-wise processing
        """
        h, w = enhanced_gray.shape
        orientation = np.zeros((h, w), dtype=np.float32)
        coherence = np.zeros((h, w), dtype=np.float32)
        
        # Compute gradients
        sobelx = cv2.Sobel(enhanced_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(enhanced_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Block-wise orientation estimation
        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block_mask = mask[i:i+block_size, j:j+block_size]
                
                if np.sum(block_mask) < (block_size * block_size * 0.3):
                    continue
                
                gx = sobelx[i:i+block_size, j:j+block_size]
                gy = sobely[i:i+block_size, j:j+block_size]
                
                # Gradient structure tensor
                gxx = np.sum(gx * gx * (block_mask > 0))
                gyy = np.sum(gy * gy * (block_mask > 0))
                gxy = np.sum(gx * gy * (block_mask > 0))
                
                # Orientation (perpendicular to gradient)
                theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
                
                # Coherence (how aligned the orientations are)
                coh = np.sqrt((gxx - gyy)**2 + 4*gxy**2) / (gxx + gyy + 1e-5)
                
                orientation[i:i+block_size, j:j+block_size] = theta
                coherence[i:i+block_size, j:j+block_size] = coh
        
        # Smooth orientation field with Gaussian
        orientation_smooth = gaussian_filter(orientation, sigma=5)
        
        # Apply mask
        orientation_smooth = orientation_smooth * (mask > 0)
        coherence = coherence * (mask > 0)
        
        return orientation_smooth, coherence
    
    def extract_minutiae_high_quality(self, enhanced_gray, mask, orientation_map):
        """
        High-quality minutiae extraction with filtering
        """
        # Binarization using local adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, -2
        )
        binary = cv2.bitwise_and(binary, binary, mask=mask)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Thinning to get skeleton
        skeleton = thin((binary > 0).astype(np.uint8))
        skeleton = (skeleton * 255).astype(np.uint8)
        
        # Extract minutiae points
        raw_minutiae = []
        h, w = skeleton.shape
        
        # Use 3x3 crossing number method
        for i in range(2, h-2):
            for j in range(2, w-2):
                if skeleton[i, j] == 255 and mask[i, j] > 0:
                    # Get 8-neighborhood
                    neighbors = [
                        skeleton[i-1, j-1], skeleton[i-1, j], skeleton[i-1, j+1],
                        skeleton[i, j+1], skeleton[i+1, j+1], skeleton[i+1, j],
                        skeleton[i+1, j-1], skeleton[i, j-1]
                    ]
                    
                    # Crossing number (CN)
                    cn = 0
                    for k in range(8):
                        if neighbors[k] == 0 and neighbors[(k+1)%8] == 255:
                            cn += 1
                    
                    # CN = 1: ridge ending, CN = 3: bifurcation
                    if cn == 1:
                        raw_minutiae.append({
                            'position': (j, i),
                            'type': 'ending',
                            'orientation': orientation_map[i, j]
                        })
                    elif cn == 3:
                        raw_minutiae.append({
                            'position': (j, i),
                            'type': 'bifurcation',
                            'orientation': orientation_map[i, j]
                        })
        
        # Filter minutiae
        filtered_minutiae = self.filter_minutiae(raw_minutiae, mask, skeleton)
        
        return filtered_minutiae, skeleton
    
    def filter_minutiae(self, minutiae, mask, skeleton, 
                       min_distance=15, border_distance=20):
        """
        Filter false minutiae using quality metrics
        """
        if not minutiae:
            return []
        
        h, w = mask.shape
        filtered = []
        
        for m in minutiae:
            x, y = m['position']
            
            # Skip if too close to border
            if (x < border_distance or x > w - border_distance or
                y < border_distance or y > h - border_distance):
                continue
            
            # Skip if not well within the mask
            local_mask = mask[y-5:y+5, x-5:x+5]
            if np.sum(local_mask) < 50:
                continue
            
            # Check if isolated (no other minutiae too close)
            is_isolated = True
            for other in filtered:
                ox, oy = other['position']
                dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                if dist < min_distance:
                    is_isolated = False
                    break
            
            if is_isolated:
                filtered.append(m)
        
        # Limit to top minutiae (by distance from center, prefer central ones)
        if len(filtered) > 100:
            center_x, center_y = w // 2, h // 2
            filtered.sort(key=lambda m: np.sqrt(
                (m['position'][0] - center_x)**2 + 
                (m['position'][1] - center_y)**2
            ))
            filtered = filtered[:100]
        
        return filtered
    
    def extract_pattern_features(self, enhanced_gray, mask, orientation_map, minutiae):
        """
        Extract comprehensive pattern features
        """
        # 1. Multi-scale LBP
        lbp_features = []
        for radius in [1, 2, 3, 4]:
            n_points = 8 * radius
            lbp = local_binary_pattern(enhanced_gray, n_points, radius, 
                                      method='uniform')
            lbp_masked = lbp[mask > 0]
            
            if len(lbp_masked) > 0:
                hist, _ = np.histogram(lbp_masked, bins=n_points+2, 
                                     range=(0, n_points+2), density=True)
                lbp_features.extend(hist)
        
        # 2. GLCM texture
        glcm = graycomatrix(enhanced_gray, distances=[1, 2, 4], 
                           angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=256, symmetric=True, normed=True)
        
        glcm_features = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 
                     'energy', 'correlation', 'ASM']:
            glcm_features.extend(graycoprops(glcm, prop).flatten())
        
        # 3. Orientation histogram (high resolution)
        orientation_masked = orientation_map[mask > 0]
        if len(orientation_masked) > 0:
            orientation_hist, _ = np.histogram(orientation_masked, bins=72, 
                                             range=(-np.pi, np.pi), density=True)
        else:
            orientation_hist = np.zeros(72)
        
        # 4. Minutiae spatial distribution
        minutiae_features = self.extract_minutiae_features(minutiae, mask.shape)
        
        # 5. Ridge frequency analysis
        freq_features = self.extract_frequency_features(enhanced_gray, mask, 
                                                       orientation_map)
        
        # Combine all features
        all_features = np.concatenate([
            lbp_features,
            glcm_features,
            orientation_hist,
            minutiae_features,
            freq_features
        ])
        
        return all_features
    
    def extract_minutiae_features(self, minutiae, shape):
        """Extract features from minutiae distribution"""
        if not minutiae:
            return np.zeros(20)
        
        positions = np.array([m['position'] for m in minutiae])
        orientations = np.array([m['orientation'] for m in minutiae])
        
        features = [
            len(minutiae),  # Total count
            sum(1 for m in minutiae if m['type'] == 'ending'),
            sum(1 for m in minutiae if m['type'] == 'bifurcation'),
            np.mean(positions[:, 0]),  # Mean x
            np.std(positions[:, 0]),   # Std x
            np.mean(positions[:, 1]),  # Mean y
            np.std(positions[:, 1]),   # Std y
            np.mean(orientations),
            np.std(orientations),
        ]
        
        # Pairwise distances between minutiae
        if len(positions) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(positions)
            features.extend([
                np.mean(distances),
                np.std(distances),
                np.min(distances),
                np.max(distances)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Orientation distribution
        orientation_hist, _ = np.histogram(orientations, bins=8, 
                                          range=(-np.pi, np.pi), density=True)
        features.extend(orientation_hist)
        
        return np.array(features)
    
    def extract_frequency_features(self, enhanced_gray, mask, 
                                   orientation_map, block_size=32):
        """Extract ridge frequency features"""
        h, w = enhanced_gray.shape
        frequencies = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = enhanced_gray[i:i+block_size, j:j+block_size]
                block_mask = mask[i:i+block_size, j:j+block_size]
                
                if np.sum(block_mask) < (block_size * block_size * 0.5):
                    continue
                
                # Project block along orientation
                orient = orientation_map[i+block_size//2, j+block_size//2]
                
                # Simple frequency estimation using FFT
                fft = np.fft.fft2(block)
                magnitude = np.abs(np.fft.fftshift(fft))
                
                center = block_size // 2
                magnitude[center-2:center+2, center-2:center+2] = 0
                
                if magnitude.max() > 0:
                    peak_idx = np.unravel_index(magnitude.argmax(), magnitude.shape)
                    freq = np.sqrt((peak_idx[0] - center)**2 + 
                                  (peak_idx[1] - center)**2)
                    frequencies.append(freq)
        
        if frequencies:
            return np.array([
                np.mean(frequencies),
                np.std(frequencies),
                np.percentile(frequencies, 25),
                np.percentile(frequencies, 50),
                np.percentile(frequencies, 75)
            ])
        else:
            return np.zeros(5)
    
    def create_pattern_fingerprint(self, image_path):
        """Main pipeline"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"\n{'='*70}")
        print(f"EXTRACTING NOSE PATTERN FINGERPRINT")
        print(f"{'='*70}")
        print(f"Image: {image_path}")
        
        # 1. Isolate nose leather
        print("Step 1/5: Isolating nose leather (excluding nostrils)...")
        nose_mask = self.isolate_nose_leather(image)
        
        # 2. Extract region
        print("Step 2/5: Extracting and normalizing region...")
        result = self.extract_nose_region(image, nose_mask)
        if result is None:
            raise ValueError("Could not extract nose region")
        
        nose_image, mask_norm, bbox = result
        
        # 3. Enhance ridges
        print("Step 3/5: Enhancing ridge patterns...")
        enhanced = self.enhance_ridge_patterns(nose_image, mask_norm)
        
        # 4. Extract orientation
        print("Step 4/5: Computing ridge orientation field...")
        orientation_map, coherence = self.extract_ridge_orientation_map(
            enhanced, mask_norm)
        
        # 5. Extract minutiae
        print("Step 5/5: Extracting and filtering minutiae...")
        minutiae, skeleton = self.extract_minutiae_high_quality(
            enhanced, mask_norm, orientation_map)
        
        # 6. Extract features
        features = self.extract_pattern_features(enhanced, mask_norm, 
                                                orientation_map, minutiae)
        
        print(f"\n✓ Extraction complete!")
        print(f"  - Feature vector size: {len(features)}")
        print(f"  - Minutiae points: {len(minutiae)}")
        print(f"  - Endings: {sum(1 for m in minutiae if m['type'] == 'ending')}")
        print(f"  - Bifurcations: {sum(1 for m in minutiae if m['type'] == 'bifurcation')}")
        print(f"{'='*70}\n")
        
        # Normalize
        feature_vector = normalize(features.reshape(1, -1))[0]
        
        return {
            'fingerprint': feature_vector.tolist(),
            'minutiae': minutiae,
            'nose_image': nose_image,
            'enhanced_image': enhanced,
            'mask': mask_norm,
            'skeleton': skeleton,
            'orientation_map': orientation_map,
            'coherence': coherence,
            'bbox': bbox,
            'timestamp': datetime.now().isoformat()
        }
    
    def visualize_pattern(self, image_path, output_path):
        """Create detailed visualization"""
        image = cv2.imread(str(image_path))
        data = self.create_pattern_fingerprint(image_path)
        
        nose_img = data['nose_image']
        enhanced = data['enhanced_image']
        mask = data['mask']
        skeleton = data['skeleton']
        minutiae = data['minutiae']
        orientation = data['orientation_map']
        
        # Panel 1: Nose leather with precise outline
        panel1 = nose_img.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel1, contours, -1, (0, 255, 0), 2)
        cv2.putText(panel1, "NOSE LEATHER", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(panel1, "(Nostrils Excluded)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Panel 2: Enhanced pattern
        panel2 = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel2, "ENHANCED PATTERN", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 3: Skeleton with filtered minutiae
        panel3 = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        
        # Draw minutiae
        for m in minutiae:
            x, y = m['position']
            if m['type'] == 'ending':
                cv2.circle(panel3, (x, y), 4, (0, 0, 255), -1)  # Red
                cv2.circle(panel3, (x, y), 6, (0, 0, 255), 1)
            else:  # bifurcation
                cv2.circle(panel3, (x, y), 4, (255, 0, 0), -1)  # Blue
                cv2.circle(panel3, (x, y), 6, (255, 0, 0), 1)
        
        cv2.putText(panel3, "MINUTIAE POINTS", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        endings = sum(1 for m in minutiae if m['type'] == 'ending')
        bifurcations = sum(1 for m in minutiae if m['type'] == 'bifurcation')
        
        cv2.putText(panel3, f"Endings: {endings}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(panel3, f"Bifurcations: {bifurcations}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Panel 4: Smooth orientation field
        # Create color-coded orientation map
        orientation_vis = np.zeros((*orientation.shape, 3), dtype=np.uint8)
        
        # Convert orientation to hue (0-180 for OpenCV HSV)
        hue = ((orientation + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        saturation = np.ones_like(hue) * 255
        value = (mask > 0).astype(np.uint8) * 255
        
        hsv = cv2.merge([hue, saturation, value])
        orientation_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Draw orientation field as lines
        h, w = orientation.shape
        step = 16
        for i in range(step, h-step, step):
            for j in range(step, w-step, step):
                if mask[i, j] > 0:
                    angle = orientation[i, j]
                    length = 10
                    dx = int(length * np.cos(angle))
                    dy = int(length * np.sin(angle))
                    cv2.line(orientation_vis, 
                            (j, i), (j+dx, i+dy), 
                            (255, 255, 255), 1)
        
        cv2.putText(orientation_vis, "RIDGE ORIENTATION", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Combine panels
        top_row = np.hstack([panel1, panel2])
        bottom_row = np.hstack([panel3, orientation_vis])
        combined = np.vstack([top_row, bottom_row])
        
        # Save
        cv2.imwrite(str(output_path), combined)
        print(f"✓ Visualization saved to: {output_path}\n")


class NoseFingerprintDatabase:
    """Enhanced database with better matching"""
    
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
        dog_id = hashlib.sha256(
            f"{name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
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
        
        print(f"\n{'='*70}")
        print(f"✓ DOG ENROLLED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Name: {name}")
        print(f"ID: {dog_id}")
        print(f"Minutiae points: {len(fingerprint_data['minutiae'])}")
        print(f"Total dogs in database: {len(self.db['dogs'])}")
        print(f"{'='*70}\n")
        
        return dog_id
    
    def identify(self, fingerprint, threshold=0.90):
        """Identify with enhanced matching"""
        if not self.db['dogs']:
            return None, 0.0, "Database is empty", []
        
        fingerprint_np = np.array(fingerprint)
        results = []
        
        for dog in self.db['dogs']:
            stored_print = np.array(dog['fingerprint'])
            
            # Multiple similarity metrics
            cosine_sim = 1 - cosine(fingerprint_np, stored_print)
            euclidean_dist = euclidean(fingerprint_np, stored_print)
            euclidean_sim = 1 / (1 + euclidean_dist)
            
            # Weighted combination
            similarity = (cosine_sim * 0.75 + euclidean_sim * 0.25)
            
            results.append((dog, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        best_match, best_similarity = results[0]
        
        if best_similarity >= threshold:
            return best_match, best_similarity, "Match found", results[:5]
        else:
            return None, best_similarity, \
                   f"No match (best: {best_similarity:.2%}, threshold: {threshold:.2%})", \
                   results[:5]


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Dog Nose Pattern Fingerprint System v2'
    )
    parser.add_argument('--input', type=str, help='Input image path')
    parser.add_argument('--output', type=str, help='Output visualization path')
    parser.add_argument('--enroll', action='store_true', help='Enroll a new dog')
    parser.add_argument('--identify', action='store_true', help='Identify a dog')
    parser.add_argument('--visualize', action='store_true', 
                       help='Visualize pattern extraction')
    parser.add_argument('--name', type=str, help='Dog name (for enrollment)')
    parser.add_argument('--db', type=str, default='nose_db.json', 
                       help='Database file')
    parser.add_argument('--threshold', type=float, default=0.90, 
                       help='Match threshold (0-1)')
    
    args = parser.parse_args()
    
    extractor = EnhancedNosePatternExtractor()
    database = NoseFingerprintDatabase(args.db)
    
    if args.visualize:
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
        
        print(f"\n{'='*70}")
        print("IDENTIFICATION RESULT")
        print(f"{'='*70}")
        
        if match:
            print(f"✓ MATCH FOUND!")
            print(f"  Name: {match['name']}")
            print(f"  ID: {match['id']}")
            print(f"  Confidence: {similarity:.2%}")
            print(f"  Enrolled: {match['enrolled_date']}")
        else:
            print(f"✗ No confident match")
            print(f"  {message}")
        
        print(f"\nTop 5 candidates:")
        for i, (dog, sim) in enumerate(top_matches, 1):
            print(f"  {i}. {dog['name']} - {sim:.2%}")
        
        print(f"{'='*70}\n")
    
    else:
        print("Error: Specify --enroll, --identify, or --visualize")


if __name__ == "__main__":
    main()