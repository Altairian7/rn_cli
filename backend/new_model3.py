"""
Dog Nose Pattern Fingerprint System v3
Based on research: "Dog Identification Method Based on Muzzle Pattern Image" 
(Jang et al., 2020, Applied Sciences)

This implementation follows the proven methodology from the paper with enhancements.

Key improvements from research:
- ROI defined as max rectangle containing nostril boundaries
- Light reflection detection and removal
- Blur detection using Haar wavelet
- Aspect-ratio preserving resize
- Iterative CLAHE until histogram stretches
- ORB feature extraction (proven best in research)
- RANSAC + Duplicate Matching Removal
- Robust to rotation, intensity, perspective, noise

Requirements:
    pip install opencv-python numpy scikit-image pywavelets

Usage:
    python nose_fingerprint_v3.py --enroll --input dog_nose.jpg --name "Max"
    python nose_fingerprint_v3.py --identify --input unknown.jpg
    python nose_fingerprint_v3.py --visualize --input dog_nose.jpg --output viz.jpg
"""

import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime
import hashlib
from scipy.spatial.distance import hamming
import pywt
import warnings
warnings.filterwarnings('ignore')


class ResearchBasedNoseExtractor:
    """
    Implementation based on Jang et al. (2020) research paper
    Uses ORB features with RANSAC + DMR postprocessing
    """
    
    def __init__(self):
        print("Initializing Research-Based Nose Pattern Extractor")
        print("Based on: Jang et al. (2020) - Applied Sciences")
        
        # ORB detector (proven best in research)
        self.orb = cv2.ORB_create(nfeatures=2000)  # More features for robustness
        
    def detect_light_reflection(self, image):
        """
        Detect light reflection using histogram analysis
        Paper method: Discard if >200 pixels in 150-255 range
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        
        # Count pixels in bright range (150-255)
        bright_pixels = np.sum(hist[150:256])
        
        if bright_pixels > 200:
            return True, bright_pixels
        return False, bright_pixels
    
    def detect_blur_wavelet(self, image, threshold=50):
        """
        Blur detection using Haar wavelet transform
        Paper method: threshold = 50
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Haar wavelet decomposition
        coeffs = pywt.dwt2(gray, 'haar')
        cA, (cH, cV, cD) = coeffs
        
        # Calculate blur measure from high-frequency components
        blur_measure = np.sqrt(np.mean(cH**2) + np.mean(cV**2) + np.mean(cD**2))
        
        is_blurry = blur_measure < threshold
        return is_blurry, blur_measure
    
    def proposed_resize(self, image, reference=300):
        """
        Resize maintaining aspect ratio
        Paper method: Smaller side to reference value, scale other side proportionally
        """
        h, w = image.shape[:2]
        
        # Calculate scale factor
        if w <= h:
            scale_factor = h / w
            new_w = reference
            new_h = int(reference * scale_factor)
        else:
            scale_factor = w / h
            new_h = reference
            new_w = int(reference * scale_factor)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return resized
    
    def proposed_clahe(self, image):
        """
        Iterative CLAHE until histogram is stretched
        Paper method: Repeat until >1000 pixels in 0-49 AND 206-255
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # CLAHE parameters from paper
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        result = gray.copy()
        iterations = 0
        max_iterations = 10  # Safety limit
        
        while iterations < max_iterations:
            hist, _ = np.histogram(result, bins=256, range=(0, 256))
            
            # Check if histogram is stretched enough
            dark_pixels = np.sum(hist[0:50])
            bright_pixels = np.sum(hist[206:256])
            
            if dark_pixels > 1000 and bright_pixels > 1000:
                break
            
            # Apply CLAHE
            result = clahe.apply(result)
            iterations += 1
        
        print(f"  CLAHE iterations: {iterations}")
        return result
    
    def isolate_nose_roi(self, image):
        """
        Isolate ROI as maximum rectangle containing nostril boundaries
        Paper definition: ROI = max rectangle with both nostril boundaries
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Focus on upper-center region
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[int(h*0.1):int(h*0.7), int(w*0.15):int(w*0.85)] = 255
        
        # Detect dark regions (nose leather)
        _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        dark_mask = cv2.bitwise_and(dark_mask, roi_mask)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour (nose region)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        
        # Detect nostrils (very dark circular regions)
        nostril_mask = self.detect_nostrils(gray, dark_mask)
        
        # Find nostril contours
        nostril_contours, _ = cv2.findContours(nostril_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding box that contains both nostrils
        if len(nostril_contours) >= 2:
            # Combine nostril contours
            all_points = np.vstack([cnt for cnt in nostril_contours[:2]])
            x, y, w, h = cv2.boundingRect(all_points)
            
            # Expand to include surrounding nose leather
            pad = 40
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(image.shape[1] - x, w + 2*pad)
            h = min(image.shape[0] - y, h + 2*pad)
        else:
            # Fallback to nose leather bounding box
            x, y, w, h = cv2.boundingRect(largest)
        
        # Extract ROI
        nose_roi = image[y:y+h, x:x+w].copy()
        
        # Create mask for ROI
        nose_mask = np.zeros_like(gray)
        cv2.drawContours(nose_mask, [largest], -1, 255, -1)
        
        # Exclude nostrils from mask
        nose_mask = cv2.bitwise_and(nose_mask, cv2.bitwise_not(nostril_mask))
        mask_roi = nose_mask[y:y+h, x:x+w].copy()
        
        return nose_roi, mask_roi, (x, y, w, h)
    
    def detect_nostrils(self, gray, nose_mask):
        """Detect nostrils as very dark circular regions"""
        _, extreme_dark = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        nostril_candidates = cv2.bitwise_and(extreme_dark, nose_mask)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        nostril_candidates = cv2.morphologyEx(nostril_candidates, cv2.MORPH_OPEN, 
                                             kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(nostril_candidates, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        nostril_mask = np.zeros_like(gray)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 5000:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:
                        cv2.drawContours(nostril_mask, [cnt], -1, 255, -1)
        
        # Dilate to ensure complete exclusion
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        nostril_mask = cv2.dilate(nostril_mask, kernel_dilate, iterations=2)
        
        return nostril_mask
    
    def extract_orb_features(self, image, mask):
        """
        Extract ORB features (best performer in research)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
        
        return keypoints, descriptors
    
    def match_features_with_ransac(self, desc1, desc2, kp1, kp2):
        """
        Match features using hamming distance + RANSAC
        Paper method: Hamming distance with threshold=64, RANSAC threshold=4
        """
        if desc1 is None or desc2 is None:
            return []
        
        # Brute force matching with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Paper threshold for ORB: 64
        matches = bf.match(desc1, desc2)
        matches = [m for m in matches if m.distance < 64]
        
        if len(matches) < 4:
            return matches
        
        # Apply RANSAC (paper threshold: 4)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
            matches_ransac = [matches[i] for i in range(len(matches)) if mask[i]]
        except:
            matches_ransac = matches
        
        return matches_ransac
    
    def remove_duplicate_matches(self, matches, kp1, kp2):
        """
        Duplicate Matching Removal (DMR) - Paper's key innovation
        Remove matches where multiple source points map to same destination
        """
        if not matches:
            return matches
        
        # Track destination points
        dst_points = {}
        unique_matches = []
        
        for m in matches:
            dst_pt = tuple(map(int, kp2[m.trainIdx].pt))
            
            if dst_pt not in dst_points:
                dst_points[dst_pt] = m
                unique_matches.append(m)
        
        return unique_matches
    
    def create_fingerprint(self, image_path):
        """
        Main pipeline following research methodology
        """
        print(f"\n{'='*70}")
        print("CREATING NOSE PATTERN FINGERPRINT (Research-Based)")
        print(f"{'='*70}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Image: {image_path}")
        print(f"Size: {image.shape[1]}x{image.shape[0]}")
        
        # Step 1: Quality checks
        print("\nStep 1/6: Quality screening...")
        
        has_reflection, bright_px = self.detect_light_reflection(image)
        if has_reflection:
            print(f"  WARNING: Light reflection detected ({bright_px} bright pixels)")
        
        is_blurry, blur_score = self.detect_blur_wavelet(image)
        if is_blurry:
            print(f"  WARNING: Image may be blurry (score: {blur_score:.2f})")
        
        # Step 2: Isolate ROI
        print("\nStep 2/6: Isolating nose ROI...")
        roi_result = self.isolate_nose_roi(image)
        if roi_result is None:
            raise ValueError("Could not extract nose ROI")
        
        nose_roi, mask_roi, bbox = roi_result
        print(f"  ROI size: {nose_roi.shape[1]}x{nose_roi.shape[0]}")
        
        # Step 3: Proposed resize
        print("\nStep 3/6: Applying proposed resize...")
        nose_resized = self.proposed_resize(nose_roi, reference=300)
        mask_resized = self.proposed_resize(mask_roi, reference=300)
        print(f"  Resized to: {nose_resized.shape[1]}x{nose_resized.shape[0]}")
        
        # Step 4: Proposed CLAHE
        print("\nStep 4/6: Applying iterative CLAHE...")
        enhanced = self.proposed_clahe(nose_resized)
        
        # Step 5: Extract ORB features
        print("\nStep 5/6: Extracting ORB features...")
        keypoints, descriptors = self.extract_orb_features(enhanced, mask_resized)
        print(f"  Keypoints detected: {len(keypoints)}")
        
        print(f"\n{'='*70}")
        print("✓ FINGERPRINT CREATED")
        print(f"{'='*70}\n")
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'enhanced_image': enhanced,
            'mask': mask_resized,
            'bbox': bbox,
            'quality': {
                'has_reflection': has_reflection,
                'is_blurry': is_blurry,
                'blur_score': blur_score
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def match_fingerprints(self, fp1, fp2):
        """
        Match two fingerprints using research methodology
        """
        # Extract features from matching
        matches = self.match_features_with_ransac(
            fp1['descriptors'], fp2['descriptors'],
            fp1['keypoints'], fp2['keypoints']
        )
        
        # Apply DMR (paper's innovation)
        matches_filtered = self.remove_duplicate_matches(
            matches, fp1['keypoints'], fp2['keypoints']
        )
        
        return matches_filtered
    
    def visualize_extraction(self, image_path, output_path):
        """Create visualization of the extraction process"""
        image = cv2.imread(str(image_path))
        fp_data = self.create_fingerprint(image_path)
        
        enhanced = fp_data['enhanced_image']
        mask = fp_data['mask']
        keypoints = fp_data['keypoints']
        
        # Create 4-panel visualization
        h, w = enhanced.shape[:2] if len(enhanced.shape) == 2 else enhanced.shape[:2]
        
        # Panel 1: Original ROI with mask outline
        panel1 = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel1, contours, -1, (0, 255, 0), 2)
        cv2.putText(panel1, "1. NOSE ROI", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Panel 2: Enhanced with CLAHE
        panel2 = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel2, "2. CLAHE ENHANCED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Panel 3: ORB keypoints
        panel3 = cv2.drawKeypoints(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), 
                                   keypoints, None, 
                                   color=(0, 255, 255),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.putText(panel3, "3. ORB KEYPOINTS", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(panel3, f"Count: {len(keypoints)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Panel 4: Info panel
        panel4 = np.ones((h, w, 3), dtype=np.uint8) * 255
        cv2.putText(panel4, "RESEARCH-BASED METHOD", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(panel4, "Jang et al. (2020)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        y = 100
        cv2.putText(panel4, "Features:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 0, 0), 2)
        y += 25
        cv2.putText(panel4, f"- ORB Keypoints: {len(keypoints)}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        y += 25
        cv2.putText(panel4, f"- Aspect Ratio Resize", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        y += 25
        cv2.putText(panel4, f"- Iterative CLAHE", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        y += 25
        cv2.putText(panel4, f"- RANSAC + DMR", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        y += 40
        if fp_data['quality']['has_reflection']:
            cv2.putText(panel4, "! Light reflection detected", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            y += 25
        
        if fp_data['quality']['is_blurry']:
            cv2.putText(panel4, "! Image may be blurry", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Combine panels
        top_row = np.hstack([panel1, panel2])
        bottom_row = np.hstack([panel3, panel4])
        combined = np.vstack([top_row, bottom_row])
        
        cv2.imwrite(str(output_path), combined)
        print(f"✓ Visualization saved to: {output_path}\n")


class NoseFingerprintDatabase:
    """Database for storing and matching nose fingerprints"""
    
    def __init__(self, db_path='nose_research_db.json'):
        self.db_path = Path(db_path)
        self.db = self.load_db()
    
    def load_db(self):
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {'dogs': [], 'metadata': {
            'created': datetime.now().isoformat(),
            'method': 'Research-Based (Jang et al. 2020)'
        }}
    
    def save_db(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.db, f, indent=2)
    
    def enroll(self, name, fingerprint_data, image_path):
        # Convert keypoints and descriptors to serializable format
        kp_data = [(kp.pt, kp.size, kp.angle) for kp in fingerprint_data['keypoints']]
        desc_data = fingerprint_data['descriptors'].tolist()
        
        dog_id = hashlib.sha256(
            f"{name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        entry = {
            'id': dog_id,
            'name': name,
            'keypoints': kp_data,
            'descriptors': desc_data,
            'keypoint_count': len(kp_data),
            'enrolled_date': datetime.now().isoformat(),
            'image_path': str(image_path),
            'quality': fingerprint_data['quality']
        }
        
        self.db['dogs'].append(entry)
        self.save_db()
        
        print(f"\n{'='*70}")
        print("✓ DOG ENROLLED")
        print(f"{'='*70}")
        print(f"Name: {name}")
        print(f"ID: {dog_id}")
        print(f"ORB Keypoints: {len(kp_data)}")
        print(f"Database size: {len(self.db['dogs'])} dogs")
        print(f"{'='*70}\n")
        
        return dog_id
    
    def identify(self, fingerprint_data, extractor, threshold=25):
        """
        Identify dog using ORB matching
        Paper's optimal threshold for ORB: 25 (from Table 4)
        """
        if not self.db['dogs']:
            return None, 0, "Database empty", []
        
        results = []
        
        for dog in self.db['dogs']:
            # Reconstruct keypoints and descriptors
            kp_stored = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=pt[1], 
                                      angle=pt[2]) for pt in dog['keypoints']]
            desc_stored = np.array(dog['descriptors'], dtype=np.uint8)
            
            # Create fingerprint dict for stored data
            fp_stored = {
                'keypoints': kp_stored,
                'descriptors': desc_stored
            }
            
            # Match
            matches = extractor.match_fingerprints(fingerprint_data, fp_stored)
            good_matches = len(matches)
            
            results.append((dog, good_matches))
        
        # Sort by match count
        results.sort(key=lambda x: x[1], reverse=True)
        best_match, best_count = results[0]
        
        if best_count >= threshold:
            return best_match, best_count, "Match found", results[:5]
        else:
            return None, best_count, \
                   f"No match (best: {best_count}, threshold: {threshold})", \
                   results[:5]


def main():
    parser = argparse.ArgumentParser(
        description='Research-Based Dog Nose Fingerprint System (Jang et al. 2020)'
    )
    parser.add_argument('--input', type=str, help='Input image')
    parser.add_argument('--output', type=str, help='Output visualization')
    parser.add_argument('--enroll', action='store_true', help='Enroll dog')
    parser.add_argument('--identify', action='store_true', help='Identify dog')
    parser.add_argument('--visualize', action='store_true', help='Visualize extraction')
    parser.add_argument('--name', type=str, help='Dog name (for enrollment)')
    parser.add_argument('--db', type=str, default='nose_research_db.json', 
                       help='Database file')
    parser.add_argument('--threshold', type=int, default=25, 
                       help='Match threshold (default: 25 from research)')
    
    args = parser.parse_args()
    
    extractor = ResearchBasedNoseExtractor()
    database = NoseFingerprintDatabase(args.db)
    
    if args.visualize:
        if not args.input or not args.output:
            print("Error: --input and --output required")
            return
        extractor.visualize_extraction(args.input, args.output)
    
    elif args.enroll:
        if not args.input or not args.name:
            print("Error: --input and --name required")
            return
        fp_data = extractor.create_fingerprint(args.input)
        database.enroll(args.name, fp_data, args.input)
    
    elif args.identify:
        if not args.input:
            print("Error: --input required")
            return
        fp_data = extractor.create_fingerprint(args.input)
        match, count, message, top_matches = database.identify(
            fp_data, extractor, args.threshold)
        
        print(f"\n{'='*70}")
        print("IDENTIFICATION RESULT")
        print(f"{'='*70}")
        
        if match:
            print(f"✓ MATCH FOUND")
            print(f"  Name: {match['name']}")
            print(f"  ID: {match['id']}")
            print(f"  Match count: {count}")
            print(f"  Enrolled: {match['enrolled_date']}")
        else:
            print(f"✗ No match")
            print(f"  {message}")
        
        print(f"\nTop 5 candidates:")
        for i, (dog, cnt) in enumerate(top_matches, 1):
            print(f"  {i}. {dog['name']} - {cnt} matches")
        
        print(f"{'='*70}\n")
    
    else:
        print("Error: Specify --enroll, --identify, or --visualize")


if __name__ == "__main__":
    main()