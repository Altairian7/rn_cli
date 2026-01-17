"""
Dog Nose Identification System - Like fingerprints for dogs!

This system detects, segments, and extracts unique features from dog noses
for identification purposes. Dog nose prints are as unique as human fingerprints.

Requirements:
    pip install torch torchvision opencv-python numpy pillow scikit-learn scipy

Usage:
    # Enroll a dog
    python dog_nose_id.py --enroll --input dog1_nose.jpg --name "Max" --db noseprints.json
    
    # Identify a dog
    python dog_nose_id.py --identify --input unknown_nose.jpg --db noseprints.json
    
    # Visualize noseprint
    python dog_nose_id.py --visualize --input dog_nose.jpg --output noseprint_viz.jpg
"""

import cv2
import numpy as np
import torch
import argparse
import json
from pathlib import Path
from datetime import datetime
import hashlib
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')


class DogNosePrintExtractor:
    """
    Extracts unique noseprint features from dog nose images.
    
    Features extracted:
    1. Ridge pattern descriptors (texture)
    2. Nostril geometry (shape, position, size)
    3. Pattern orientation map
    4. Local binary patterns
    5. SIFT/ORB keypoints from texture
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load pretrained model for initial segmentation
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 
                                     'deeplabv3_resnet101', 
                                     pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize feature detectors
        self.orb = cv2.ORB_create(nfeatures=500)
        self.sift = cv2.SIFT_create(nfeatures=500)
        
    def preprocess_image(self, image):
        """Preprocess image for segmentation model"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (520, 520))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_normalized = (img_tensor - mean) / std
        
        return img_normalized.unsqueeze(0).to(self.device)
    
    def segment_nose(self, image):
        """Segment and extract the nose region"""
        h, w = image.shape[:2]
        
        # Get semantic segmentation
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        output_predictions = output.argmax(0).cpu().numpy()
        mask = cv2.resize((output_predictions == 12).astype(np.uint8), 
                         (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Focus on nose region (upper center)
        roi_mask = np.zeros_like(mask)
        roi_mask[int(h*0.15):int(h*0.65), int(w*0.25):int(w*0.75)] = 1
        
        candidate_mask = mask * roi_mask
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        masked_gray = gray.copy()
        masked_gray[candidate_mask == 0] = 255
        
        # Adaptive threshold for dark nose region
        thresh = cv2.adaptiveThreshold(masked_gray, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 5)
        
        nose_mask = cv2.bitwise_and(thresh, thresh, mask=candidate_mask)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        nose_mask = cv2.morphologyEx(nose_mask, cv2.MORPH_CLOSE, kernel)
        nose_mask = cv2.morphologyEx(nose_mask, cv2.MORPH_OPEN, kernel)
        
        # Get largest contour
        contours, _ = cv2.findContours(nose_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            nose_mask = np.zeros_like(nose_mask)
            cv2.drawContours(nose_mask, [largest_contour], -1, 255, -1)
        
        return nose_mask
    
    def extract_nose_region(self, image, mask):
        """Extract and normalize the nose region"""
        # Find bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # Add padding
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2*pad)
        h = min(image.shape[0] - y, h + 2*pad)
        
        # Extract region
        nose_roi = image[y:y+h, x:x+w].copy()
        mask_roi = mask[y:y+h, x:x+w].copy()
        
        # Normalize size for consistent feature extraction
        target_size = (256, 256)
        nose_normalized = cv2.resize(nose_roi, target_size)
        mask_normalized = cv2.resize(mask_roi, target_size)
        
        return nose_normalized, mask_normalized, (x, y, w, h)
    
    def detect_nostrils(self, nose_image, nose_mask):
        """Detect and characterize nostrils"""
        gray = cv2.cvtColor(nose_image, cv2.COLOR_BGR2GRAY)
        
        # Nostrils are darkest regions
        _, nostril_mask = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY_INV)
        nostril_mask = cv2.bitwise_and(nostril_mask, nostril_mask, mask=nose_mask)
        
        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        nostril_mask = cv2.morphologyEx(nostril_mask, cv2.MORPH_OPEN, kernel)
        nostril_mask = cv2.morphologyEx(nostril_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find nostril contours
        contours, _ = cv2.findContours(nostril_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and get top 2
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 8000:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.25:
                        valid_contours.append(cnt)
        
        nostril_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:2]
        
        # Extract nostril features
        nostril_features = []
        for cnt in nostril_contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            # Moments for position
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = 0, 0
            
            # Ellipse fitting for orientation
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                angle = ellipse[2]
            else:
                angle = 0
            
            nostril_features.append({
                'area': area,
                'perimeter': perimeter,
                'center': (cx, cy),
                'angle': angle,
                'circularity': 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            })
        
        return nostril_contours, nostril_features
    
    def extract_texture_features(self, nose_image, nose_mask):
        """Extract texture features from nose surface"""
        gray = cv2.cvtColor(nose_image, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        masked_gray = cv2.bitwise_and(gray, gray, mask=nose_mask)
        
        # 1. Local Binary Pattern (LBP) for texture
        lbp = self.compute_lbp(masked_gray, nose_mask)
        
        # 2. Gabor filters for ridge patterns
        gabor_features = self.compute_gabor_features(masked_gray, nose_mask)
        
        # 3. Edge orientation histogram
        edge_hist = self.compute_edge_orientation(masked_gray, nose_mask)
        
        # 4. SIFT keypoints for unique texture patterns
        keypoints, descriptors = self.sift.detectAndCompute(masked_gray, nose_mask)
        
        # Aggregate SIFT descriptors
        if descriptors is not None and len(descriptors) > 0:
            sift_features = np.mean(descriptors, axis=0)
        else:
            sift_features = np.zeros(128)
        
        return {
            'lbp': lbp,
            'gabor': gabor_features,
            'edge_orientation': edge_hist,
            'sift': sift_features,
            'keypoints': keypoints
        }
    
    def compute_lbp(self, gray, mask):
        """Compute Local Binary Pattern histogram"""
        h, w = gray.shape
        lbp = np.zeros_like(gray)
        
        # Simple 8-neighbor LBP
        for i in range(1, h-1):
            for j in range(1, w-1):
                if mask[i, j] == 0:
                    continue
                    
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] > center) << 7
                code |= (gray[i-1, j] > center) << 6
                code |= (gray[i-1, j+1] > center) << 5
                code |= (gray[i, j+1] > center) << 4
                code |= (gray[i+1, j+1] > center) << 3
                code |= (gray[i+1, j] > center) << 2
                code |= (gray[i+1, j-1] > center) << 1
                code |= (gray[i, j-1] > center) << 0
                lbp[i, j] = code
        
        # Histogram of LBP codes
        hist, _ = np.histogram(lbp[mask > 0], bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist = hist / (hist.sum() + 1e-7)  # Normalize
        
        return hist
    
    def compute_gabor_features(self, gray, mask):
        """Compute Gabor filter responses for ridge detection"""
        features = []
        
        # Multiple orientations to capture ridge patterns
        for theta in np.arange(0, np.pi, np.pi / 8):
            kernel = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            
            # Get statistics from masked region
            masked_response = filtered[mask > 0]
            if len(masked_response) > 0:
                features.extend([
                    np.mean(masked_response),
                    np.std(masked_response),
                    np.max(masked_response),
                    np.min(masked_response)
                ])
            else:
                features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def compute_edge_orientation(self, gray, mask):
        """Compute histogram of edge orientations"""
        # Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        orientation = np.arctan2(sobely, sobelx)
        
        # Only consider edges within mask
        magnitude = magnitude * (mask > 0)
        
        # Histogram of orientations (weighted by magnitude)
        hist, _ = np.histogram(orientation[mask > 0], bins=36, range=(-np.pi, np.pi),
                              weights=magnitude[mask > 0])
        hist = hist.astype(float)
        hist = hist / (hist.sum() + 1e-7)
        
        return hist
    
    def create_noseprint_vector(self, texture_features, nostril_features):
        """Create a single feature vector representing the noseprint"""
        vector = []
        
        # Texture features
        vector.extend(texture_features['lbp'])
        vector.extend(texture_features['gabor'])
        vector.extend(texture_features['edge_orientation'])
        vector.extend(texture_features['sift'])
        
        # Nostril geometry features
        if len(nostril_features) >= 2:
            # Distance between nostrils
            dist = np.sqrt((nostril_features[0]['center'][0] - nostril_features[1]['center'][0])**2 +
                          (nostril_features[0]['center'][1] - nostril_features[1]['center'][1])**2)
            vector.append(dist)
            
            # Angle between nostrils
            dx = nostril_features[1]['center'][0] - nostril_features[0]['center'][0]
            dy = nostril_features[1]['center'][1] - nostril_features[0]['center'][1]
            angle = np.arctan2(dy, dx)
            vector.append(angle)
            
            # Individual nostril features
            for nf in nostril_features:
                vector.extend([
                    nf['area'],
                    nf['perimeter'],
                    nf['circularity'],
                    nf['angle']
                ])
        else:
            # Pad with zeros if nostrils not detected
            vector.extend([0] * 10)
        
        # Normalize vector
        vector = np.array(vector, dtype=np.float32)
        vector = normalize(vector.reshape(1, -1))[0]
        
        return vector
    
    def extract_noseprint(self, image_path):
        """Main pipeline: extract complete noseprint from image"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing: {image_path}")
        
        # 1. Segment nose
        nose_mask = self.segment_nose(image)
        
        # 2. Extract and normalize nose region
        result = self.extract_nose_region(image, nose_mask)
        if result is None:
            raise ValueError("Could not extract nose region")
        
        nose_image, mask_norm, bbox = result
        
        # 3. Detect nostrils
        nostril_contours, nostril_features = self.detect_nostrils(nose_image, mask_norm)
        print(f"  Detected {len(nostril_features)} nostrils")
        
        # 4. Extract texture features
        texture_features = self.extract_texture_features(nose_image, mask_norm)
        print(f"  Extracted texture features: {len(texture_features['lbp'])} LBP bins")
        
        # 5. Create noseprint vector
        noseprint = self.create_noseprint_vector(texture_features, nostril_features)
        print(f"  Noseprint vector size: {len(noseprint)}")
        
        return {
            'noseprint': noseprint.tolist(),
            'nostril_features': nostril_features,
            'bbox': bbox,
            'image_shape': image.shape,
            'timestamp': datetime.now().isoformat()
        }
    
    def visualize_noseprint(self, image_path, output_path):
        """Create visualization of extracted noseprint features"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Segment and extract
        nose_mask = self.segment_nose(image)
        result = self.extract_nose_region(image, nose_mask)
        if result is None:
            raise ValueError("Could not extract nose region")
        
        nose_image, mask_norm, bbox = result
        nostril_contours, nostril_features = self.detect_nostrils(nose_image, mask_norm)
        texture_features = self.extract_texture_features(nose_image, mask_norm)
        
        # Create visualization
        h, w = nose_image.shape[:2]
        vis = cv2.cvtColor(nose_image.copy(), cv2.COLOR_BGR2RGB)
        
        # Draw nose outline
        nose_contours, _ = cv2.findContours(mask_norm, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, nose_contours, -1, (0, 255, 0), 3)
        
        # Draw nostrils
        cv2.drawContours(vis, nostril_contours, -1, (255, 0, 0), 2)
        for i, nf in enumerate(nostril_features):
            cx, cy = int(nf['center'][0]), int(nf['center'][1])
            cv2.circle(vis, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(vis, f"N{i+1}", (cx+10, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw SIFT keypoints
        if texture_features['keypoints']:
            for kp in texture_features['keypoints'][:50]:  # Top 50
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(vis, (x, y), 2, (0, 255, 255), -1)
        
        # Create info panel
        panel = np.ones((h, 300, 3), dtype=np.uint8) * 255
        
        y_offset = 30
        cv2.putText(panel, "NOSEPRINT ANALYSIS", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_offset += 40
        
        cv2.putText(panel, f"Nostrils: {len(nostril_features)}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 25
        
        cv2.putText(panel, f"Keypoints: {len(texture_features['keypoints'])}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 35
        
        # Plot LBP histogram
        lbp_hist = texture_features['lbp']
        max_val = lbp_hist.max()
        for i in range(0, min(len(lbp_hist), 256), 4):
            bar_height = int((lbp_hist[i] / max_val) * 80) if max_val > 0 else 0
            cv2.line(panel, (10 + i//4, y_offset + 80), 
                    (10 + i//4, y_offset + 80 - bar_height), (100, 100, 100), 1)
        
        cv2.putText(panel, "LBP Pattern", (10, y_offset + 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Combine
        combined = np.hstack([vis, panel])
        
        # Save
        cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization to: {output_path}")


class DogNoseDatabase:
    """Database for storing and matching dog noseprints"""
    
    def __init__(self, db_path='noseprints.json'):
        self.db_path = Path(db_path)
        self.db = self.load_db()
    
    def load_db(self):
        """Load database from file"""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {'dogs': [], 'metadata': {'created': datetime.now().isoformat()}}
    
    def save_db(self):
        """Save database to file"""
        with open(self.db_path, 'w') as f:
            json.dump(self.db, f, indent=2)
    
    def enroll(self, name, noseprint_data, image_path):
        """Enroll a new dog in the database"""
        dog_id = hashlib.sha256(f"{name}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        entry = {
            'id': dog_id,
            'name': name,
            'noseprint': noseprint_data['noseprint'],
            'enrolled_date': datetime.now().isoformat(),
            'image_path': str(image_path),
            'metadata': {
                'nostril_count': len(noseprint_data['nostril_features']),
                'image_shape': noseprint_data['image_shape']
            }
        }
        
        self.db['dogs'].append(entry)
        self.save_db()
        
        print(f"\n✓ Enrolled: {name}")
        print(f"  ID: {dog_id}")
        print(f"  Total dogs in database: {len(self.db['dogs'])}")
        
        return dog_id
    
    def identify(self, noseprint, threshold=0.85):
        """Identify a dog from their noseprint"""
        if not self.db['dogs']:
            return None, 0.0, "Database is empty"
        
        noseprint_np = np.array(noseprint)
        
        best_match = None
        best_similarity = 0.0
        
        for dog in self.db['dogs']:
            stored_print = np.array(dog['noseprint'])
            
            # Cosine similarity
            similarity = 1 - cosine(noseprint_np, stored_print)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = dog
        
        if best_similarity >= threshold:
            return best_match, best_similarity, "Match found"
        else:
            return None, best_similarity, f"No match (best: {best_similarity:.2%}, threshold: {threshold:.2%})"
    
    def list_dogs(self):
        """List all enrolled dogs"""
        return self.db['dogs']


def main():
    parser = argparse.ArgumentParser(description='Dog Nose Identification System')
    parser.add_argument('--enroll', action='store_true',
                       help='Enroll a new dog')
    parser.add_argument('--identify', action='store_true',
                       help='Identify a dog')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize noseprint extraction')
    parser.add_argument('--list', action='store_true',
                       help='List all enrolled dogs')
    parser.add_argument('--input', type=str,
                       help='Input image path')
    parser.add_argument('--output', type=str,
                       help='Output path for visualization')
    parser.add_argument('--name', type=str,
                       help='Dog name (for enrollment)')
    parser.add_argument('--db', type=str, default='noseprints.json',
                       help='Database file path')
    parser.add_argument('--threshold', type=float, default=0.85,
                       help='Matching threshold (0-1)')
    
    args = parser.parse_args()
    
    # Initialize
    extractor = DogNosePrintExtractor()
    database = DogNoseDatabase(args.db)
    
    # Commands
    if args.enroll:
        if not args.input or not args.name:
            print("Error: --input and --name required for enrollment")
            return
        
        noseprint_data = extractor.extract_noseprint(args.input)
        database.enroll(args.name, noseprint_data, args.input)
    
    elif args.identify:
        if not args.input:
            print("Error: --input required for identification")
            return
        
        noseprint_data = extractor.extract_noseprint(args.input)
        match, similarity, message = database.identify(noseprint_data['noseprint'], 
                                                       args.threshold)
        
        print(f"\n{'='*50}")
        print("IDENTIFICATION RESULT")
        print(f"{'='*50}")
        
        if match:
            print(f"✓ MATCH FOUND!")
            print(f"  Name: {match['name']}")
            print(f"  ID: {match['id']}")
            print(f"  Confidence: {similarity:.2%}")
            print(f"  Enrolled: {match['enrolled_date']}")
        else:
            print(f"✗ No match found")
            print(f"  {message}")
        
        print(f"{'='*50}\n")
    
    elif args.visualize:
        if not args.input or not args.output:
            print("Error: --input and --output required for visualization")
            return
        
        extractor.visualize_noseprint(args.input, args.output)
    
    elif args.list:
        dogs = database.list_dogs()
        print(f"\n{'='*50}")
        print(f"ENROLLED DOGS ({len(dogs)})")
        print(f"{'='*50}")
        
        for dog in dogs:
            print(f"\nName: {dog['name']}")
            print(f"  ID: {dog['id']}")
            print(f"  Enrolled: {dog['enrolled_date']}")
            print(f"  Nostrils: {dog['metadata']['nostril_count']}")
    
    else:
        print("Error: Specify --enroll, --identify, --visualize, or --list")


if __name__ == "__main__":
    main()