
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# For semantic segmentation (alternative approach)
try:
    import segmentation_models_pytorch as smp
    from PIL import Image
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not available. Install with: pip install segmentation-models-pytorch")

class DogNoseDetector:
    """
    Dog nose detector using a combination of object detection and segmentation.
    
    This implementation uses:
    1. A pretrained segmentation model (DeepLabV3) for initial nose region detection
    2. Traditional CV techniques for nostril identification
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load pretrained DeepLabV3 model (trained on COCO/Pascal VOC)
        # This can detect animals but may need fine-tuning for specific nose detection
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 
                                     'deeplabv3_resnet101', 
                                     pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transforms
        self.preprocess = torch.nn.Sequential(
            # Normalize using ImageNet statistics
        )
        
    def preprocess_image(self, image):
        """Preprocess image for the model"""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize and normalize
        img_resized = cv2.resize(img_rgb, (520, 520))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_normalized = (img_tensor - mean) / std
        
        return img_normalized.unsqueeze(0).to(self.device)
    
    def detect_nose_region(self, image):
        """
        Detect the nose region using semantic segmentation.
        Returns a binary mask of the detected nose area.
        """
        h, w = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # Get segmentation mask (class 12 is 'dog' in Pascal VOC)
        output_predictions = output.argmax(0).cpu().numpy()
        
        # Resize mask to original image size
        mask = cv2.resize((output_predictions == 12).astype(np.uint8), 
                         (w, h), 
                         interpolation=cv2.INTER_NEAREST)
        
        # Focus on upper center region (where nose typically is)
        nose_region_mask = self.isolate_nose_from_dog(mask, image)
        
        return nose_region_mask
    
    def isolate_nose_from_dog(self, dog_mask, image):
        """
        Isolate the nose region from a full dog segmentation mask.
        Uses heuristics: nose is typically in upper-center, dark, and has specific texture.
        """
        h, w = image.shape[:2]
        
        # Create a region of interest (upper center portion)
        roi_mask = np.zeros_like(dog_mask)
        roi_mask[int(h*0.2):int(h*0.6), int(w*0.3):int(w*0.7)] = 1
        
        # Apply ROI to dog mask
        candidate_mask = dog_mask * roi_mask
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Nose is typically darker - use adaptive thresholding
        # Apply only within the candidate region
        masked_gray = gray.copy()
        masked_gray[candidate_mask == 0] = 255
        
        # Adaptive threshold to find dark regions
        thresh = cv2.adaptiveThreshold(masked_gray, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Combine with candidate mask
        nose_mask = cv2.bitwise_and(thresh, thresh, mask=candidate_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        nose_mask = cv2.morphologyEx(nose_mask, cv2.MORPH_CLOSE, kernel)
        nose_mask = cv2.morphologyEx(nose_mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour (likely the nose)
        contours, _ = cv2.findContours(nose_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            nose_mask = np.zeros_like(nose_mask)
            cv2.drawContours(nose_mask, [largest_contour], -1, 255, -1)
        
        return nose_mask
    
    def detect_nostrils(self, nose_mask, image):
        """
        Detect individual nostrils within the nose mask.
        Returns contours of the two nostrils.
        """
        # Extract nose region
        nose_region = cv2.bitwise_and(image, image, mask=nose_mask)
        gray_nose = cv2.cvtColor(nose_region, cv2.COLOR_BGR2GRAY)
        
        # Nostrils are the darkest regions within the nose
        # Apply threshold to find very dark regions
        _, nostril_mask = cv2.threshold(gray_nose, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Apply nose mask to ensure we're only looking at nose region
        nostril_mask = cv2.bitwise_and(nostril_mask, nostril_mask, mask=nose_mask)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        nostril_mask = cv2.morphologyEx(nostril_mask, cv2.MORPH_OPEN, kernel)
        nostril_mask = cv2.morphologyEx(nostril_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(nostril_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and circularity
        nostril_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50 and area < 5000:  # Reasonable size for nostrils
                # Check circularity
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Somewhat circular
                        nostril_contours.append(cnt)
        
        # Keep top 2 largest (the nostrils)
        nostril_contours = sorted(nostril_contours, 
                                 key=cv2.contourArea, 
                                 reverse=True)[:2]
        
        return nostril_contours, nostril_mask
    
    def annotate_image(self, image, nose_mask, nostril_contours):
        """
        Draw annotations on the image.
        """
        result = image.copy()
        
        # Draw nose outline in green
        nose_contours, _ = cv2.findContours(nose_mask, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, nose_contours, -1, (0, 255, 0), 3)
        
        # Draw nostrils in red
        cv2.drawContours(result, nostril_contours, -1, (0, 0, 255), 2)
        
        # Fill nostrils semi-transparently
        overlay = result.copy()
        cv2.drawContours(overlay, nostril_contours, -1, (0, 0, 255), -1)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
        
        # Add labels
        cv2.putText(result, "Nose Region", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        for i, cnt in enumerate(nostril_contours):
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(result, f"Nostril {i+1}", 
                           (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 2)
        
        return result
    
    def process_image(self, image_path, output_path, save_masks=True):
        """
        Main processing pipeline.
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing image: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Detect nose region
        print("Detecting nose region...")
        nose_mask = self.detect_nose_region(image)
        
        # Detect nostrils
        print("Detecting nostrils...")
        nostril_contours, nostril_mask = self.detect_nostrils(nose_mask, image)
        print(f"Found {len(nostril_contours)} nostril(s)")
        
        # Annotate image
        print("Creating annotated image...")
        annotated = self.annotate_image(image, nose_mask, nostril_contours)
        
        # Save results
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved annotated image to: {output_path}")
        
        if save_masks:
            # Save nose mask
            mask_path = output_path.parent / f"{output_path.stem}_nose_mask.png"
            cv2.imwrite(str(mask_path), nose_mask)
            print(f"Saved nose mask to: {mask_path}")
            
            # Save nostril mask
            nostril_mask_path = output_path.parent / f"{output_path.stem}_nostril_mask.png"
            cv2.imwrite(str(nostril_mask_path), nostril_mask)
            print(f"Saved nostril mask to: {nostril_mask_path}")
        
        return annotated, nose_mask, nostril_mask


def main():
    parser = argparse.ArgumentParser(description='Detect and segment dog nose in images')
    parser.add_argument('--input', type=str, default='image.jpg',
                       help='Input image path')
    parser.add_argument('--output', type=str, default='annotated_image.jpg',
                       help='Output annotated image path')
    parser.add_argument('--no-masks', action='store_true',
                       help='Do not save binary masks')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Check if input exists
    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}")
        return
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = DogNoseDetector()
    
    # Process image
    try:
        detector.process_image(input_path, output_path, save_masks=not args.no_masks)
        print("\n✓ Processing complete!")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


"""
=== TRAINING A CUSTOM MODEL ===

For better results, you should train a custom model specifically for dog nose segmentation.

Dataset Requirements:
1. Images: 500-2000+ images of dog faces/noses from various angles, breeds, lighting
2. Annotations: Pixel-level segmentation masks with labels:
   - Background (0)
   - Nose region (1)
   - Left nostril (2)
   - Right nostril (3)

Recommended Datasets:
- Oxford-IIIT Pet Dataset (has dog images, needs custom annotation)
- Stanford Dogs Dataset (120+ breeds)
- Custom dataset using labelme or CVAT for annotation

Training Approach:
1. Use U-Net, DeepLabV3+, or Mask R-CNN architecture
2. Transfer learning from ImageNet weights
3. Data augmentation: rotation, scaling, color jitter, flips
4. Loss function: Dice loss + Cross-entropy
5. Train for 50-100 epochs with learning rate scheduling

Sample Training Code Structure:
```python
import segmentation_models_pytorch as smp

# Create model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    classes=4,  # background, nose, nostril_left, nostril_right
    activation=None,
)

# Train with your dataset
# ... training loop ...
```

For production use, consider:
- Fine-tuning on your specific dog breeds
- Ensemble methods for better accuracy
- Post-processing with conditional random fields (CRF)
"""