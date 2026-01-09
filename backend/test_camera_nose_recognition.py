"""
Dog Nose Recognition Test Script - CNN Based
Captures dog nose images from camera and compares against CNN embeddings
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
try:
    from cnn_encoder import CNNDogNoseEncoder, cosine_similarity
    USE_CNN = True
except ImportError:
    from vecEncoder_demo import SimpleDogNoseEncoder as CNNDogNoseEncoder, cosine_similarity
    USE_CNN = False


class CameraNoseRecognizer:
    """Real-time dog nose recognition from camera feed using CNN."""
    
    def __init__(self, json_database: str, embedding_dim: int = 256):
        """
        Initialize recognizer with CNN encoder and database.
        
        Args:
            json_database: Path to encoded_noses_cnn.json
            embedding_dim: Embedding dimension
        """
        self.encoder = CNNDogNoseEncoder(embedding_dim=embedding_dim)
        self.database_path = Path(json_database)
        self.database = self.load_database()
        
        print(f"✅ Loaded database with {len(self.database)} dogs")
        for dog_id, data in self.database.items():
            model_info = data.get('model', 'CNN-MobileNetV2')
            print(f"   - {dog_id}: {data['num_images']} images ({model_info})")
    
    def load_database(self) -> Dict:
        """Load encoded noses from JSON database."""
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {self.database_path}")
        
        with open(self.database_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        for dog_id in data:
            data[dog_id]['embeddings'] = np.array(data[dog_id]['embeddings'])
            data[dog_id]['avg_embedding'] = np.array(data[dog_id]['avg_embedding'])
        
        return data
    
    def identify_nose(self, image: np.ndarray, threshold: float = 0.3, top_k: int = 3) -> Dict:
        """
        Identify dog nose from image.
        
        Args:
            image: Input image (BGR from OpenCV)
            threshold: Minimum similarity threshold
            top_k: Number of top matches to return
        
        Returns:
            Dict with match results
        """
        try:
            # Save temporary image
            temp_path = "/tmp/nose_frame.jpg"
            cv2.imwrite(temp_path, image)
            
            # Extract embedding
            embedding = self.encoder.extract_embedding(temp_path)
            
            # Compare against all registered dogs
            similarities = {}
            for dog_id, data in self.database.items():
                avg_emb = data['avg_embedding']
                sim = cosine_similarity(embedding, avg_emb)
                similarities[dog_id] = sim
            
            # Sort by similarity
            sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Get top matches above threshold
            matches = [
                {"dog_id": dog_id, "similarity": float(sim)}
                for dog_id, sim in sorted_matches[:top_k]
                if sim >= threshold
            ]
            
            best_match = sorted_matches[0] if sorted_matches else None
            
            result = {
                "found_match": len(matches) > 0,
                "matches": matches,
                "best_match": {
                    "dog_id": best_match[0],
                    "similarity": float(best_match[1])
                } if best_match else None,
                "threshold": threshold,
                "all_similarities": {dog_id: float(sim) for dog_id, sim in sorted_matches}
            }
            
            return result
        
        except Exception as e:
            return {
                "found_match": False,
                "error": str(e),
                "matches": []
            }
    
    def run_camera_test(
        self,
        camera_source = 0,
        threshold: float = 0.3,
        capture_delay: int = 500,
        display: bool = True,
        retry_seconds: int = 10,
        exit_on_match: bool = False,
        confidence_threshold: float = 0.4
    ):
        """
        Run real-time camera test for nose recognition.
        
        Args:
            camera_source: Camera device ID (int) or URL (str)
            threshold: Similarity threshold for match
            capture_delay: Milliseconds between captures
            display: Whether to display results on screen
            retry_seconds: Seconds to retry before timeout (0 = infinite)
            exit_on_match: Exit on successful match
            confidence_threshold: High-confidence match threshold (for early exit)
        """
        print(f"\n🎥 Starting camera test...")
        print(f"   Source: {camera_source}")
        print(f"   Threshold: {threshold}")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   Capture interval: {capture_delay}ms")
        if retry_seconds > 0:
            print(f"   ⏱️  Timeout: {retry_seconds} seconds")
        if exit_on_match:
            print(f"   🎯 EXIT ON MATCH enabled")
        print(f"   🤖 Model: CNN-ResNet50 (Transfer Learning) - ROBUST")
        print(f"   📊 Embedding: 256-dimensional vectors")
        print(f"   Press 'c' to capture, 's' to save, 'q' to quit")
        print("=" * 60)
        
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_source}")
            return
        
        frame_count = 0
        last_capture = 0
        start_time = time.time()
        match_found = False
        
        while True:
            # Check timeout
            if retry_seconds > 0:
                elapsed = time.time() - start_time
                if elapsed > retry_seconds:
                    print(f"\n⏱️  Timeout! No match found after {retry_seconds} seconds")
                    print(f"   Total frames processed: {frame_count}")
                    break
            
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Auto-capture periodically
            if (current_time - last_capture) * 1000 >= capture_delay:
                result = self.identify_nose(frame, threshold=threshold)
                last_capture = current_time
                
                # Display result
                if result.get('found_match', False):
                    best_sim = result['best_match']['similarity']
                    best_dog = result['best_match']['dog_id']
                    
                    if best_sim >= confidence_threshold:
                        status = f"🎯 CONFIDENT MATCH: {best_dog} ({best_sim:.3f})"
                        color = (0, 255, 0)  # Green
                        match_found = True
                    else:
                        status = f"✅ MATCH: {best_dog} ({best_sim:.3f})"
                        color = (0, 255, 0)  # Green
                else:
                    best_match = result.get('best_match')
                    if best_match:
                        status = f"❌ NO MATCH - Best: {best_match['dog_id']} ({best_match['similarity']:.3f})"
                    else:
                        status = f"❌ NO MATCH - Processing frames"
                    color = (0, 0, 255)  # Red
                
                print(f"[Frame {frame_count}] {status}")
                
                # Exit on confident match
                if exit_on_match and match_found and best_sim >= confidence_threshold:
                    print(f"\n✅ SUCCESS! Dog identified as: {best_dog}")
                    print(f"   Confidence: {best_sim:.4f}")
                    print(f"   Frames processed: {frame_count}")
                    print(f"   Time taken: {elapsed:.2f}s")
                    break
            
            # Display on screen
            if display:
                display_frame = frame.copy()
                
                # Add text
                cv2.putText(
                    display_frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                if (current_time - last_capture) * 1000 >= capture_delay:
                    if result['found_match']:
                        cv2.putText(
                            display_frame,
                            f"MATCH: {result['matches'][0]['dog_id']} ({result['matches'][0]['similarity']:.3f})",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2
                        )
                    else:
                        cv2.putText(
                            display_frame,
                            f"NO MATCH",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2
                        )
                        if result['best_match']:
                            cv2.putText(
                                display_frame,
                                f"Best: {result['best_match']['dog_id']} ({result['best_match']['similarity']:.3f})",
                                (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (255, 165, 0),
                                2
                            )
                
                cv2.imshow('Dog Nose Recognition', display_frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quit requested")
                break
            elif key == ord('c'):
                print("📸 Manual capture requested")
            elif key == ord('s'):
                save_path = f"nose_capture_{int(time.time())}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"💾 Saved: {save_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera test ended")
    
    def test_image_file(self, image_path: str, threshold: float = 0.3):
        """
        Test recognition on a single image file.
        
        Args:
            image_path: Path to image file
            threshold: Similarity threshold
        """
        image_file = Path(image_path)
        
        if not image_file.exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"\n📸 Testing image: {image_path}")
        print("=" * 60)
        
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"❌ Cannot read image: {image_path}")
            return
        
        result = self.identify_nose(image, threshold=threshold)
        
        # Print results
        print(f"\nThreshold: {threshold}")
        print(f"All similarities:")
        for dog_id, sim in sorted(result['all_similarities'].items(), key=lambda x: x[1], reverse=True):
            status = "✓" if sim >= threshold else " "
            print(f"  [{status}] {dog_id}: {sim:.4f}")
        
        if result['found_match']:
            print(f"\n✅ MATCH FOUND!")
            print(f"   Best match: {result['best_match']['dog_id']}")
            print(f"   Similarity: {result['best_match']['similarity']:.4f}")
            print(f"\n   All matches above threshold ({threshold}):")
            for match in result['matches']:
                print(f"     - {match['dog_id']}: {match['similarity']:.4f}")
        else:
            print(f"\n❌ NO MATCH FOUND")
            if result['best_match']:
                print(f"   Best attempt: {result['best_match']['dog_id']}")
                print(f"   Similarity: {result['best_match']['similarity']:.4f} < {threshold}")
    
    def batch_test(self, image_dir: str, threshold: float = 0.3):
        """
        Test recognition on all images in a directory.
        
        Args:
            image_dir: Directory containing test images
            threshold: Similarity threshold
        """
        test_dir = Path(image_dir)
        
        if not test_dir.exists():
            print(f"❌ Directory not found: {image_dir}")
            return
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [
            f for f in test_dir.rglob('*')
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        print(f"\n📂 Batch test on {len(images)} images from: {image_dir}")
        print("=" * 60)
        
        results = {
            'total': len(images),
            'matches': 0,
            'no_matches': 0,
            'errors': 0,
            'details': []
        }
        
        for i, image_path in enumerate(images, 1):
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"[{i}/{len(images)}] ⚠️  Cannot read: {image_path.name}")
                    results['errors'] += 1
                    continue
                
                result = self.identify_nose(image, threshold=threshold)
                
                if result['found_match']:
                    match_info = result['best_match']
                    print(f"[{i}/{len(images)}] ✅ {image_path.name} -> {match_info['dog_id']} ({match_info['similarity']:.4f})")
                    results['matches'] += 1
                else:
                    print(f"[{i}/{len(images)}] ❌ {image_path.name} -> No match")
                    results['no_matches'] += 1
                
                results['details'].append({
                    'image': str(image_path.relative_to(test_dir.parent)),
                    'found_match': result['found_match'],
                    'best_match': result['best_match'],
                    'all_similarities': result['all_similarities']
                })
            
            except Exception as e:
                print(f"[{i}/{len(images)}] ❌ Error: {image_path.name} - {e}")
                results['errors'] += 1
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 BATCH TEST SUMMARY")
        print("=" * 60)
        print(f"Total: {results['total']}")
        print(f"Matches: {results['matches']} ({100*results['matches']/results['total']:.1f}%)")
        print(f"No matches: {results['no_matches']} ({100*results['no_matches']/results['total']:.1f}%)")
        print(f"Errors: {results['errors']} ({100*results['errors']/results['total']:.1f}%)")
        
        return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dog Nose Recognition Test - Camera & Image Testing"
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--camera',
        type=str,
        nargs='?',
        const='0',
        metavar='DEVICE',
        help='Run real-time camera test (device ID or URL, default: 0)'
    )
    mode_group.add_argument(
        '--source',
        type=str,
        metavar='SOURCE',
        help='Camera source (device ID or URL, e.g., http://ip:port/video)'
    )
    mode_group.add_argument(
        '--image',
        type=str,
        metavar='PATH',
        help='Test single image file'
    )
    mode_group.add_argument(
        '--batch',
        type=str,
        metavar='DIR',
        help='Batch test all images in directory'
    )
    
    # Database
    parser.add_argument(
        '--database',
        type=str,
        default='encoded_noses_cnn.json',
        help='Path to encoded noses database (default: encoded_noses_cnn.json)'
    )
    
    # Recognition parameters
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Similarity threshold for match (default: 0.3)'
    )
    
    parser.add_argument(
        '--capture-interval',
        type=int,
        default=500,
        metavar='MS',
        help='Milliseconds between camera captures (default: 500ms)'
    )
    
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=128,
        help='Embedding dimension (default: 128)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable display window for camera test'
    )
    
    parser.add_argument(
        '--exit-on-match',
        action='store_true',
        help='Exit immediately on successful match'
    )
    
    parser.add_argument(
        '--retry',
        type=int,
        default=10,
        metavar='SECONDS',
        help='Retry timeout in seconds (0 = infinite, default: 10)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        metavar='SCORE',
        help='Confidence threshold for auto-match (default: 0.25)'
    )
    
    args = parser.parse_args()
    
    print("🚀 Dog Nose Recognition Test System")
    print("=" * 60)
    
    try:
        recognizer = CameraNoseRecognizer(args.database, embedding_dim=256)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Determine camera source
    camera_source = None
    if args.camera is not None:
        # Try to convert to int for device ID, otherwise treat as URL
        try:
            camera_source = int(args.camera)
        except ValueError:
            camera_source = args.camera
    elif args.source is not None:
        # Try to convert to int for device ID, otherwise treat as URL
        try:
            camera_source = int(args.source)
        except ValueError:
            camera_source = args.source
    
    # Camera/source mode
    if camera_source is not None:
        recognizer.run_camera_test(
            camera_source=camera_source,
            threshold=args.threshold,
            capture_delay=args.capture_interval,
            display=not args.no_display,
            retry_seconds=args.retry,
            exit_on_match=args.exit_on_match,
            confidence_threshold=args.confidence
        )
    
    # Single image mode
    elif args.image:
        recognizer.test_image_file(args.image, threshold=args.threshold)
    
    # Batch mode
    elif args.batch:
        recognizer.batch_test(args.batch, threshold=args.threshold)


if __name__ == "__main__":
    main()
