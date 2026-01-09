"""
Dog Nose Vector Encoding Demo - Works without TFLite model
Extracts simple feature embeddings from dog nose images using OpenCV
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ============================================================================
# SIMPLE FEATURE EXTRACTION (No TFLite needed)
# ============================================================================

class SimpleDogNoseEncoder:
    """Extracts feature embeddings from dog nose images using traditional CV."""
    
    def __init__(self, embedding_dim: int = 128):
        """
        Initialize encoder.
        
        Args:
            embedding_dim: Dimension of embedding vector
        """
        self.embedding_dim = embedding_dim
        print(f"✅ Initialized SimpleDogNoseEncoder")
        print(f"   Embedding dimension: {self.embedding_dim}")
    
    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract embedding from a single image using feature descriptors.
        
        Uses: histogram, edges, corners, and texture features
        
        Args:
            image_path: Path to dog nose image
        
        Returns:
            embedding_dim-dimensional embedding (L2-normalized)
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to consistent size
        gray = cv2.resize(gray, (224, 224))
        
        features = []
        
        # 1. Histogram features (16 bins)
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
        
        # 2. Edge detection (Canny)
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [8], [0, 256])
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
        features.extend(edge_hist)
        
        # 3. Corner detection (Harris)
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corners = cv2.normalize(corners, corners).flatten()[:16]
        features.extend(corners)
        
        # 4. Texture features using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_features = laplacian.flatten()[:32]
        features.extend(laplacian_features)
        
        # 5. Gaussian blur difference (LoG)
        blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
        log_features = (gray.astype(np.float32) - blur.astype(np.float32)).flatten()[:32]
        features.extend(log_features)
        
        # 6. Local Binary Pattern-like features
        lbp_features = []
        for i in range(1, 7):
            hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            lbp_features.extend(hist)
        features.extend(lbp_features)
        
        # Convert to numpy array
        embedding = np.array(features, dtype=np.float32)
        
        # Trim or pad to target dimension
        if len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]
        else:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)), mode='constant')
        
        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding.astype(np.float32)
    
    def extract_batch_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract embeddings from multiple images.
        
        Args:
            image_paths: List of image paths
        
        Returns:
            Array of embeddings (N x embedding_dim)
        """
        embeddings = []
        
        for i, path in enumerate(image_paths, 1):
            try:
                emb = self.extract_embedding(path)
                embeddings.append(emb)
                print(f"  [{i}/{len(image_paths)}] ✅ Encoded: {Path(path).name}")
            except Exception as e:
                print(f"  [{i}/{len(image_paths)}] ⚠️  Failed: {Path(path).name} - {e}")
        
        return np.array(embeddings) if embeddings else np.array([])


# ============================================================================
# SIMILARITY COMPUTATION
# ============================================================================

def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(embedding1, embedding2))


def find_most_similar(
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Find most similar embeddings to query."""
    if len(gallery_embeddings) == 0:
        return []
    
    similarities = np.dot(gallery_embeddings, query_embedding)
    top_k_indices = np.argsort(similarities)[::-1][:min(top_k, len(similarities))]
    
    results = [
        (gallery_labels[idx], float(similarities[idx]))
        for idx in top_k_indices
    ]
    
    return results


# ============================================================================
# DATASET PROCESSING
# ============================================================================

def load_dataset(dataset_dir: Path) -> Dict[str, List[Path]]:
    """Load dataset organized as dataset/dog_xxx/img.jpg"""
    dataset = {}
    
    for dog_dir in sorted(dataset_dir.iterdir()):
        if dog_dir.is_dir():
            dog_id = dog_dir.name
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            images = [
                f for f in dog_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            
            if images:
                dataset[dog_id] = sorted(images)
    
    return dataset


def encode_dataset(encoder: SimpleDogNoseEncoder, dataset: Dict[str, List[Path]], dataset_dir: Path = None) -> Dict[str, Dict]:
    """Encode entire dataset."""
    encoded_dataset = {}
    
    print(f"\n🔄 Encoding {len(dataset)} dogs...")
    
    for dog_id, image_paths in dataset.items():
        print(f"\n📸 Dog: {dog_id} ({len(image_paths)} images)")
        
        embeddings = encoder.extract_batch_embeddings(image_paths)
        
        if len(embeddings) > 0:
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
            
            if dataset_dir:
                rel_paths = [str(p.relative_to(dataset_dir.parent)) for p in image_paths]
            else:
                rel_paths = [str(p) for p in image_paths]
            
            encoded_dataset[dog_id] = {
                'embeddings': embeddings.tolist(),
                'avg_embedding': avg_embedding.tolist(),
                'image_paths': rel_paths,
                'num_images': len(embeddings)
            }
            
            print(f"  ✅ Encoded {len(embeddings)} images")
        else:
            print(f"  ⚠️  No valid embeddings extracted")
    
    return encoded_dataset


def save_to_json(encoded_dataset: Dict, output_path: Path):
    """Save encoded dataset to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(encoded_dataset, f, indent=2)
    
    print(f"\n💾 Saved to: {output_path}")
    print(f"   Total dogs: {len(encoded_dataset)}")


def load_from_json(json_path: Path) -> Dict:
    """Load encoded dataset from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for dog_id in data:
        data[dog_id]['embeddings'] = np.array(data[dog_id]['embeddings'])
        data[dog_id]['avg_embedding'] = np.array(data[dog_id]['avg_embedding'])
    
    return data


def search_in_json(
    query_image: str,
    encoder: SimpleDogNoseEncoder,
    json_path: Path,
    threshold: float = 0.75
) -> Optional[Tuple[str, float]]:
    """Search if query image is already registered."""
    database = load_from_json(json_path)
    
    print(f"\n🔍 Extracting embedding from query image...")
    query_embedding = encoder.extract_embedding(query_image)
    
    print(f"🔎 Searching {len(database)} registered dogs...")
    
    best_match = None
    best_similarity = 0.0
    
    for dog_id, data in database.items():
        avg_embedding = data['avg_embedding']
        similarity = cosine_similarity(query_embedding, avg_embedding)
        
        print(f"  Dog {dog_id}: similarity = {similarity:.4f}")
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = dog_id
    
    if best_similarity >= threshold:
        print(f"\n✅ MATCH FOUND!")
        print(f"   Dog ID: {best_match}")
        print(f"   Similarity: {best_similarity:.4f} (threshold: {threshold})")
        return best_match, best_similarity
    else:
        print(f"\n❌ NO MATCH FOUND")
        print(f"   Best similarity: {best_similarity:.4f} < threshold: {threshold}")
        print(f"   This appears to be a NEW dog (not registered)")
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dog Nose Vector Encoding Demo (No TFLite required)"
    )
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--encode',
        type=str,
        metavar='DIR',
        help='Encode dataset directory'
    )
    mode_group.add_argument(
        '--query',
        type=str,
        metavar='IMAGE',
        help='Query single image to check if registered'
    )
    
    parser.add_argument(
        '--json',
        type=str,
        default='encoded_noses.json',
        help='JSON file for storage (default: encoded_noses.json)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Similarity threshold (default: 0.7)'
    )
    
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=128,
        help='Embedding dimension (default: 128)'
    )
    
    args = parser.parse_args()
    
    print("🚀 Dog Nose Vector Encoding Demo")
    print("=" * 60)
    
    encoder = SimpleDogNoseEncoder(embedding_dim=args.embedding_dim)
    
    # ENCODE MODE
    if args.encode:
        dataset_dir = Path(args.encode)
        
        if not dataset_dir.exists():
            print(f"❌ Dataset directory not found: {dataset_dir}")
            return
        
        print(f"\n📂 Loading dataset from: {dataset_dir}")
        dataset = load_dataset(dataset_dir)
        
        if not dataset:
            print("❌ No dogs found in dataset!")
            print("Expected structure: dataset/dog_001/img.jpg")
            return
        
        print(f"✅ Found {len(dataset)} dogs")
        
        encoded_dataset = encode_dataset(encoder, dataset, dataset_dir)
        save_to_json(encoded_dataset, Path(args.json))
    
    # QUERY MODE
    elif args.query:
        query_image = Path(args.query)
        
        if not query_image.exists():
            print(f"❌ Query image not found: {query_image}")
            return
        
        print(f"\n📸 Query image: {query_image}")
        
        json_path = Path(args.json)
        
        if not json_path.exists():
            print(f"❌ Database file not found: {json_path}")
            print("Run --encode first to create database!")
            return
        
        search_in_json(str(query_image), encoder, json_path, args.threshold)


if __name__ == "__main__":
    main()
