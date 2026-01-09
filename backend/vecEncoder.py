"""
Dog Nose Vector Encoding & Registration System

Creates unique 128-dimensional embeddings from dog nose images and stores them
in a searchable database. Identifies if a nose is already registered.

Features:
- Automatic embedding extraction from folders
- PostgreSQL + pgvector storage
- Duplicate detection
- JSON export option
- Batch processing
- Similarity search

Usage:
    # Encode all dogs and store in database
    python encode_noses.py --input dataset/dogs --register
    
    # Search if a nose is already registered
    python encode_noses.py --query path/to/new_nose.jpg --threshold 0.75

Requirements:
    pip install opencv-python numpy tensorflow asyncpg pgvector pillow
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# TensorFlow Lite for inference
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    try:
        from tensorflow.lite import Interpreter
    except Exception:
        print("⚠️  No TFLite runtime found. Install: pip install tflite-runtime")
        exit(1)

# Database
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    print("⚠️  asyncpg not installed. Database features disabled.")


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for embedding extraction.
    
    Args:
        image_path: Path to image
        target_size: Target image size (width, height)
    
    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize with high-quality resampling
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Apply CLAHE enhancement (improves ridge visibility)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    
    # Normalize to 0-1
    image_normalized = enhanced_rgb.astype(np.float32) / 255.0
    
    # Standardize (ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_standardized = (image_normalized - mean) / std
    
    # Add batch dimension
    image_batch = np.expand_dims(image_standardized, axis=0).astype(np.float32)
    
    return image_batch


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

class DogNoseEncoder:
    """Extracts 128-dimensional embeddings from dog nose images using TFLite model."""
    
    def __init__(self, model_path: str):
        """
        Initialize encoder.
        
        Args:
            model_path: Path to TFLite model (.tflite file)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load TFLite model
        self.interpreter = Interpreter(model_path=model_path, num_threads=4)
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.embedding_dim = self.output_details[0]['shape'][-1]
        
        print(f"✅ Loaded model: {model_path}")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Embedding dimension: {self.embedding_dim}")
    
    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract embedding from a single image.
        
        Args:
            image_path: Path to dog nose image
        
        Returns:
            128-dimensional embedding (L2-normalized)
        """
        # Preprocess image
        h, w = self.input_shape[1], self.input_shape[2]
        preprocessed = preprocess_image(image_path, target_size=(w, h))
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
        self.interpreter.invoke()
        
        # Extract embedding
        embedding = self.interpreter.get_tensor(self.output_details[0]['index'])
        embedding = np.squeeze(embedding)  # Remove batch dimension
        
        # L2 normalize (for cosine similarity)
        embedding = embedding / np.linalg.norm(embedding)
        
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
                print(f"  [{i}/{len(image_paths)}] Encoded: {Path(path).name}")
            except Exception as e:
                print(f"  ⚠️  Failed to encode {Path(path).name}: {e}")
        
        return np.array(embeddings)


# ============================================================================
# SIMILARITY COMPUTATION
# ============================================================================

def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Returns:
        Similarity score (0-1, higher = more similar)
    """
    return float(np.dot(embedding1, embedding2))


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two embeddings.
    
    Returns:
        Distance (lower = more similar)
    """
    return float(np.linalg.norm(embedding1 - embedding2))


def find_most_similar(
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Find most similar embeddings to query.
    
    Args:
        query_embedding: Query embedding
        gallery_embeddings: Database of embeddings (N x D)
        gallery_labels: Labels for each embedding
        top_k: Number of top matches to return
    
    Returns:
        List of (label, similarity_score) tuples
    """
    # Compute cosine similarities
    similarities = np.dot(gallery_embeddings, query_embedding)
    
    # Get top-k indices
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return (label, similarity) pairs
    results = [
        (gallery_labels[idx], float(similarities[idx]))
        for idx in top_k_indices
    ]
    
    return results


# ============================================================================
# DATASET PROCESSING
# ============================================================================

def load_dataset(dataset_dir: Path) -> Dict[str, List[Path]]:
    """
    Load dataset organized as:
    dataset/
      dog_001/
        img1.jpg
        img2.jpg
      dog_002/
        img1.jpg
        ...
    
    Returns:
        Dict mapping dog_id to list of image paths
    """
    dataset = {}
    
    for dog_dir in sorted(dataset_dir.iterdir()):
        if dog_dir.is_dir():
            dog_id = dog_dir.name
            
            # Find all images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            images = [
                f for f in dog_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            
            if images:
                dataset[dog_id] = images
    
    return dataset


def encode_dataset(encoder: DogNoseEncoder, dataset: Dict[str, List[Path]]) -> Dict[str, Dict]:
    """
    Encode entire dataset and compute average embedding per dog.
    
    Returns:
        Dict mapping dog_id to {
            'embeddings': List of individual embeddings,
            'avg_embedding': Average embedding,
            'image_paths': List of image paths
        }
    """
    encoded_dataset = {}
    
    print(f"\n🔄 Encoding {len(dataset)} dogs...")
    
    for dog_id, image_paths in dataset.items():
        print(f"\n📸 Dog: {dog_id} ({len(image_paths)} images)")
        
        # Extract embeddings for all images
        embeddings = encoder.extract_batch_embeddings(image_paths)
        
        if len(embeddings) > 0:
            # Compute average embedding
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # Normalize
            
            encoded_dataset[dog_id] = {
                'embeddings': embeddings.tolist(),
                'avg_embedding': avg_embedding.tolist(),
                'image_paths': [str(p) for p in image_paths],
                'num_images': len(embeddings)
            }
            
            print(f"  ✅ Encoded {len(embeddings)} images, avg embedding computed")
        else:
            print(f"  ⚠️  No valid embeddings extracted")
    
    return encoded_dataset


# ============================================================================
# JSON STORAGE (Simple option without database)
# ============================================================================

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
    
    # Convert lists back to numpy arrays
    for dog_id in data:
        data[dog_id]['embeddings'] = np.array(data[dog_id]['embeddings'])
        data[dog_id]['avg_embedding'] = np.array(data[dog_id]['avg_embedding'])
    
    return data


def search_in_json(
    query_image: str,
    encoder: DogNoseEncoder,
    json_path: Path,
    threshold: float = 0.75
) -> Optional[Tuple[str, float]]:
    """
    Search if query image is already registered (JSON storage).
    
    Returns:
        (dog_id, similarity) if match found, else None
    """
    # Load database
    database = load_from_json(json_path)
    
    # Extract query embedding
    print(f"\n🔍 Extracting embedding from query image...")
    query_embedding = encoder.extract_embedding(query_image)
    
    # Search for matches
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
    
    # Check threshold
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
# POSTGRESQL + PGVECTOR STORAGE (Production option)
# ============================================================================

async def init_database(database_url: str):
    """Initialize PostgreSQL database with pgvector extension."""
    conn = await asyncpg.connect(database_url)
    
    # Enable pgvector extension
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Create tables
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS dogs (
            dog_id VARCHAR(255) PRIMARY KEY,
            dog_name VARCHAR(255),
            num_images INTEGER,
            registered_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS nose_embeddings (
            embedding_id SERIAL PRIMARY KEY,
            dog_id VARCHAR(255) REFERENCES dogs(dog_id) ON DELETE CASCADE,
            embedding vector(128),
            image_path TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # Create vector index for fast similarity search
    try:
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS nose_embeddings_vector_idx 
            ON nose_embeddings 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
    except Exception as e:
        print(f"⚠️  Could not create vector index (normal if few rows): {e}")
    
    await conn.close()
    print("✅ Database initialized")


async def register_to_database(
    encoded_dataset: Dict,
    database_url: str
):
    """Register encoded dataset to PostgreSQL."""
    conn = await asyncpg.connect(database_url)
    
    print(f"\n💾 Registering {len(encoded_dataset)} dogs to database...")
    
    for dog_id, data in encoded_dataset.items():
        # Insert dog record
        await conn.execute("""
            INSERT INTO dogs (dog_id, num_images)
            VALUES ($1, $2)
            ON CONFLICT (dog_id) DO UPDATE
            SET num_images = EXCLUDED.num_images
        """, dog_id, data['num_images'])
        
        # Delete existing embeddings for this dog
        await conn.execute("DELETE FROM nose_embeddings WHERE dog_id = $1", dog_id)
        
        # Insert embeddings
        for i, (embedding, image_path) in enumerate(zip(data['embeddings'], data['image_paths'])):
            await conn.execute("""
                INSERT INTO nose_embeddings (dog_id, embedding, image_path)
                VALUES ($1, $2, $3)
            """, dog_id, embedding.tolist(), image_path)
        
        print(f"  ✅ Registered: {dog_id} ({data['num_images']} embeddings)")
    
    await conn.close()
    print(f"\n✅ Registration complete!")


async def search_in_database(
    query_image: str,
    encoder: DogNoseEncoder,
    database_url: str,
    threshold: float = 0.75,
    top_k: int = 5
) -> Optional[List[Tuple[str, float]]]:
    """
    Search if query image is already registered (database storage).
    
    Returns:
        List of (dog_id, similarity) tuples if matches found
    """
    conn = await asyncpg.connect(database_url)
    
    # Extract query embedding
    print(f"\n🔍 Extracting embedding from query image...")
    query_embedding = encoder.extract_embedding(query_image)
    
    # Search database using pgvector similarity
    print(f"🔎 Searching database...")
    
    rows = await conn.fetch("""
        SELECT 
            ne.dog_id,
            d.dog_name,
            d.num_images,
            1 - (ne.embedding <=> $1) AS similarity
        FROM nose_embeddings ne
        JOIN dogs d ON ne.dog_id = d.dog_id
        ORDER BY similarity DESC
        LIMIT $2
    """, query_embedding.tolist(), top_k)
    
    await conn.close()
    
    # Filter by threshold
    matches = [
        (row['dog_id'], float(row['similarity']))
        for row in rows
        if float(row['similarity']) >= threshold
    ]
    
    if matches:
        print(f"\n✅ FOUND {len(matches)} MATCH(ES):")
        for dog_id, similarity in matches:
            print(f"   Dog {dog_id}: similarity = {similarity:.4f}")
        return matches
    else:
        best_similarity = float(rows[0]['similarity']) if rows else 0.0
        print(f"\n❌ NO MATCH FOUND")
        print(f"   Best similarity: {best_similarity:.4f} < threshold: {threshold}")
        print(f"   This appears to be a NEW dog (not registered)")
        return None


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dog Nose Vector Encoding & Registration System"
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--encode',
        type=str,
        metavar='DIR',
        help='Encode dataset directory (e.g., dataset/dogs)'
    )
    mode_group.add_argument(
        '--query',
        type=str,
        metavar='IMAGE',
        help='Query single image to check if registered'
    )
    
    # Model
    parser.add_argument(
        '--model',
        type=str,
        default='models/dog_nose_embedder.tflite',
        help='Path to TFLite model (default: models/dog_nose_embedder.tflite)'
    )
    
    # Storage options
    storage_group = parser.add_mutually_exclusive_group()
    storage_group.add_argument(
        '--json',
        type=str,
        default='encoded_noses.json',
        help='JSON file for storage (default: encoded_noses.json)'
    )
    storage_group.add_argument(
        '--database',
        type=str,
        metavar='URL',
        help='PostgreSQL database URL (e.g., postgresql://user:pass@localhost/db)'
    )
    
    # Search parameters
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.75,
        help='Similarity threshold for matching (default: 0.75)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Return top-k matches (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Initialize encoder
    print("🚀 Dog Nose Vector Encoding System")
    print("=" * 60)
    
    encoder = DogNoseEncoder(args.model)
    
    # ENCODE MODE
    if args.encode:
        dataset_dir = Path(args.encode)
        
        if not dataset_dir.exists():
            print(f"❌ Dataset directory not found: {dataset_dir}")
            return
        
        # Load dataset
        print(f"\n📂 Loading dataset from: {dataset_dir}")
        dataset = load_dataset(dataset_dir)
        
        if not dataset:
            print("❌ No dogs found in dataset!")
            print("Expected structure:")
            print("  dataset/")
            print("    dog_001/")
            print("      img1.jpg")
            print("      img2.jpg")
            return
        
        print(f"✅ Found {len(dataset)} dogs")
        
        # Encode
        encoded_dataset = encode_dataset(encoder, dataset)
        
        # Save
        if args.database:
            if not HAS_ASYNCPG:
                print("❌ asyncpg not installed. Install: pip install asyncpg")
                return
            
            # Database storage
            asyncio.run(init_database(args.database))
            asyncio.run(register_to_database(encoded_dataset, args.database))
        else:
            # JSON storage
            save_to_json(encoded_dataset, Path(args.json))
    
    # QUERY MODE
    elif args.query:
        query_image = Path(args.query)
        
        if not query_image.exists():
            print(f"❌ Query image not found: {query_image}")
            return
        
        print(f"\n📸 Query image: {query_image}")
        
        if args.database:
            if not HAS_ASYNCPG:
                print("❌ asyncpg not installed. Install: pip install asyncpg")
                return
            
            # Search database
            matches = asyncio.run(search_in_database(
                str(query_image),
                encoder,
                args.database,
                args.threshold,
                args.top_k
            ))
        else:
            # Search JSON
            json_path = Path(args.json)
            
            if not json_path.exists():
                print(f"❌ Database file not found: {json_path}")
                print("Run --encode first to create database!")
                return
            
            match = search_in_json(
                str(query_image),
                encoder,
                json_path,
                args.threshold
            )


if __name__ == "__main__":
    main()
