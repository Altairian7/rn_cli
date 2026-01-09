"""
CNN-Based Dog Nose Recognition Using Transfer Learning
Uses MobileNetV2 for real-time inference with high accuracy
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CNNDogNoseEncoder:
    """CNN-based dog nose encoder using transfer learning (MobileNetV2)."""
    
    def __init__(self, embedding_dim: int = 256, model_path: str = None):
        """
        Initialize CNN encoder with transfer learning.
        
        Args:
            embedding_dim: Embedding dimension (256 for MobileNetV2)
            model_path: Path to pre-trained model weights
        """
        self.embedding_dim = embedding_dim
        self.model_path = model_path
        self.model = None
        self.encoder = None
        
        print("🚀 Initializing CNN Dog Nose Encoder...")
        print(f"   Model: ResNet50 (Transfer Learning)")
        print(f"   Embedding dimension: {embedding_dim}")
        
        self._build_model()
        
        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)
            print(f"✅ Loaded pre-trained weights: {model_path}")
        else:
            print("✅ Using pre-trained ResNet50 from ImageNet")
    
    def _build_model(self):
        """Build CNN model with ResNet50 transfer learning."""
        # Load pre-trained ResNet50 (better for dog noses than MobileNetV2)
        base_model = ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Build model
        inputs = Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        
        # Add dense layers
        x = Dense(1024, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Dense(self.embedding_dim, activation='relu')(x)
        
        # L2 normalize output
        outputs = tf.keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1)
        )(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.encoder = self.model
        
        print(f"✅ CNN model built successfully (ResNet50)")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for CNN."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (224, 224))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return np.expand_dims(image, axis=0)
    
    def extract_embedding(self, image_path: str) -> np.ndarray:
        """Extract embedding from image using CNN."""
        image_batch = self.preprocess_image(image_path)
        embedding = self.encoder.predict(image_batch, verbose=0)
        return embedding[0].astype(np.float32)
    
    def extract_batch_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """Extract embeddings from multiple images."""
        embeddings = []
        
        for i, path in enumerate(image_paths, 1):
            try:
                emb = self.extract_embedding(path)
                embeddings.append(emb)
                print(f"  [{i}/{len(image_paths)}] ✅ Encoded: {Path(path).name}")
            except Exception as e:
                print(f"  [{i}/{len(image_paths)}] ⚠️  Failed: {Path(path).name} - {e}")
        
        return np.array(embeddings) if embeddings else np.array([])
    
    def save_weights(self, path: str):
        """Save model weights."""
        self.model.save_weights(path)
        print(f"✅ Weights saved: {path}")
    
    def load_weights(self, path: str):
        """Load model weights."""
        self.model.load_weights(path)
        print(f"✅ Weights loaded: {path}")


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(embedding1, embedding2))


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


def encode_dataset(encoder: CNNDogNoseEncoder, dataset: Dict[str, List[Path]], dataset_dir: Path = None) -> Dict[str, Dict]:
    """Encode entire dataset using CNN."""
    encoded_dataset = {}
    
    print(f"\n🔄 Encoding {len(dataset)} dogs using CNN...")
    
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
                'num_images': len(embeddings),
                'model': 'CNN-MobileNetV2'
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
    print(f"   Model: CNN-MobileNetV2 Transfer Learning")


def load_from_json(json_path: Path) -> Dict:
    """Load encoded dataset from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for dog_id in data:
        data[dog_id]['embeddings'] = np.array(data[dog_id]['embeddings'])
        data[dog_id]['avg_embedding'] = np.array(data[dog_id]['avg_embedding'])
    
    return data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CNN Dog Nose Encoder")
    parser.add_argument('--encode', type=str, metavar='DIR', help='Encode dataset directory')
    parser.add_argument('--json', type=str, default='encoded_noses_cnn.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    if args.encode:
        dataset_dir = Path(args.encode)
        
        if not dataset_dir.exists():
            print(f"❌ Dataset directory not found: {dataset_dir}")
            exit(1)
        
        print(f"\n📂 Loading dataset from: {dataset_dir}")
        dataset = load_dataset(dataset_dir)
        
        if not dataset:
            print("❌ No dogs found in dataset!")
            exit(1)
        
        print(f"✅ Found {len(dataset)} dogs")
        
        encoder = CNNDogNoseEncoder()
        encoded_dataset = encode_dataset(encoder, dataset, dataset_dir)
        save_to_json(encoded_dataset, Path(args.json))
