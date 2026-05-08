import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

IMG_SIZE    = (299, 299)
BATCH_SIZE  = 32
FEATURE_DIM = 2048

def build_inception_encoder() -> keras.Model:
    base = InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False
    print(f"[InceptionV3] Loaded. Output shape: {base.output_shape}")
    return base

def load_image_inception(file_path: str) -> np.ndarray:
    img = Image.open(file_path).convert("RGB")
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return arr


def load_batch_inception(file_paths: list) -> np.ndarray:
    return np.stack([load_image_inception(fp) for fp in file_paths], axis=0)

def get_flickr8k_image_paths(
    images_dir: str,
    split_file: str = None,
) -> list:
    if split_file and os.path.exists(split_file):
        with open(split_file) as f:
            filenames = [line.strip() for line in f if line.strip()]
        paths = [(fn, os.path.join(images_dir, fn)) for fn in filenames
                 if os.path.exists(os.path.join(images_dir, fn))]
    else:
        filenames = sorted([
            fn for fn in os.listdir(images_dir)
            if fn.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        paths = [(fn, os.path.join(images_dir, fn)) for fn in filenames]

    print(f"[Flickr8k] Found {len(paths)} images from {images_dir}")
    return paths

def extract_features_flickr8k(
    encoder: keras.Model,
    image_paths: list,
    output_npy: str,
    batch_size: int = BATCH_SIZE,
    force: bool = False,
) -> dict:
    if os.path.exists(output_npy) and not force:
        print(f"[FeatureExtraction] Cache found. Loading from {output_npy}")
        features = np.load(output_npy, allow_pickle=True).item()
        print(f"[FeatureExtraction] Loaded {len(features)} features.")
        return features

    os.makedirs(os.path.dirname(output_npy) or ".", exist_ok=True)

    features = {}
    n        = len(image_paths)

    for start in range(0, n, batch_size):
        batch_info  = image_paths[start:start + batch_size]
        batch_fnames = [info[0] for info in batch_info]
        batch_fpaths = [info[1] for info in batch_info]

        # Load & preprocess batch
        batch_imgs = load_batch_inception(batch_fpaths)

        # Forward pass (frozen InceptionV3)
        batch_feats = encoder.predict(batch_imgs, verbose=0)

        for fname, feat in zip(batch_fnames, batch_feats):
            features[fname] = feat.astype(np.float32)

        end = min(start + batch_size, n)
        print(f"  [{end:>5}/{n}] Extracted features...", end="\r")

    print(f"\n[FeatureExtraction] Done. Saving to {output_npy}")
    np.save(output_npy, features)
    print(f"[FeatureExtraction] Saved {len(features)} feature vectors (dim={FEATURE_DIM}).")
    return features

def verify_features(features: dict, n_samples: int = 3):
    print("\n[Verify] Feature stats:")
    print(f"  Total images : {len(features)}")
    sample_keys = list(features.keys())[:n_samples]
    for k in sample_keys:
        v = features[k]
        print(f"  {k}: shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")

def run_feature_extraction_pipeline(
    flickr8k_images_dir: str,
    output_dir: str = "features",
    split_files: dict = None,
    batch_size: int = BATCH_SIZE,
    force: bool = False,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    print("[Pipeline] Building InceptionV3 encoder...")
    encoder = build_inception_encoder()

    if split_files is None:
        # Mode semua gambar sekaligus
        all_paths   = get_flickr8k_image_paths(flickr8k_images_dir)
        output_path = os.path.join(output_dir, "flickr8k_features.npy")
        features    = extract_features_flickr8k(
            encoder, all_paths, output_path,
            batch_size=batch_size, force=force,
        )
        verify_features(features)
        return features

    else:
        # Mode per split
        all_features = {}
        for split_name, split_txt in split_files.items():
            print(f"\n[Pipeline] Processing split: {split_name}")
            paths       = get_flickr8k_image_paths(flickr8k_images_dir, split_txt)
            output_path = os.path.join(output_dir, f"{split_name}.npy")
            features    = extract_features_flickr8k(
                encoder, paths, output_path,
                batch_size=batch_size, force=force,
            )
            verify_features(features)
            all_features[split_name] = features

        return all_features

def load_features(npy_path: str) -> dict:
    features = np.load(npy_path, allow_pickle=True).item()
    print(f"[LoadFeatures] Loaded {len(features)} vectors from {npy_path}")
    return features


def features_to_matrix(
    features: dict,
    image_filenames: list,
) -> np.ndarray:
    matrix = np.stack([features[fn] for fn in image_filenames], axis=0)
    return matrix.astype(np.float32)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flickr8k Feature Extraction")
    parser.add_argument("--images_dir",  required=True,
                        help="Folder berisi semua gambar Flickr8k (.jpg)")
    parser.add_argument("--output_dir",  default="features",
                        help="Folder output .npy (default: features/)")
    parser.add_argument("--train_txt",   default=None,
                        help="Path Flickr_8k.trainImages.txt (opsional)")
    parser.add_argument("--val_txt",     default=None,
                        help="Path Flickr_8k.devImages.txt (opsional)")
    parser.add_argument("--test_txt",    default=None,
                        help="Path Flickr_8k.testImages.txt (opsional)")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--force",       action="store_true",
                        help="Paksa re-ekstraksi meski cache ada")
    args = parser.parse_args()

    split_files = None
    if args.train_txt:
        split_files = {
            "train": args.train_txt,
            "val":   args.val_txt,
            "test":  args.test_txt,
        }

    run_feature_extraction_pipeline(
        flickr8k_images_dir=args.images_dir,
        output_dir=args.output_dir,
        split_files=split_files,
        batch_size=args.batch_size,
        force=args.force,
    )