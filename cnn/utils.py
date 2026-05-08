import os
import numpy as np
from PIL import Image

def load_image(file_path: str, target_size: tuple = (150, 150)) -> np.ndarray:
    # Ukurannya diset 150x150
    img = Image.open(file_path).convert("RGB")
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def load_batch(file_paths: list, target_size: tuple = (150, 150)) -> np.ndarray:
    batch = [load_image(fp, target_size) for fp in file_paths]
    return np.stack(batch, axis=0)

def extract_and_save_features(
    image_paths: list,
    encoder,
    output_path: str,
    target_size: tuple = (150, 150),
    batch_size: int = 32,
) -> np.ndarray:
    # Ekstraksi feature vector dari semua gambar menggunakan Keras CNN encoder
    # yang sudah di-freeze, lalu simpan hasilnya ke disk (.npy).

    if os.path.exists(output_path): # Jika file output sudah ada, langsung load dari disk
        print(f"[FeatureExtractor] Loading cached features from {output_path}")
        return np.load(output_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_features = []
    n = len(image_paths)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_paths = image_paths[start:end]
        batch_imgs  = load_batch(batch_paths, target_size)

        feats = encoder.predict(batch_imgs, verbose=0)
        if feats.ndim > 2:
            feats = feats.reshape(feats.shape[0], -1) # Flatten 

        all_features.append(feats)
        print(f"[FeatureExtractor] Processed {end}/{n} images", end="\r")

    features = np.concatenate(all_features, axis=0)
    np.save(output_path, features)
    print(f"\n[FeatureExtractor] Saved features to {output_path}  shape={features.shape}")
    return features

def build_dataset_from_directory(
    root_dir: str,
    target_size: tuple = (150, 150),
    splits: tuple = ("train", "val", "test"),
) -> dict:
    class_names = sorted([
        d for d in os.listdir(os.path.join(root_dir, splits[0]))
        if os.path.isdir(os.path.join(root_dir, splits[0], d))
    ])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    result = {}
    for split in splits:
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            print(f"[Warning] Split directory not found: {split_dir}")
            continue
        paths, labels = [], []
        for cls in class_names:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(cls_dir, fname))
                    labels.append(class_to_idx[cls])
        result[split] = {
            "paths": paths,
            "labels": np.array(labels, dtype=np.int32),
            "class_names": class_names,
        }
        print(f"[Dataset] {split}: {len(paths)} images, {len(class_names)} classes")
    return result