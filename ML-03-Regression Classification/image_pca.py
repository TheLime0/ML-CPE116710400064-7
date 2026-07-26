"""
image_pca.py
------------
LAB objective: "ประยุกต์ใช้เทคนิค Principal Component Analysis เพื่อลดจำนวนคุณลักษณะ
(Features) และเพิ่มประสิทธิภาพของการเรียนรู้" (โจทย์ข้อ 2)

This module demonstrates PCA-based feature reduction directly on the raw pixel
data of the dog photo dataset ("Eigen-dog" analysis, same idea as classic
Eigenfaces). It is run as an UNSUPERVISED exploratory step because the photo
dataset has no age/gender ground-truth labels attached to it (see project
README / lab report for why age & gender modelling instead uses the labelled
dogs_dataset.csv table).

Outputs (written to ../outputs/):
  - explained_variance.png   : cumulative explained variance vs #components
  - eigen_dogs.png           : top principal components visualised as images
  - reconstruction.png       : original vs PCA-reconstructed images at
                                different numbers of components
  - image_pca_summary.json   : numeric summary used in the report
"""
import os
import json
import random
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

IMG_ROOT = "/home/claude/dataset/train"
OUT_DIR = "/home/claude/lab_project/outputs"
N_SAMPLES = 800          # subset of images for speed
IMG_SIZE = (64, 64)      # downsize for tractable PCA on raw pixels

os.makedirs(OUT_DIR, exist_ok=True)


def collect_image_paths(root, n_samples):
    all_paths = []
    for folder in os.listdir(root):
        fdir = os.path.join(root, folder)
        if not os.path.isdir(fdir):
            continue
        for fname in os.listdir(fdir):
            if fname.lower().endswith(".jpg"):
                all_paths.append(os.path.join(fdir, fname))
    random.shuffle(all_paths)
    return all_paths[:n_samples]


def load_images_as_matrix(paths, size):
    vectors = []
    for p in paths:
        im = Image.open(p).convert("L").resize(size)  # grayscale, resized
        vectors.append(np.asarray(im, dtype=np.float32).flatten())
    return np.vstack(vectors)


def main():
    paths = collect_image_paths(IMG_ROOT, N_SAMPLES)
    print(f"Loaded {len(paths)} image paths")

    X = load_images_as_matrix(paths, IMG_SIZE)
    print("Raw pixel matrix shape:", X.shape)

    mean_face = X.mean(axis=0)
    X_centered = X - mean_face

    # Fit PCA keeping enough components to explain 95% variance
    pca_full = PCA(n_components=min(200, X.shape[0] - 1), random_state=42)
    pca_full.fit(X_centered)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = int(np.argmax(cum_var >= 0.95) + 1)

    # --- Plot 1: explained variance curve ---
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(cum_var) + 1), cum_var, color="#2b6cb0")
    plt.axhline(0.95, color="gray", linestyle="--", linewidth=1)
    plt.axvline(n_95, color="gray", linestyle="--", linewidth=1)
    plt.title(f"PCA Cumulative Explained Variance (95% at {n_95} components)")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "explained_variance.png"), dpi=130)
    plt.close()

    # --- Plot 2: top eigen-dogs (principal component directions as images) ---
    n_show = 10
    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    for i, ax in enumerate(axes.flat):
        comp = pca_full.components_[i].reshape(IMG_SIZE)
        ax.imshow(comp, cmap="gray")
        ax.set_title(f"PC{i+1}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Top 10 Principal Components ('Eigen-dogs')")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eigen_dogs.png"), dpi=130)
    plt.close()

    # --- Plot 3: reconstruction quality at different k ---
    sample_idx = [0, 1, 2]
    ks = [5, 20, 50, n_95]
    fig, axes = plt.subplots(len(sample_idx), len(ks) + 1, figsize=(12, 7))
    for row, idx in enumerate(sample_idx):
        orig = X[idx].reshape(IMG_SIZE)
        axes[row, 0].imshow(orig, cmap="gray")
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis("off")
        for col, k in enumerate(ks):
            pca_k = PCA(n_components=k, random_state=42)
            proj = pca_k.fit_transform(X_centered)
            recon = pca_k.inverse_transform(proj)[idx] + mean_face
            axes[row, col + 1].imshow(recon.reshape(IMG_SIZE), cmap="gray")
            axes[row, col + 1].set_title(f"k={k}" if row == 0 else "")
            axes[row, col + 1].axis("off")
    fig.suptitle("Original vs PCA Reconstruction at Increasing Component Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "reconstruction.png"), dpi=130)
    plt.close()

    summary = {
        "n_images_used": len(paths),
        "image_size": IMG_SIZE,
        "raw_feature_dim": int(X.shape[1]),
        "components_for_95pct_variance": n_95,
        "compression_ratio": round(X.shape[1] / n_95, 2),
    }
    with open(os.path.join(OUT_DIR, "image_pca_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
