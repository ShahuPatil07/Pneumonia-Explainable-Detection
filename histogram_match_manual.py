"""
Histogram-match an entire dataset to a manually chosen reference image.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


# -----------------------------
# Argument Parsing
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Histogram-match dataset to a manually chosen reference image."
    )
    parser.add_argument(
        "image_root",
        type=Path,
        help="Directory containing ALL images to histogram-match."
    )
    parser.add_argument(
        "reference_image",
        type=Path,
        help="Path to the manually selected reference image."
    )
    parser.add_argument(
        "--pattern",
        default="*.png;*.jpg;*.jpeg",
        help="Semicolon-separated glob patterns."
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=256,
        help="Histogram bins."
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Where to save equalized images."
    )
    return parser.parse_args()


# -----------------------------
# Image Helpers
# -----------------------------
def iter_image_paths(root: Path, pattern_string: str) -> Iterable[Path]:
    for pattern in pattern_string.split(";"):
        yield from root.rglob(pattern)


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def save_gray(img: np.ndarray, out_path: Path):
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(out_path)


# -----------------------------
# Histogram Matching
# -----------------------------
def match_histogram(img: np.ndarray, ref: np.ndarray, bins=256) -> np.ndarray:
    src = img.ravel()
    tgt = ref.ravel()

    # Source hist
    s_hist, s_bins = np.histogram(src, bins=bins, range=(0, 1))
    s_cdf = np.cumsum(s_hist).astype(float) / (s_hist.sum() + 1e-12)

    # Reference hist
    r_hist, r_bins = np.histogram(tgt, bins=bins, range=(0, 1))
    r_cdf = np.cumsum(r_hist).astype(float) / (r_hist.sum() + 1e-12)

    s_centers = (s_bins[:-1] + s_bins[1:]) / 2
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2

    # Match via CDF inversion
    mapping = np.interp(s_cdf, r_cdf, r_centers)
    matched = np.interp(src, s_centers, mapping)

    return matched.reshape(img.shape)


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    # Resolve output directory
    if args.output_root is None:
        output_root = args.image_root / "equalized_manual"
    else:
        output_root = args.output_root

    # Load manually chosen reference
    reference = load_gray(args.reference_image)
    print(f"Using manually selected reference image:\n  {args.reference_image}")

    # Load each dataset image
    paths = sorted(iter_image_paths(args.image_root, args.pattern))
    print(f"Found {len(paths)} images to equalize.")

    for p in paths:
        img = load_gray(p)
        matched = match_histogram(img, reference, bins=args.bins)

        # Keep structure
        rel = p.relative_to(args.image_root)
        out_path = output_root / rel
        save_gray(matched, out_path)

    print(f"\nDone! Equalized images saved to:\n  {output_root}")


if __name__ == "__main__":
    main()


## Example Usage

# python histogram_match_manual.py `
# "C:\Users\Shahu Patil\Desktop\CS663 Project\data_binary_classification\train\NORMAL" `
# "C:\Users\Shahu Patil\Desktop\CS663 Project\data_binary_classification\train\NORMAL\IM-0226-0001.jpeg" `
# --pattern "*.png;*.jpg;*.jpeg" `
# --bins 256 `
# --output_root "C:\Users\Shahu Patil\Desktop\CS663 Project\data_binary_classification\train\NORMAL_EQ"

