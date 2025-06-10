import os
import numpy as np
from PIL import Image

def infer_dataset_stats(images_dir, vocab_file=None, verbose=True, max_inspect=5):
    max_w, max_h = 0, 0
    n_channels_set = set()
    unexpected_images = []

    print(f"📂 Checking images in: {images_dir}")
    all_fnames = sorted(os.listdir(images_dir))
    for fname in all_fnames:
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue

        img_path = os.path.join(images_dir, fname)
        try:
            img = Image.open(img_path)
            w, h = img.size
            max_w = max(max_w, w)
            max_h = max(max_h, h)

            bands = img.getbands()  # e.g., ('R', 'G', 'B')
            n_ch = len(bands)
            n_channels_set.add(n_ch)

            if verbose and n_ch > 3 and len(unexpected_images) < max_inspect:
                unexpected_images.append((fname, bands))

            img.close()
        except Exception as e:
            print(f"⚠️ Error reading {img_path}: {e}")

    # Try to infer number of classes
    o_classes = None
    if vocab_file:
        if vocab_file.endswith(".npy"):
            vocab = np.load(vocab_file, allow_pickle=True).item()
            o_classes = len(vocab) + 1
        elif vocab_file.endswith(".txt"):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab = [line.strip() for line in f if line.strip()]
                o_classes = len(vocab)

    print("✅ Max width:", max_w)
    print("✅ Max height:", max_h)
    print("✅ Detected image channel counts:", sorted(n_channels_set))
    if o_classes is not None:
        print("✅ Vocabulary size (o_classes):", o_classes)

    if unexpected_images:
        print("⚠️ Found images with >3 channels:")
        for fname, bands in unexpected_images:
            print(f"  - {fname} has bands {bands}")

    return {
        "n_channels_set": sorted(n_channels_set),
        "max_w": max_w,
        "max_h": max_h,
        "o_classes": o_classes,
        "unexpected": unexpected_images,
    }

# Example usage:
if __name__ == "__main__":
    dataset = "Solesmes"
    stats = infer_dataset_stats(
        f"data/{dataset}/Images",
        f"data/{dataset}/vocab/{dataset}w2i.npy"
    )
    print("\n🔍 Final Stats:", stats)
