"""
Turnkey Downloader for LOL-v2 (Real & Synthetic) and ExDark Datasets.

Downloads and sets up:
1. LOL-v2-Real (689 train pairs, 100 test pairs) from Hugging Face (okhater/lolv2-real)
2. LOL-v2-Synthetic (900 train pairs, 100 test pairs) from Hugging Face (okhater/lolv2-synthetic)
3. ExDark (7,363 images) from official GitHub repository

Works on Kaggle, Colab, Linux, and Windows.
"""

import argparse
import os
import sys
import shutil
from pathlib import Path


def download_hf_dataset(repo_id: str, dest_dir: Path, dataset_type: str = "real"):
    """
    Downloads LOL-v2 subset from Hugging Face mirror and saves as PNGs in standard folders:
    Train/Low, Train/Normal, Test/Low, Test/Normal.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Download] Fetching '{repo_id}' from Hugging Face into {dest_dir}...")

    try:
        # Method 1: Using huggingface_hub snapshot_download if available
        from huggingface_hub import snapshot_download
        cache_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
        print(f"[Download] Snapshot downloaded to cache: {cache_dir}")

        # Check if cache contains parquet or image files
        cache_path = Path(cache_dir)
        has_png = any(cache_path.rglob("*.png"))
        if has_png:
            # Copy or symlink directly
            for p in cache_path.rglob("*.png"):
                rel = p.relative_to(cache_path)
                target = dest_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(p, target)
            print(f"[Done] Copied image files to {dest_dir}")
            return True
    except ImportError:
        pass

    try:
        # Method 2: Using Hugging Face datasets library
        print("[Download] Loading via 'datasets' library...")
        import datasets
        ds = datasets.load_dataset(repo_id)

        for split_name in ds.keys():
            # Standardize split name (train -> Train, test -> Test)
            folder_split = "Train" if "train" in split_name.lower() else "Test"
            low_dir = dest_dir / folder_split / "Low"
            normal_dir = dest_dir / folder_split / "Normal"
            low_dir.mkdir(parents=True, exist_ok=True)
            normal_dir.mkdir(parents=True, exist_ok=True)

            split_data = ds[split_name]
            print(f"  Exporting {len(split_data)} samples for split '{folder_split}'...")

            for idx, item in enumerate(split_data):
                # Check column names: commonly ('low', 'normal') or ('low_light', 'high_light') or ('image', 'label')
                low_img = None
                normal_img = None
                for k in item.keys():
                    k_low = k.lower()
                    if "low" in k_low:
                        low_img = item[k]
                    elif "norm" in k_low or "high" in k_low:
                        normal_img = item[k]

                if low_img is None or normal_img is None:
                    # Generic fallback by index
                    vals = list(item.values())
                    if len(vals) >= 2 and hasattr(vals[0], "save") and hasattr(vals[1], "save"):
                        low_img, normal_img = vals[0], vals[1]

                fname = f"{idx+1:05d}.png"
                if low_img is not None and hasattr(low_img, "save"):
                    low_img.save(low_dir / fname)
                if normal_img is not None and hasattr(normal_img, "save"):
                    normal_img.save(normal_dir / fname)

        print(f"[Done] Exported {repo_id} to {dest_dir}")
        return True

    except Exception as e:
        print(f"[Warning] Hugging Face download method encountered: {e}")

    # Method 3: Fallback via git clone from HF
    hf_git_url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"[Download] Falling back to git clone: {hf_git_url}")
    ret = os.system(f"git clone --depth 1 {hf_git_url} {dest_dir}/_hf_raw")
    if ret == 0:
        print(f"[Done] Cloned raw repo to {dest_dir}/_hf_raw")
        return True

    return False


def download_exdark(dest_dir: Path):
    """Downloads ExDark dataset from official GitHub repository."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Download] Fetching ExDark dataset into {dest_dir}...")
    zip_url = "https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/archive/refs/heads/master.zip"
    zip_path = dest_dir / "exdark_master.zip"

    # Download zip via urllib or git clone
    import urllib.request
    try:
        print(f"  Downloading from {zip_url} ...")
        urllib.request.urlretrieve(zip_url, zip_path)
        print("  Extracting archive...")
        shutil.unpack_archive(zip_path, dest_dir)
        if zip_path.exists():
            zip_path.unlink()
        print(f"[Done] ExDark ready at {dest_dir}")
        return True
    except Exception as e:
        print(f"[Warning] Direct ExDark zip download failed: {e}")
        ret = os.system(f"git clone --depth 1 https://github.com/cs-chan/Exclusively-Dark-Image-Dataset.git {dest_dir}")
        return (ret == 0)


def main():
    parser = argparse.ArgumentParser(description="Download LOL-v2 Real, LOL-v2 Synthetic, and ExDark datasets.")
    parser.add_argument("--output_root", type=str, default="data",
                        help="Root directory where dataset folders will be saved (e.g. 'data' or '/kaggle/working/datasets').")
    parser.add_argument("--datasets", nargs="+", default=["lol_v2_real", "lol_v2_synthetic", "exdark"],
                        choices=["lol_v2_real", "lol_v2_synthetic", "exdark", "all"],
                        help="Datasets to download.")
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    ds_list = ["lol_v2_real", "lol_v2_synthetic", "exdark"] if "all" in args.datasets else args.datasets

    print("==================================================")
    print("      MULTI-DATASET AUTOMATED DOWNLOADER")
    print(f"  Target directory: {root.resolve()}")
    print(f"  Requested: {ds_list}")
    print("==================================================")

    # 1. LOL-v2 Real
    if "lol_v2_real" in ds_list:
        real_dest = root / "LOL-v2-Real" / "Real_captured"
        download_hf_dataset("okhater/lolv2-real", real_dest, dataset_type="real")

    # 2. LOL-v2 Synthetic
    if "lol_v2_synthetic" in ds_list:
        syn_dest = root / "LOL-v2-Synthetic" / "Synthetic"
        download_hf_dataset("okhater/lolv2-synthetic", syn_dest, dataset_type="synthetic")

    # 3. ExDark
    if "exdark" in ds_list:
        exdark_dest = root / "ExDark"
        download_exdark(exdark_dest)

    print("\n==================================================")
    print("              DOWNLOAD COMPLETE")
    print("==================================================")
    print("Verify with:")
    print("  python -c \"from datasets.multi_dataset import MultiDatasetManager; print(MultiDatasetManager.auto_discover_all())\"")


if __name__ == "__main__":
    main()
