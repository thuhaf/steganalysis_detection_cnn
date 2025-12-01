import sys
import argparse
from pathlib import Path
import shutil
from typing import List, Tuple
import random

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils.helpers import set_seed, find_files_with_extension


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Prepare dataset for training')

    parser.add_argument('--cover-dir', type=str, required=True,
                        help='Directory containing cover (clean) images')
    parser.add_argument('--stego-dir', type=str, required=True,
                        help='Directory containing stego (hidden data) images')

    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Output directory for organized dataset')

    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test set ratio')

    parser.add_argument('--extensions', type=str, nargs='+',
                        default=['.jpg', '.jpeg', '.png', '.bmp'],
                        help='Valid file extensions')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    parser.add_argument('--symlink', action='store_true',
                        help='Create symlinks instead of copying files')

    return parser.parse_args()


def split_files(
    files: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split files into train/val/test"""

    # Verify ratios sum to 1
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1, got {total}"

    # Shuffle
    random.seed(seed)
    files = files.copy()
    random.shuffle(files)

    # Calculate split points
    n = len(files)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    return train_files, val_files, test_files


def copy_or_link_files(
    files: List[Path],
    dest_dir: Path,
    use_symlink: bool = False
):
    """Copy or symlink files to destination"""

    dest_dir.mkdir(parents=True, exist_ok=True)

    for src_file in files:
        dest_file = dest_dir / src_file.name

        if use_symlink:
            # Create symlink
            if dest_file.exists():
                dest_file.unlink()
            dest_file.symlink_to(src_file.absolute())
        else:
            # Copy file
            shutil.copy2(src_file, dest_file)


def main():
    """Main dataset preparation function"""

    args = parse_args()

    # Set seed
    set_seed(args.seed)

    print("\n" + "="*60)
    print("DATASET PREPARATION")
    print("="*60 + "\n")

    # Find cover files
    cover_dir = Path(args.cover_dir)
    print(f"Searching for cover images in: {cover_dir}")
    cover_files = find_files_with_extension(
        cover_dir,
        args.extensions,
        recursive=True
    )
    print(f"Found {len(cover_files)} cover images")

    # Find stego files
    stego_dir = Path(args.stego_dir)
    print(f"Searching for stego images in: {stego_dir}")
    stego_files = find_files_with_extension(
        stego_dir,
        args.extensions,
        recursive=True
    )
    print(f"Found {len(stego_files)} stego images")

    if len(cover_files) == 0 or len(stego_files) == 0:
        print("\nError: No files found! Check your directories and extensions.")
        return

    # Split datasets
    print(f"\nSplitting datasets with ratios:")
    print(f"  Train: {args.train_ratio*100:.1f}%")
    print(f"  Val:   {args.val_ratio*100:.1f}%")
    print(f"  Test:  {args.test_ratio*100:.1f}%")

    cover_train, cover_val, cover_test = split_files(
        cover_files,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

    stego_train, stego_val, stego_test = split_files(
        stego_files,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

    print(f"\nCover splits:")
    print(f"  Train: {len(cover_train)}")
    print(f"  Val:   {len(cover_val)}")
    print(f"  Test:  {len(cover_test)}")

    print(f"\nStego splits:")
    print(f"  Train: {len(stego_train)}")
    print(f"  Val:   {len(stego_val)}")
    print(f"  Test:  {len(stego_test)}")

    # Create output directories
    output_dir = Path(args.output_dir)

    action = "Symlinking" if args.symlink else "Copying"
    print(f"\n{action} files to {output_dir}...")

    # Copy/link training files
    print("  Processing training set...")
    copy_or_link_files(
        cover_train,
        output_dir / 'train' / 'cover',
        args.symlink
    )
    copy_or_link_files(
        stego_train,
        output_dir / 'train' / 'stego',
        args.symlink
    )

    # Copy/link validation files
    print("  Processing validation set...")
    copy_or_link_files(
        cover_val,
        output_dir / 'val' / 'cover',
        args.symlink
    )
    copy_or_link_files(
        stego_val,
        output_dir / 'val' / 'stego',
        args.symlink
    )

    # Copy/link test files
    print("  Processing test set...")
    copy_or_link_files(
        cover_test,
        output_dir / 'test' / 'cover',
        args.symlink
    )
    copy_or_link_files(
        stego_test,
        output_dir / 'test' / 'stego',
        args.symlink
    )

    # Create info file
    info = {
        'cover_dir': str(cover_dir),
        'stego_dir': str(stego_dir),
        'total_cover': len(cover_files),
        'total_stego': len(stego_files),
        'train_cover': len(cover_train),
        'train_stego': len(stego_train),
        'val_cover': len(cover_val),
        'val_stego': len(stego_val),
        'test_cover': len(cover_test),
        'test_stego': len(stego_test),
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'seed': args.seed,
    }

    import json
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETED")
    print("="*60)
    print(f"\nDataset organized in: {output_dir}")
    print("\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── cover/  ({len(cover_train)} images)")
    print(f"    │   └── stego/  ({len(stego_train)} images)")
    print(f"    ├── val/")
    print(f"    │   ├── cover/  ({len(cover_val)} images)")
    print(f"    │   └── stego/  ({len(stego_val)} images)")
    print(f"    └── test/")
    print(f"        ├── cover/  ({len(cover_test)} images)")
    print(f"        └── stego/  ({len(stego_test)} images)")
    print("\nYou can now train with:")
    print(f"  python scripts/train.py --cover-train {output_dir}/train/cover --stego-train {output_dir}/train/stego")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
