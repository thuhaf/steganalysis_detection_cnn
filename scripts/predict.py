#!/usr/bin/env python3
"""
Prediction Script for Steganalysis

Make predictions on new images or audio files using trained models.

Usage:
    # Single file
    python scripts/predict.py --model checkpoints/best_model.pth --input image.jpg

    # Directory
    python scripts/predict.py --model checkpoints/best_model.pth --input images/ --batch

    # With custom threshold
    python scripts/predict.py --model checkpoints/best_model.pth --input images/ --threshold 0.6
"""



import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

# THÊM 2 DÒNG NÀY NGAY SAU CÁC IMPORT CƠ BẢN, TRƯỚC MỌI THỨ KHÁC!
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# BÂY GIỜ IMPORT BÌNH THƯỜNG, KHÔNG CẦN TRY-EXCEPT NỮA!
from src.inference.pipeline import StegAnalysisPipeline, PipelineConfig
from src.inference.predictor import StegPredictor, load_model_for_inference
from src.inference.batch_predictor import BatchPredictor
from src.models.model_registry import ModelRegistry
from src.data.augmentation import get_augmentation
from src.data.preprocessing import ImagePreprocessor, AudioPreprocessor
from src.utils.helpers import get_device, print_header, ensure_dir, find_files_with_extension




def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Make predictions with trained steganalysis model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50_steg",
        help="Model architecture name"
    )

    # Input
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input file or directory"
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=['image', 'audio'],
        default='image',
        help="Data modality"
    )
    parser.add_argument(
        "--recursive",
        action='store_true',
        default=True,
        help="Search directories recursively"
    )

    # Prediction options
    parser.add_argument(
        "--batch",
        action='store_true',
        help="Use batch prediction (faster for multiple files)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for batch prediction"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold"
    )
    parser.add_argument(
        "--show-confidence",
        action='store_true',
        default=True,
        help="Show confidence scores"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON/CSV)"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=['json', 'csv', 'txt'],
        default='json',
        help="Output format"
    )
    parser.add_argument(
        "--save-report",
        action='store_true',
        help="Generate and save detailed report"
    )
    parser.add_argument(
        "--filter-stego",
        type=str,
        help="Copy detected stego files to this directory"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence for filtering stego files"
    )

    # Display options
    parser.add_argument(
        "--verbose", "-v",
        action='store_true',
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action='store_true',
        help="Minimal output"
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--no-amp",
        action='store_true',
        help="Disable automatic mixed precision"
    )

    return parser.parse_args()


def predict_single_file(args, pipeline):
    """Predict a single file"""
    print_header("SINGLE FILE PREDICTION")

    print(f"Input: {args.input}")

    result = pipeline.predict_single(
        args.input,
        return_details=True
    )

    # Display result
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"File: {result['file_path']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")

    if 'class_probabilities' in result:
        print("\nClass Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")

    print(f"\nInference time: {result.get('inference_time', 0):.4f}s")
    print("="*60)

    return [result]


def predict_directory(args, pipeline):
    """Predict all files in a directory"""
    print_header("BATCH PREDICTION")

    input_path = Path(args.input)
    print(f"Input directory: {input_path}")

    # Find files
    if args.modality == 'image':
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    else:
        extensions = ['.wav', '.mp3', '.flac', '.ogg']

    files = find_files_with_extension(
        input_path,
        extensions,
        recursive=args.recursive
    )

    if len(files) == 0:
        print(f"❌ No {args.modality} files found in {input_path}")
        return []

    print(f"Found {len(files)} files")

    # Predict
    file_paths = [str(f) for f in files]

    if args.batch:
        results = pipeline.predict_batch(file_paths, show_progress=True)
    else:
        results = []
        for file_path in file_paths:
            result = pipeline.predict_single(file_path, return_details=True)
            results.append(result)
            if args.verbose:
                print(f"  {Path(file_path).name}: {result['prediction']} "
                      f"({result['confidence']:.4f})")

    return results


def display_summary(results, args):
    """Display prediction summary"""
    if args.quiet:
        return

    print_header("PREDICTION SUMMARY")

    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]
    errors = [r for r in results if 'error' in r]

    if len(valid_results) == 0:
        print("❌ No valid predictions")
        return

    # Count predictions
    stego_count = sum(1 for r in valid_results if r['prediction'] == 'Stego')
    cover_count = len(valid_results) - stego_count

    # Calculate average confidence
    avg_confidence = sum(r['confidence'] for r in valid_results) / len(valid_results)

    # Display
    print(f"Total files processed: {len(results)}")
    print(f"  Successful: {len(valid_results)}")
    print(f"  Errors: {len(errors)}")
    print()
    print(f"Predictions:")
    print(f"  Stego detected: {stego_count} ({stego_count/len(valid_results)*100:.1f}%)")
    print(f"  Cover images:   {cover_count} ({cover_count/len(valid_results)*100:.1f}%)")
    print()
    print(f"Average confidence: {avg_confidence:.4f}")

    # High-confidence stego files
    high_conf_stego = [
        r for r in valid_results
        if r['prediction'] == 'Stego' and r['confidence'] >= args.min_confidence
    ]

    if high_conf_stego:
        print(f"\nHigh-confidence stego files (>= {args.min_confidence}):")
        for r in high_conf_stego[:10]:  # Show top 10
            print(f"  {Path(r['file_path']).name}: {r['confidence']:.4f}")
        if len(high_conf_stego) > 10:
            print(f"  ... and {len(high_conf_stego) - 10} more")


def save_results(results, args):
    """Save prediction results to file"""
    if not args.output:
        return

    print_header("SAVING RESULTS")

    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    if args.output_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {output_path}")

    elif args.output_format == 'csv':
        import pandas as pd

        # Flatten results
        flat_results = []
        for r in results:
            if 'error' in r:
                flat = {
                    'file_path': r['file_path'],
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'error': r['error']
                }
            else:
                flat = {
                    'file_path': r['file_path'],
                    'prediction': r['prediction'],
                    'confidence': r['confidence']
                }
                if 'class_probabilities' in r:
                    for cls, prob in r['class_probabilities'].items():
                        flat[f'prob_{cls}'] = prob
            flat_results.append(flat)

        df = pd.DataFrame(flat_results)
        df.to_csv(output_path, index=False)
        print(f"✅ Results saved to {output_path}")

    elif args.output_format == 'txt':
        with open(output_path, 'w') as f:
            f.write("STEGANALYSIS PREDICTION RESULTS\n")
            f.write("="*60 + "\n\n")

            for r in results:
                if 'error' in r:
                    f.write(f"{r['file_path']}: ERROR - {r['error']}\n")
                else:
                    f.write(f"{r['file_path']}\n")
                    f.write(f"  Prediction: {r['prediction']}\n")
                    f.write(f"  Confidence: {r['confidence']:.4f}\n")
                    f.write("\n")

        print(f"✅ Results saved to {output_path}")


def filter_stego_files(results, args):
    """Copy high-confidence stego files to separate directory"""
    if not args.filter_stego:
        return

    print_header("FILTERING STEGO FILES")

    filter_dir = Path(args.filter_stego)
    ensure_dir(filter_dir)

    # Find high-confidence stego files
    stego_files = [
        r for r in results
        if 'error' not in r
        and r['prediction'] == 'Stego'
        and r['confidence'] >= args.min_confidence
    ]

    print(f"Found {len(stego_files)} high-confidence stego files (>= {args.min_confidence})")

    if len(stego_files) == 0:
        print("No files to copy")
        return

    # Copy files
    import shutil
    copied = 0

    for r in stego_files:
        src = Path(r['file_path'])
        dst = filter_dir / src.name

        try:
            shutil.copy2(src, dst)
            copied += 1
            if args.verbose:
                print(f"  Copied: {src.name}")
        except Exception as e:
            print(f"  ❌ Error copying {src.name}: {e}")

    print(f"\n✅ Copied {copied} files to {filter_dir}")


def generate_report(results, args, output_path):
    """Generate detailed text report"""
    report_lines = [
        "="*70,
        "STEGANALYSIS PREDICTION REPORT",
        "="*70,
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {args.model}",
        f"Input: {args.input}",
        f"Threshold: {args.threshold}",
        "\n" + "-"*70,
        "SUMMARY",
        "-"*70,
    ]

    # Statistics
    valid_results = [r for r in results if 'error' not in r]
    stego_count = sum(1 for r in valid_results if r['prediction'] == 'Stego')
    cover_count = len(valid_results) - stego_count

    report_lines.extend([
        f"Total files: {len(results)}",
        f"Stego detected: {stego_count} ({stego_count/len(valid_results)*100:.1f}%)",
        f"Cover images: {cover_count} ({cover_count/len(valid_results)*100:.1f}%)",
        "\n" + "-"*70,
        "STEGO FILES (High Confidence)",
        "-"*70,
    ])

    # List stego files
    stego_files = [
        r for r in valid_results
        if r['prediction'] == 'Stego'
    ]
    stego_files.sort(key=lambda x: x['confidence'], reverse=True)

    for r in stego_files:
        report_lines.append(
            f"{Path(r['file_path']).name:<50} Confidence: {r['confidence']:.4f}"
        )

    report_lines.append("\n" + "="*70)

    # Save report
    report_file = output_path.parent / "prediction_report.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✅ Report saved to {report_file}")


def main():
    """Main prediction pipeline"""
    args = parse_args()

    if not args.quiet:
        print_header("STEGANALYSIS PREDICTION")

    # Setup
    device = get_device(args.device)

    # Create pipeline
    if not args.quiet:
        print(f"Loading model: {args.model}")

    pipeline_config = PipelineConfig(
        model_path=args.model,
        model_name=args.model_name,
        modality=args.modality,
        device=str(device),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        use_amp=not args.no_amp
    )

    pipeline = StegAnalysisPipeline(pipeline_config)

    # Make predictions
    input_path = Path(args.input)

    if input_path.is_file():
        results = predict_single_file(args, pipeline)
    elif input_path.is_dir():
        results = predict_directory(args, pipeline)
    else:
        print(f"❌ Error: {args.input} not found")
        return 1

    # Display summary
    display_summary(results, args)

    # Save results
    if args.output:
        save_results(results, args)

    # Filter stego files
    if args.filter_stego:
        filter_stego_files(results, args)

    # Generate report
    if args.save_report and args.output:
        generate_report(results, args, Path(args.output))

    if not args.quiet:
        print_header("PREDICTION COMPLETE")
        print(f"✅ Processed {len(results)} files successfully!")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Prediction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
