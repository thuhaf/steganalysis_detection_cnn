import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import ImageStegDataset
from data.dataloader import StegDataLoaderFactory
from data.preprocessing import ImagePreprocessor

from models.model_registry import get_model
from training.validator import StegValidator
from training.metrics import MetricsCalculator

from utils.config import Config, load_config
from utils.helpers import set_seed, get_device
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_all_evaluation_metrics
)
import warnings
warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
warnings.filterwarnings("ignore", message=".*weights.*deprecated.*")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate steganalysis model')

    parser.add_argument('--model-path', type=str, required=True,
                        help='Đường dẫn đến checkpoint mô hình đã được huấn luyện')
    parser.add_argument('--config', type=str, default=None,
                        help='Đường dẫn đến tệp cấu hình')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Tên kiến trúc mô hình')

    # Data
    parser.add_argument('--cover-test', type=str, default='data/raw/images/cover',
                        help='Đường dẫn đến ảnh gốc (cover) thử nghiệm')
    parser.add_argument('--stego-test', type=str, default='data/raw/images/stego',
                        help='Đường dẫn đến ảnh ẩn mã (stego) thử nghiệm')

    # Evaluation options
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Kích thước lô (batch size) cho đánh giá')
    parser.add_argument('--save-dir', type=str, default='./results',
                        help='Thư mục để lưu trữ kết quả')
    parser.add_argument('--save-plots', action='store_true',
                        help='Lưu các biểu đồ đánh giá')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Thiết bị (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Số lượng tiến trình con để tải dữ liệu')

    return parser.parse_args()


def detect_model_architecture(checkpoint):
    """Detect model architecture from checkpoint"""
    # First check if model_name is stored in checkpoint
    if 'model_name' in checkpoint:
        return checkpoint['model_name']

    # Otherwise infer from state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Check fc layer size to determine ResNet variant
    if 'backbone.fc.1.weight' in state_dict:
        fc_in_features = state_dict['backbone.fc.1.weight'].shape[1]
        print(f"Đã phát hiện đầu vào lớp FC: {fc_in_features}")

        if fc_in_features == 512:
            return 'resnet18_steg'
        elif fc_in_features == 2048:
            return 'resnet50_steg'

    # Check for other model types
    if 'backbone.features.0.0.weight' in state_dict:
        return 'efficientnet_b0_steg'

    # Default fallback
    print("Cảnh báo: Không thể phát hiện kiến trúc mô hình, mặc định là resnet18_steg")
    return 'resnet18_steg'


def main():
    """Main evaluation function"""

    # Parse arguments
    args = parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = Config()

    # Set device
    device = get_device(args.device)

    # Set seed
    set_seed(config.seed)

    print("\n" + "="*60)
    print("ĐÁNH GIÁ MÔ HÌNH")
    print("="*60 + "\n")

    # Create test dataset
    print("Đang tải tập dữ liệu thử nghiệm...")
    test_transform = ImagePreprocessor.get_val_transforms(config.data.img_size)

    test_dataset = ImageStegDataset(
        cover_dir=args.cover_test,
        stego_dir=args.stego_test,
        transform=test_transform,
        split='test'
    )

    test_loader = StegDataLoaderFactory.create_test_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Load checkpoint
    print(f"\nĐang tải mô hình từ {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Determine model architecture
    if args.model_name:
        model_name = args.model_name
        print(f"Sử dụng mô hình được chỉ định: {model_name}")
    else:
        model_name = detect_model_architecture(checkpoint)
        print(f"Mô hình tự động phát hiện: {model_name}")

    # Create model with detected architecture
    model = get_model(
        model_name=model_name,
        num_classes=config.model.num_classes,
        pretrained=False
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print("Đã tải mô hình thành công!")

    # Create validator
    validator = StegValidator(
        model=model,
        val_loader=test_loader,
        device=str(device),
        use_amp=True
    )

    # Run evaluation
    print("\nĐang đánh giá mô hình...")
    result = validator.validate(
        return_predictions=True,
        verbose=True
    )

    metrics = result['metrics']
    predictions = result['predictions']

    # Print results
    print("\n" + "="*60)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("="*60)
    print(f"\nĐộ chính xác:  {metrics['accuracy']:.4f}")
    print(f"Độ chuẩn xác: {metrics['precision']:.4f}")
    print(f"Độ nhạy:    {metrics['recall']:.4f}")
    print(f"Điểm F1:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics.get('auc', 0.0):.4f}")
    print()

    # Print confusion matrix
    cm = metrics['confusion_matrix']
    print("Ma trận Nhầm lẫn:")
    print(f"                Dự đoán")
    print(f"              Cover  Stego")
    print(f"  Thực tế Cover  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"        Stego  {cm[1][0]:4d}  {cm[1][1]:4d}")
    print()

    if 'sensitivity' in metrics:
        print(f"Độ nhạy (Sensitivity): {metrics['sensitivity']:.4f}")
    if 'specificity' in metrics:
        print(f"Độ đặc hiệu (Specificity): {metrics['specificity']:.4f}")

    print("="*60 + "\n")

    # Generate detailed report
    report = validator.generate_report()
    print(report)

    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save report
    with open(save_dir / 'evaluation_report.txt', 'w') as f:
        f.write(report)
    print(f"Báo cáo đã lưu tại {save_dir / 'evaluation_report.txt'}")

    # Save plots
    if args.save_plots:
        print("\nĐang tạo và lưu biểu đồ...")

        y_true = predictions['y_true']
        y_pred = predictions['y_pred']
        y_prob = predictions['y_prob'][:, 1]

        # Plot confusion matrix
        plot_confusion_matrix(
            y_true, y_pred,
            save_path=str(save_dir / 'confusion_matrix.png')
        )

        # Only plot ROC/PR if we have both classes
        if len(set(y_true)) > 1:
            plot_roc_curve(
                y_true, y_prob,
                save_path=str(save_dir / 'roc_curve.png')
            )

            plot_precision_recall_curve(
                y_true, y_prob,
                save_path=str(save_dir / 'pr_curve.png')
            )

            plot_all_evaluation_metrics(
                y_true, y_pred, y_prob,
                save_dir=str(save_dir)
            )

        print(f"Biểu đồ đã lưu tại {save_dir}")

    print("\nĐánh giá hoàn tất!")


if __name__ == '__main__':
    main()
