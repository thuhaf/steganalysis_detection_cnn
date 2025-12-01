import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from data.dataset import ImageStegDataset, AudioStegDataset
from data.dataloader import StegDataLoaderFactory
from data.augmentation import get_augmentation
from data.preprocessing import ImagePreprocessor, AudioPreprocessor

from models.model_registry import ModelRegistry, get_model
from training.trainer import StegTrainer
from training.losses import get_loss_function
from training.metrics import MetricsCalculator

from utils.config import Config, load_config, create_default_config
from utils.helpers import set_seed, get_device, print_model_summary, ensure_dir
from utils.logger import TrainingLogger

import warnings

warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
warnings.filterwarnings("ignore", message=".*weights.*deprecated.*")
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train steganalysis model')

    # Config
    parser.add_argument('--config', type=str, default=None,
                        help='Đường dẫn tới tệp cấu hình (YAML/JSON)')

    # Data
    parser.add_argument('--cover-train', type=str, default='data/raw/images/cover',
                        help='Đường dẫn đến ảnh gốc huấn luyện')
    parser.add_argument('--stego-train', type=str, default='data/raw/images/stego',
                        help='Đường dẫn đến aarnh ẩn mã huấn luyện')
    parser.add_argument('--cover-val', type=str, default='',
                        help='Đường dẫn đến ảnh gốc xác thực')
    parser.add_argument('--stego-val', type=str, default='',
                        help='Đường dẫn đến ảnh ẩn mã xác thực')

    # Model
    parser.add_argument('--model', type=str, default='resnet50_steg',
                        help='Mô mô hình')
    parser.add_argument('--pretrained', action='store_true',
                        help='Sử dụng trọng số đã được huấn luyện trước')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Số lượng epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Kích thước lô')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Tốc độ học')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'adamw'],
                        help='Bộ tối ưu hóa')

    # Augmentation
    parser.add_argument('--augmentation', type=str, default='medium',
                        choices=['none', 'light', 'medium', 'strong', 'auto'],
                        help='Mức độ tăng cường dữ liệu')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Thiết bị (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Hạt giống ngẫu nhiên')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Số lượng tiến trình con để tải dữ liệu')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Thư mục để lưu trữ các checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Tiếp tục từ checkpoint đã có')

    # Experiment
    parser.add_argument('--experiment-name', type=str, default='steg_exp',
                        help='Tên thử nghiệm')

    return parser.parse_args()


def create_datasets(config: Config):
    """Create train and validation datasets"""

    # Training augmentation
    if config.data.use_augmentation:
        train_transform = get_augmentation(
            preset=config.data.augmentation_strength,
            img_size=config.data.img_size
        )
    else:
        train_transform = ImagePreprocessor.get_train_transforms(config.data.img_size)

    # Validation transform (no augmentation)
    val_transform = ImagePreprocessor.get_val_transforms(config.data.img_size)

    # Create datasets
    train_dataset = ImageStegDataset(
        cover_dir=config.data.cover_train_dir,
        stego_dir=config.data.stego_train_dir,
        transform=train_transform,
        split='train'
    )

    # Validation dataset
    if config.data.cover_val_dir and config.data.stego_val_dir:
        val_dataset = ImageStegDataset(
            cover_dir=config.data.cover_val_dir,
            stego_dir=config.data.stego_val_dir,
            transform=val_transform,
            split='val'
        )
    else:
        # Split from training data
        print(f"Không có bộ xác thực nào được chỉ định. Đang sử dụng {config.data.val_split*100}% of training data.")
        val_dataset = ImageStegDataset(
            cover_dir=config.data.cover_train_dir,
            stego_dir=config.data.stego_train_dir,
            transform=val_transform,
            split='val'
        )

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config: Config):
    """Create train and validation dataloaders"""

    train_loader = StegDataLoaderFactory.create_train_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        use_balanced_sampling=config.data.use_balanced_sampling
    )

    val_loader = StegDataLoaderFactory.create_val_dataloader(
        val_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers
    )

    return train_loader, val_loader


def create_model(config: Config):
    """Create model"""

    model = get_model(
        model_name=config.model.model_name,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
        freeze_backbone=config.model.freeze_backbone
    )

    return model


def create_optimizer(model: nn.Module, config: Config):
    """Create optimizer"""

    if config.training.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Bộ tối ưu hóa không xác định: {config.training.optimizer}")

    return optimizer


def create_scheduler(optimizer, config: Config) -> _LRScheduler | None:
    """Create learning rate scheduler"""

    if config.training.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs
        )
    elif config.training.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.scheduler_step_size,
            gamma=config.training.scheduler_factor
        )
    elif config.training.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.training.scheduler_patience,
            factor=config.training.scheduler_factor
        )
    elif config.training.scheduler == 'none':
        scheduler = None
    else:
        raise ValueError(f"Bộ lập lịch không xác định: {config.training.scheduler}")

    return scheduler


def main():
    """Main training function"""

    # Parse arguments
    args = parse_args()

    # Load or create config
    if args.config:
        config = load_config(args.config)
        print(f"Đã tải cấu hình từ {args.config}")
    else:
        config = Config()
        print("Đang sử dụng cấu hình mặc định")

    # Override config with command line arguments
    if args.cover_train:
        config.data.cover_train_dir = args.cover_train
    if args.stego_train:
        config.data.stego_train_dir = args.stego_train
    if args.cover_val:
        config.data.cover_val_dir = args.cover_val
    if args.stego_val:
        config.data.stego_val_dir = args.stego_val

    config.model.model_name = args.model
    config.model.pretrained = args.pretrained
    config.training.num_epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.optimizer = args.optimizer
    config.data.augmentation_strength = args.augmentation
    config.device = args.device
    config.seed = args.seed
    config.data.num_workers = args.num_workers
    config.logging.checkpoint_dir = args.checkpoint_dir
    config.experiment_name = args.experiment_name

    # Validate config
    config.validate()

    # Print config
    config.print_config()

    # Set seed for reproducibility
    set_seed(config.seed)

    # Get device
    device = get_device(config.device)

    # Create directories
    ensure_dir(config.logging.checkpoint_dir)
    ensure_dir(config.logging.log_dir)

    # Setup logger
    logger = TrainingLogger(config.logging.log_dir, config.experiment_name)
    logger.log_config(config.to_dict())

    # Create datasets
    print("\n" + "="*60)
    print("Đang tạo bộ dữ liệu...")
    print("="*60)
    train_dataset, val_dataset = create_datasets(config)

    # Create dataloaders
    print("\nĐang tạo dataloader...")
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)

    # Create model
    print("\n" + "="*60)
    print("Đang tạo mô hình...")
    print("="*60)
    model = create_model(config)
    print_model_summary(model)

    # Create loss function
    if config.training.loss_function == 'focal':
        criterion = get_loss_function(
            config.training.loss_function,
            alpha=config.training.focal_alpha,
            gamma=config.training.focal_gamma
        )
    elif config.training.loss_function == 'label_smoothing':
        criterion = get_loss_function(
            config.training.loss_function,
            num_classes=config.model.num_classes,
            smoothing=config.training.label_smoothing
        )
    else:
        # For 'ce' and 'weighted_ce'
        criterion = get_loss_function(
            config.training.loss_function
        )

    # Create optimizer
    optimizer = create_optimizer(model, config)
    print(f"Bộ tối ưu hóa: {config.training.optimizer}")

    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    if scheduler:
        print(f"Bộ lập lịch: {config.training.scheduler}")

    # Create trainer
    trainer = StegTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        save_dir=config.logging.checkpoint_dir,
        use_amp=config.training.use_mixed_precision
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nĐang tiếp tục từ checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n" + "="*60)
    print("Bắt đầu huấn luyện...")
    print("="*60 + "\n")

    history = trainer.train(
        num_epochs=config.training.num_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        save_best_only=config.logging.save_best_only
    )

    # Save final config
    config_save_path = Path(config.logging.checkpoint_dir) / f"{config.experiment_name}_config.yaml"
    config.save_yaml(str(config_save_path))

    print("\n" + "="*60)
    print("Huấn luyện hoàn tất!")
    print(f"Mô hình tốt nhất đã lưu tại: {config.logging.checkpoint_dir}")
    print(f"Cấu hình đã lưu tại: {config_save_path}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
