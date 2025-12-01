import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
from .metrics import MetricsCalculator, AverageMeter, MetricsTracker
from .losses import get_loss_function


class StegTrainer:
    """Trainer class for steganalysis models"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        save_dir: str = './checkpoints',
        use_amp: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and device == 'cuda'

        # Mixed precision training - Fixed deprecation warning
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0

        print(f"Đang huấn luyện trên thiết bị: {device}")
        print(f"Tính toán hỗn hợp (Mixed precision): {self.use_amp}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]', leave=False)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            batch_size = inputs.size(0)

            # Forward pass with mixed precision
            if self.use_amp and self.scaler is not None:
                # Fixed deprecation warning
                with torch.amp.autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            accuracy = correct / batch_size

            # Update meters
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy, batch_size)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}'
            })

        return {
            'train_loss': loss_meter.avg,
            'train_acc': acc_meter.avg
        }

    def _run_validation(self, epoch: int) -> Dict[str, float]:
        """Run validation epoch (renamed from 'validate' to avoid naming conflicts)"""
        self.model.eval()
        loss_meter = AverageMeter()
        all_preds = []
        all_targets = []
        all_probs = []

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]', leave=False)

        with torch.no_grad():
            for inputs, targets in pbar:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                batch_size = inputs.size(0)

                if self.use_amp:
                    # Fixed deprecation warning
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                loss_meter.update(loss.item(), batch_size)
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        # Convert to numpy
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # Calculate all metrics
        metrics = MetricsCalculator.calculate_metrics(
            all_targets, all_preds, all_probs
        )
        metrics['val_loss'] = loss_meter.avg

        return metrics

    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
        save_best_only: bool = True
    ) -> Dict:
        """
        Train the model for multiple epochs with early stopping and checkpointing.
        """
        print(f"\nBắt đầu huấn luyện trong {num_epochs} epoch...")
        print(f"{'='*60}\n")

        best_val_f1 = 0.0
        patience_counter = 0
        history = {}

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Training epoch
            train_metrics = self.train_epoch(epoch)

            # Validation epoch
            val_metrics = self._run_validation(epoch)

            # Learning rate scheduler step - FIXED: Move before metrics tracking
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            # Combine metrics for logging
            epoch_metrics = {**train_metrics, **val_metrics}
            self.metrics_tracker.update(epoch_metrics)

            # Logging
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{num_epochs} - Time: {epoch_time:.2f}s")
            print(f"Huấn luyện → Mất mát: {train_metrics['train_loss']:.4f} | Đ.chính xác: {train_metrics['train_acc']:.4f}")
            print(f"Xác thực   → Mất mát: {val_metrics['val_loss']:.4f} | "
                  f"Đ.chính xác: {val_metrics['accuracy']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | "
                  f"AUC: {val_metrics.get('auc', 0.0):.4f}")
            print(f"{'='*60}\n")

            # Checkpointing logic
            is_best = val_metrics['f1'] > best_val_f1
            if is_best:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                self.save_checkpoint(epoch, epoch_metrics, is_best=True)
                print(f"Đã lưu mô hình tốt nhất mới! Val F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                if not save_best_only:
                    self.save_checkpoint(epoch, epoch_metrics, is_best=False)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Đã kích hoạt dừng sớm sau {epoch} epoch (không cải thiện trong {early_stopping_patience} epochs)")
                break

        print(f"\nHuấn luyện hoàn tất! F1 Xác thực tốt nhất: {best_val_f1:.4f}")
        return self.metrics_tracker.get_history()

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict,
        is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Đã lưu mô hình tốt nhất tới {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Đã tải checkpoint từ {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}")

        return checkpoint['epoch'], checkpoint['metrics']
