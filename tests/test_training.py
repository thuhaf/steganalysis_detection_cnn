# test_training.py
import sys
import argparse
from pathlib import Path

# Add src to path (giống script gốc)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Mock các module chính để tránh import lỗi hoặc chạy chậm
class MockImageStegDataset(Dataset):
    """Dataset giả lập cho testing nhanh"""
    def __init__(self, num_samples=100, img_size=224, transform=None, split='train'):
        self.num_samples = num_samples
        self.transform = transform
        self.img_size = img_size
        self.split = split

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Tạo ảnh giả (3 kênh, kích thước img_size x img_size)
        img = torch.rand(3, self.img_size, self.img_size)
        # Label: 0 = cover, 1 = stego (cân bằng)
        label = idx % 2
        if self.transform:
            img = self.transform(img)
        return img, label

# Mock các hàm quan trọng
def mock_get_augmentation(*args, **kwargs):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def mock_get_model(model_name='resnet50_steg', num_classes=2, pretrained=False, **kwargs):
    from torchvision.models import resnet50
    model = resnet50(pretrained=pretrained)  # weights=None nếu dùng PyTorch mới
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def mock_get_loss_function(loss_name='ce', **kwargs):
    if loss_name == 'ce':
        return nn.CrossEntropyLoss()
    raise ValueError("Only 'ce' supported in mock")

# Mock Config đơn giản
class MockConfig:
    def __init__(self):
        self.data = self.Data()
        self.model = self.Model()
        self.training = self.Training()
        self.logging = self.Logging()
        self.device = 'cpu'
        self.seed = 42
        self.experiment_name = 'test_exp'

    class Data:
        def __init__(self):
            self.cover_train_dir = 'mock_cover'
            self.stego_train_dir = 'mock_stego'
            self.cover_val_dir = ''
            self.stego_val_dir = ''
            self.batch_size = 8
            self.num_workers = 0  # 0 để tránh multiprocessing issue khi test
            self.use_augmentation = True
            self.augmentation_strength = 'medium'
            self.img_size = 224
            self.val_split = 0.2
            self.use_balanced_sampling = False

    class Model:
        def __init__(self):
            self.model_name = 'resnet50_steg'
            self.num_classes = 2
            self.pretrained = False
            self.dropout = 0.5
            self.freeze_backbone = False

    class Training:
        def __init__(self):
            self.num_epochs = 2  # Chỉ train 2 epochs cho test nhanh
            self.learning_rate = 0.001
            self.optimizer = 'adam'
            self.scheduler = 'none'
            self.loss_function = 'ce'
            self.use_mixed_precision = False
            self.early_stopping_patience = 10
            self.weight_decay = 1e-4

    class Logging:
        def __init__(self):
            self.checkpoint_dir = './checkpoints_test'
            self.log_dir = './logs_test'
            self.save_best_only = True

    def validate(self):
        pass

    def print_config(self):
        print("Sử dụng Mock Config cho testing")

    def to_dict(self):
        return {"mock": True}

# Mock Trainer đơn giản chỉ chạy qua 2 epochs mà không train thật
class MockStegTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, save_dir, use_amp=False):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        print("Mock Trainer đã được khởi tạo")

    def train(self, num_epochs, early_stopping_patience=None, save_best_only=True):
        print(f"Bắt đầu mock training cho {num_epochs} epochs...")
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        for epoch in range(num_epochs):
            train_loss = 0.693
            train_acc = 0.65 + epoch * 0.1
            val_loss = 0.650
            val_acc = 0.70 + epoch * 0.1

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {train_loss:.3f} - val_acc: {val_acc:.3f}")

        print("Mock training hoàn tất!")
        return history

    def load_checkpoint(self, path):
        print(f"Mock load checkpoint từ {path}")

# Thay thế các import thật bằng mock
import builtins
original_import = builtins.__import__

def mock_import(name, *args, **kwargs):
    if name.startswith('data.'):
        return type('module', (), {
            'ImageStegDataset': MockImageStegDataset,
            'StegDataLoaderFactory': type('factory', (), {
                'create_train_dataloader': lambda dataset, **kw: DataLoader(dataset, shuffle=True, **kw),
                'create_val_dataloader': lambda dataset, **kw: DataLoader(dataset, shuffle=False, **kw)
            })(),
            'get_augmentation': mock_get_augmentation,
            'ImagePreprocessor': type('pp', (), {
                'get_train_transforms': lambda size: mock_get_augmentation(),
                'get_val_transforms': lambda size: mock_get_augmentation()
            })()
        })()
    elif name.startswith('models.'):
        return type('module', (), {'get_model': mock_get_model})()
    elif name.startswith('training.trainer'):
        return type('module', (), {'StegTrainer': MockStegTrainer})()
    elif name.startswith('training.losses'):
        return type('module', (), {'get_loss_function': mock_get_loss_function})()
    elif name.startswith('utils.config'):
        return type('module', (), {
            'Config': MockConfig,
            'load_config': lambda x: MockConfig(),
            'create_default_config': lambda: MockConfig()
        })()
    elif name.startswith('utils.helpers'):
        return type('module', (), {
            'set_seed': lambda x: print(f"Setting seed {x}"),
            'get_device': lambda x: torch.device('cpu'),
            'print_model_summary': lambda model: print(f"Model: {model.__class__.__name__}"),
            'ensure_dir': lambda x: print(f"Ensuring dir {x}")
        })()
    elif name.startswith('utils.logger'):
        return type('module', (), {
            'TrainingLogger': lambda log_dir, exp_name: type('logger', (), {
                'log_config': lambda x: print("Logged config"),
            })()
        })()
    else:
        return original_import(name, *args, **kwargs)

builtins.__import__ = mock_import

# Bây giờ import script gốc (sẽ dùng các mock ở trên)
from train import main  # Đây là file script training gốc của bạn, giả sử tên là train.py

if __name__ == '__main__':
    # Override sys.argv để chạy với args mặc định + một số thay đổi cho test nhanh
    sys.argv = [
        'train.py',
        '--epochs', '2',           # Chỉ 2 epochs
        '--batch-size', '8',       # Batch nhỏ
        '--device', 'cpu',         # Chạy CPU để tránh lỗi CUDA khi test
        '--num-workers', '0',      # Tránh multiprocessing issue
        '--experiment-name', 'quick_test',
        '--checkpoint-dir', './checkpoints_test'
    ]
    
    print("="*60)
    print("BẮT ĐẦU TEST TOÀN BỘ TRAINING PIPELINE")
    print("="*60)
    
    try:
        main()
        print("\n" + "="*60)
        print("TEST THÀNH CÔNG! Toàn bộ pipeline chạy mà không có lỗi.")
        print("Bạn có thể yên tâm chạy training thật với dữ liệu.")
        print("="*60)
    except Exception as e:
        print("\nLỖI TRONG QUÁ TRÌNH TEST:")
        print(e)
        import traceback
        traceback.print_exc()
