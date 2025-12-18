import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import numpy as np

# Import các modules nội bộ
from src.config import Config
from src.data.dataset import StegoImageDataset, StegoAudioDataset
from src.models.stego_cnn import StegoCNN

def load_model(model, checkpoint_path, device):
    """Tải trọng số đã huấn luyện vào mô hình."""
    if not os.path.exists(checkpoint_path):
        print(f"LỖI: Không tìm thấy checkpoint tại {checkpoint_path}")
        return None

    # Tải trọng số
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"-> Đã tải model thành công từ {checkpoint_path}")
    return model

def test_model(model, loader, criterion, device):
    """
    Chạy quá trình kiểm thử và tính toán các chỉ số hiệu suất.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0.0

    with torch.no_grad():
        loop = tqdm(loader, desc="Testing")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Tính xác suất (probability) bằng Sigmoid
            probs = torch.sigmoid(outputs)

            # Dự đoán nhãn (0 hoặc 1)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(loader)

    # Chuyển đổi kết quả sang định dạng numpy mảng 1 chiều
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()

    # Tính toán các chỉ số
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    # Kiểm tra để tránh lỗi ROC AUC khi chỉ có 1 class
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.5 # Gán giá trị mặc định nếu không thể tính toán

    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1_score, roc_auc, cm

def print_metrics(loss, acc, prec, rec, f1, auc, cm):
    """In kết quả ra console."""
    print("\n==================== KẾT QUẢ KIỂM THỬ ====================")
    print(f"Tổng Loss: {loss:.4f}")
    print(f"Accuracy (Độ chính xác): {acc * 100:.2f}%")
    print(f"Precision (Độ chuẩn xác): {prec:.4f}")
    print(f"Recall (Độ nhạy): {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("\nMa trận nhầm lẫn (Confusion Matrix):")
    print(cm)
    print("  [TN  FP]")
    print("  [FN  TP]")
    print("=========================================================")

def main():
    parser = argparse.ArgumentParser(description="Kiểm thử mô hình Steganalysis đã huấn luyện")
    parser.add_argument("--data_dir", type=str, default=Config.DATA_ROOT, help="Đường dẫn thư mục data")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"), help="Đường dẫn file checkpoint model")
    parser.add_argument("--media_type", type=str, default="image", choices=["image", "audio"], help="Loại dữ liệu (image hoặc audio)")

    args = parser.parse_args()

    # 1. Load Data
    if args.media_type == "image":
        DatasetClass = StegoImageDataset
    elif args.media_type == "audio":
        DatasetClass = StegoAudioDataset

    test_path = os.path.join(args.data_dir, 'test')
    if not os.path.exists(test_path):
        print(f"LỖI: Không tìm thấy thư mục kiểm thử: {test_path}. Vui lòng tạo thư mục này.")
        return

    test_dataset = DatasetClass(test_path, image_size=Config.IMAGE_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Khởi tạo Model và Load Checkpoint
    model = StegoCNN()
    model = load_model(model, args.checkpoint, Config.DEVICE)
    if model is None:
        return

    # Loss Function (dùng cùng loại với lúc train)
    criterion = nn.BCEWithLogitsLoss()

    # 3. Chạy kiểm thử
    loss, acc, prec, rec, f1, auc, cm = test_model(model, test_loader, criterion, Config.DEVICE)

    # 4. In kết quả
    print_metrics(loss, acc, prec, rec, f1, auc, cm)

if __name__ == "__main__":
    main()

