import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from src.models.model_registry import ModelRegistry, get_model


# ============================
# 1. INPUT MẪU GIẢ LẬP
# ============================


# Ảnh RGB 224x224
dummy_image = torch.randn(1, 3, 224, 224)


# Âm thanh dạng Mel 128x128
dummy_audio = torch.randn(1, 1, 128, 128)


# ============================
# 2. DANH SÁCH MODEL
# ============================


image_models = ModelRegistry.list_models("image")
audio_models = ModelRegistry.list_models("audio")


print("===== IMAGE MODELS =====")
print(image_models)


print("\n===== AUDIO MODELS =====")
print(audio_models)


# ============================
# 3. DATASET IMAGE THẬT
# ============================


class ImageStegoDataset(Dataset):
    """
    Dataset ảnh: sạch=0, ẩn mã=1
    Folder structure:
        root/clean/*.png/jpg
        root/stego/*.png/jpg
    """
    def __init__(self, root_dir, transform=None):
        self.images, self.labels = [], []
        self.transform = transform


        for label, subdir in enumerate(["clean", "stego"]):
            folder = os.path.join(root_dir, subdir)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.images.append(os.path.join(folder, f))
                        self.labels.append(label)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx])


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# Thay bằng đường dẫn dataset ảnh thật
dataset_path_images = "./dataset/images"
image_dataset = ImageStegoDataset(dataset_path_images, transform=image_transform)
image_loader = DataLoader(image_dataset, batch_size=8, shuffle=False)


# ============================
# 4. DATASET AUDIO THẬT (Placeholder)
# ============================


class AudioStegoDataset(Dataset):
    """
    Dataset audio: sạch=0, ẩn mã=1
    Folder structure:
        root/clean/*.pt  (tensor)
        root/stego/*.pt
    """
    def __init__(self, root_dir):
        self.data, self.labels = [], []


        for label, subdir in enumerate(["clean", "stego"]):
            folder = os.path.join(root_dir, subdir)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.endswith(".pt"):
                        self.data.append(os.path.join(folder, f))
                        self.labels.append(label)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        audio = torch.load(self.data[idx])
        return audio, torch.tensor(self.labels[idx])


# Thay bằng đường dẫn dataset audio thật
dataset_path_audio = "./dataset/audio"
audio_dataset = AudioStegoDataset(dataset_path_audio)
audio_loader = DataLoader(audio_dataset, batch_size=8, shuffle=False)


# ============================
# 5. HÀM TEST IMAGE MODEL
# ============================


def test_image_model(model_name, use_dummy=True):
    print(f"\n=== Testing Image Model: {model_name} ===")
    try:
        model = get_model(model_name, num_classes=2, pretrained=False)
        model.eval()


        correct, total = 0, 0


        data_iter = [(dummy_image, torch.tensor([0]))] if use_dummy else image_loader


        with torch.no_grad():
            for batch_idx, batch in enumerate(data_iter):
                images, labels = batch
                out = model(images)
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)


                print(f"Batch {batch_idx+1}: Output shape {out.shape}" if not use_dummy else f"Output shape: {out.shape}")


        if not use_dummy:
            accuracy = correct / total * 100
            print(f"✅ Accuracy on dataset: {accuracy:.2f}%")
        else:
            print("✅ PASS (dummy input)")


    except Exception as e:
        print(f"❌ FAIL: {e}")


# ============================
# 6. HÀM TEST AUDIO MODEL
# ============================


def test_audio_model(model_name, use_dummy=True):
    print(f"\n=== Testing Audio Model: {model_name} ===")
    try:
        model = get_model(model_name, num_classes=2, pretrained=False)
        model.eval()


        correct, total = 0, 0


        data_iter = [(dummy_audio, torch.tensor([0]))] if use_dummy else audio_loader


        with torch.no_grad():
            for batch_idx, batch in enumerate(data_iter):
                audios, labels = batch
                out = model(audios)
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)


                print(f"Batch {batch_idx+1}: Output shape {out.shape}" if not use_dummy else f"Output shape: {out.shape}")


        if not use_dummy and data_iter is not None:
            accuracy = correct / total * 100
            print(f"✅ Accuracy on dataset: {accuracy:.2f}%")
        else:
            print("✅ PASS (dummy input)")


    except Exception as e:
        print(f"❌ FAIL: {e}")


# ============================
# 7. CHẠY TEST TẤT CẢ
# ============================


print("\n==============================")
print("TESTING ALL IMAGE MODELS (DUMMY)")
print("==============================")
for m in image_models:
    test_image_model(m, use_dummy=True)


# Nếu muốn test dataset thật cho ảnh, bỏ comment phần dưới
# print("\n==============================")
# print("TESTING ALL IMAGE MODELS (DATASET)")
# print("==============================")
# for m in image_models:
#     test_image_model(m, use_dummy=False)


print("\n==============================")
print("TESTING ALL AUDIO MODELS (DUMMY)")
print("==============================")
for m in audio_models:
    test_audio_model(m, use_dummy=True)


# Nếu muốn test dataset thật cho audio, bỏ comment phần dưới
# print("\n==============================")
# print("TESTING ALL AUDIO MODELS (DATASET)")
# print("==============================")
# for m in audio_models:
#     test_audio_model(m, use_dummy=False)


print("\n🎉 DONE! Toàn bộ model đã được test xong.")



