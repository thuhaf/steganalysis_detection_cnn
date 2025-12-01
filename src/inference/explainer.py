import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import os
import time



# Đảm bảo thư mục tồn tại
EXPLAIN_DIR = Path("temp/explanations")
EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)


class StegExplainer:
    """Giải thích model nhìn vào đâu để phát hiện ảnh giấu tin"""

    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        # Biến lưu gradient và activation
        self.gradients = None
        self.activations = None

        # Đăng ký hook vào layer conv cuối cùng
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Tìm layer conv cuối cùng (hỗ trợ ResNet, EfficientNet, v.v.)
        target_layer = None
        for module in reversed(list(self.model.modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break

        if target_layer is None:
            raise ValueError("Không tìm thấy Conv2d layer nào trong model!")

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def grad_cam(self, image_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Tạo heatmap Grad-CAM"""
        image_tensor = image_tensor.to(self.device)

        # Forward
        output = self.model(image_tensor)
        self.model.zero_grad()

        # Backward từ class được dự đoán
        score = output[:, target_class]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        # Global Average Pooling
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = F.relu(cam)

        # Chuẩn hóa
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize về kích thước ảnh gốc
        cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
        return cam

    def explain_prediction(
        self,
        image_path: str or Image.Image,
        method: str = "gradcam"
    ) -> Dict:
        """
        Giải thích dự đoán cho 1 ảnh

        Args:
            image_path: đường dẫn hoặc PIL.Image
            method: "gradcam" (mặc định)

        Returns:
            dict chứa kết quả + đường dẫn heatmap
        """
        # Load ảnh
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")

        # Preprocess (dùng transform giống lúc train)
        from src.data.preprocessing import ImagePreprocessor
        transform = ImagePreprocessor.get_val_transforms()
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        # Dự đoán
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probs, 1)
            predicted_class = predicted_class.item()
            confidence = confidence.item()

        prediction = "Stego" if predicted_class == 1 else "Cover"

        # Tạo Grad-CAM
        heatmap = self.grad_cam(input_tensor, predicted_class)

        # Lưu visualization
        save_path = self._save_visualization(image, heatmap, prediction, confidence)

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "heatmap_path": str(save_path),
            "message": f"Model tập trung vào các vùng đỏ/vàng để kết luận ảnh là {prediction}"
        }

    def _save_visualization(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        prediction: str,
        confidence: float
    ) -> Path:
        """Lưu 3 ảnh: gốc + heatmap + overlay"""
        orig_np = np.array(original_image)
        h, w = orig_np.shape[:2]

        # Resize heatmap
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = cv2.addWeighted(orig_np, 0.6, heatmap_colored, 0.4, 0)

        # Vẽ 3 ảnh cạnh nhau
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(orig_np)
        axes[0].set_title("Ảnh gốc")
        axes[0].axis("off")

        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title("Vùng model tập trung (Grad-CAM)")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Dự đoán: {prediction} ({confidence:.1%})")
        axes[2].axis("off")

        plt.suptitle("StegExplainer - Giải thích dự đoán Steganalysis", fontsize=16)
        plt.tight_layout()

        # Lưu
        filename = f"explain_{int(time.time() * 1000)}.png"
        save_path = EXPLAIN_DIR / filename
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Đã lưu giải thích tại: {save_path}")
        return save_path


# ==================== DÙNG NHANH (TEST) ====================
if __name__ == "__main__":
    import argparse
    from src.models.model_registry import get_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh cần giải thích")
    parser.add_argument("--model", default="checkpoints/best_model.pth", help="Model path")
    args = parser.parse_args()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("resnet50_steg", num_classes=2)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()

    # Giải thích
    explainer = StegExplainer(model, device)
    result = explainer.explain_prediction(args.image)
    print(result)
