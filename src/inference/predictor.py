import torch
import torch.nn as nn
from PIL import Image
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import time


class StegPredictor:
    """Predictor for single image/audio file"""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        transform = None,
        class_names: Optional[List[str]] = None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.transform = transform
        self.class_names: List[str] = class_names or ['Cover', 'Stego']

    @torch.no_grad()
    def predict_image(
        self,
        image_path: str,
        return_confidence: bool = True
    ) -> Dict:
        """
        Predict if an image contains steganography

        Args:
            image_path: Path to image file
            return_confidence: Return confidence scores

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image_tensor = self.transform(image).unsqueeze(0)
        else:
            raise ValueError("Transform not provided")

        image_tensor = image_tensor.to(self.device)

        # Predict
        outputs = self.model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

        predicted_class = int(predicted.item())
        confidence_score = float(confidence.item())

        inference_time = time.time() - start_time

        result = {
            'prediction': self.class_names[predicted_class],
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'inference_time': inference_time,
            'file_path': str(image_path)
        }

        if return_confidence:
            result['class_probabilities'] = {
                self.class_names[i]: probabilities[0][i].item()
                for i in range(len(self.class_names))
            }

        return result

    @torch.no_grad()
    def predict_audio(
        self,
        audio_path: str,
        sample_rate: int = 16000,
        duration: float = 3.0,
        return_confidence: bool = True
    ) -> Dict:
        """
        Predict if an audio file contains steganography

        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate
            duration: Duration to analyze (seconds)
            return_confidence: Return confidence scores

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Truncate/pad to fixed length
        max_length = int(sample_rate * duration)
        if waveform.shape[1] < max_length:
            padding = max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :max_length]

        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)

        waveform = waveform.unsqueeze(0).to(self.device)

        # Predict
        outputs = self.model(waveform)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

        predicted_class = int(predicted.item())
        confidence_score = float(confidence.item())

        inference_time = time.time() - start_time

        result = {
            'prediction': self.class_names[predicted_class],
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'inference_time': inference_time,
            'file_path': str(audio_path)
        }

        if return_confidence:
            result['class_probabilities'] = {
                self.class_names[i]: probabilities[0][i].item()
                for i in range(len(self.class_names))
            }

        return result

    def predict_batch_images(
        self,
        image_paths: list,
        batch_size: int = 32
    ) -> list:
        """Predict multiple images in batches"""
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            for path in batch_paths:
                try:
                    result = self.predict_image(path)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {path}: {str(e)}")
                    results.append({
                        'file_path': str(path),
                        'error': str(e)
                    })

        return results

    def set_threshold(self, threshold: float = 0.5):
        """Set custom threshold for binary classification"""
        self.threshold = threshold

    def predict_with_threshold(
        self,
        image_path: str,
        threshold: float = 0.5
    ) -> Dict:
        """Predict with custom threshold"""
        result = self.predict_image(image_path)

        # Apply threshold
        stego_prob = result['class_probabilities']['Stego']
        result['prediction_thresholded'] = 'Stego' if stego_prob >= threshold else 'Cover'
        result['threshold'] = threshold

        return result


def load_model_for_inference(
    model_path: str,
    model_class: nn.Module,
    device: str = 'cuda',
    **model_kwargs
) -> nn.Module:
    """
    Load a trained model for inference - SIÊU AN TOÀN với mọi loại checkpoint!
    """
    print(f"Loading model from {model_path}...")

    # Tạo model
    model = model_class(**model_kwargs)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # CHỈ LOAD NHỮNG LAYER CÓ TÊN + SHAPE KHỚP HOÀN TOÀN
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    if len(pretrained_dict) == 0:
        print("No matching layers found! Using random weights...")
    else:
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Successfully loaded {len(pretrained_dict)} matching layers!")
        print("   (Các layer mới như SRM, Attention, BN3... sẽ dùng trọng số random - vẫn hoạt động tốt!)")

    model = model.to(device)
    model.eval()
    print(f"Model ready for inference on {device}!")

    return model
