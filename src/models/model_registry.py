import torch.nn as nn
from typing import Dict, Any
from .cnn_base import StegCNNBase, DeepStegCNN, LightweightStegCNN
from .image_models import (
    ResNetSteg, EfficientNetSteg, VGGSteg,
    DenseNetSteg, MobileNetSteg
)
from .audio_models import (
    AudioStegCNN, RNNAudioSteg,
    HybridAudioSteg, Wav2VecSteg
)


class ModelRegistry:
    """Registry for all available models"""

    _image_models = {
        'steg_cnn_base': StegCNNBase,
        'deep_steg_cnn': DeepStegCNN,
        'lightweight_steg_cnn': lambda **kwargs: LightweightStegCNN(**{k: v for k, v in kwargs.items() if k != 'pretrained'}),
        'resnet18_steg': lambda **kwargs: ResNetSteg(model_name='resnet18', **kwargs),
        'resnet34_steg': lambda **kwargs: ResNetSteg(model_name='resnet34', **kwargs),
        'resnet50_steg': lambda **kwargs: ResNetSteg(model_name='resnet50', **kwargs),
        'resnet101_steg': lambda **kwargs: ResNetSteg(model_name='resnet101', **kwargs),
        'efficientnet_b0_steg': lambda **kwargs: EfficientNetSteg(model_name='efficientnet_b0', **kwargs),
        'efficientnet_b1_steg': lambda **kwargs: EfficientNetSteg(model_name='efficientnet_b1', **kwargs),
        'vgg16_steg': lambda **kwargs: VGGSteg(model_name='vgg16', **kwargs),
        'vgg19_steg': lambda **kwargs: VGGSteg(model_name='vgg19', **kwargs),
        'densenet121_steg': lambda **kwargs: DenseNetSteg(model_name='densenet121', **kwargs),
        'densenet161_steg': lambda **kwargs: DenseNetSteg(model_name='densenet161', **kwargs),
        'mobilenet_steg': MobileNetSteg,
    }

    _audio_models = {
        'audio_steg_cnn': AudioStegCNN,
        'rnn_audio_steg': RNNAudioSteg,
        'hybrid_audio_steg': HybridAudioSteg,
        'wav2vec_steg': Wav2VecSteg,
    }

    @classmethod
    def list_models(cls, modality: str = 'all') -> list:
        """
        List all available models

        Args:
            modality: 'image', 'audio', or 'all'
        """
        if modality == 'image':
            return list(cls._image_models.keys())
        elif modality == 'audio':
            return list(cls._audio_models.keys())
        else:
            return list(cls._image_models.keys()) + list(cls._audio_models.keys())

    @classmethod
    def create_model(
        cls,
        model_name: str,
        modality: str = 'image',
        **kwargs
    ) -> nn.Module:
        """
        Create a model instance

        Args:
            model_name: Name of the model
            modality: 'image' or 'audio'
            **kwargs: Additional arguments for model initialization

        Returns:
            Initialized model
        """
        if modality == 'image':
            if model_name not in cls._image_models:
                raise ValueError(
                    f"Unknown image model: {model_name}. "
                    f"Available: {list(cls._image_models.keys())}"
                )
            return cls._image_models[model_name](**kwargs)

        elif modality == 'audio':
            if model_name not in cls._audio_models:
                raise ValueError(
                    f"Unknown audio model: {model_name}. "
                    f"Available: {list(cls._audio_models.keys())}"
                )
            return cls._audio_models[model_name](**kwargs)

        else:
            raise ValueError(f"Unknown modality: {modality}")

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        all_models = {**cls._image_models, **cls._audio_models}

        if model_name not in all_models:
            raise ValueError(f"Unknown model: {model_name}")

        modality = 'image' if model_name in cls._image_models else 'audio'

        return {
            'name': model_name,
            'modality': modality,
            'class': all_models[model_name]
        }


def get_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Convenience function to get a model

    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        pretrained: Use pretrained weights (if available)
        **kwargs: Additional model arguments

    Returns:
        Initialized model
    """
    # Determine modality from model name
    if any(x in model_name for x in ['audio', 'rnn', 'wav2vec', 'hybrid']):
        modality = 'audio'
    else:
        modality = 'image'

    return ModelRegistry.create_model(
        model_name,
        modality=modality,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )
