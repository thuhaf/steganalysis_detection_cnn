
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, cast


class ResNetSteg(nn.Module):
    """ResNet-based steganalysis model"""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = 'resnet50',
        freeze_backbone: bool = False,
        dropout: float = 0.5
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            model_name: Which ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            freeze_backbone: Freeze backbone weights
            dropout: Dropout probability
        """
        super(ResNetSteg, self).__init__()

        # Load pretrained ResNet
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final layer with custom classifier
        self.backbone.fc = cast(nn.Linear, nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, num_classes)
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EfficientNetSteg(nn.Module):
    """EfficientNet-based steganalysis model"""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = 'efficientnet_b0',
        freeze_backbone: bool = False,
        dropout: float = 0.4
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            model_name: Which EfficientNet variant
            freeze_backbone: Freeze backbone weights
            dropout: Dropout probability
        """
        super(EfficientNetSteg, self).__init__()

        # Load pretrained EfficientNet
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = 1280
        elif model_name == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            num_features = 1280
        elif model_name == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
            num_features = 1408
        elif model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            num_features = 1536
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class VGGSteg(nn.Module):
    """VGG-based steganalysis model"""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = 'vgg16',
        freeze_backbone: bool = False,
        dropout: float = 0.5
    ):
        super(VGGSteg, self).__init__()

        # Load pretrained VGG
        if model_name == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
        elif model_name == 'vgg19':
            self.backbone = models.vgg19(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DenseNetSteg(nn.Module):
    """DenseNet-based steganalysis model"""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = 'densenet121',
        freeze_backbone: bool = False,
        dropout: float = 0.5
    ):
        super(DenseNetSteg, self).__init__()

        # Load pretrained DenseNet
        if model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = 1024
        elif model_name == 'densenet161':
            self.backbone = models.densenet161(pretrained=pretrained)
            num_features = 2208
        elif model_name == 'densenet169':
            self.backbone = models.densenet169(pretrained=pretrained)
            num_features = 1664
        elif model_name == 'densenet201':
            self.backbone = models.densenet201(pretrained=pretrained)
            num_features = 1920
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier
        self.backbone.classifier = cast(nn.Linear, nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, num_classes)
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MobileNetSteg(nn.Module):
    """MobileNet-based lightweight model for steganalysis"""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        super(MobileNetSteg, self).__init__()

        self.backbone = models.mobilenet_v2(pretrained=pretrained)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.67),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class InceptionV3Steg(nn.Module):
    """Inception V3-based steganalysis model"""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5
    ):
        super(InceptionV3Steg, self).__init__()

        self.backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = cast(nn.Linear, nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, num_classes)
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ViTSteg(nn.Module):
    """Vision Transformer-based steganalysis model"""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = 'vit_b_16',
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        super(ViTSteg, self).__init__()

        # Load pretrained ViT
        if model_name == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            hidden_dim = 768
        elif model_name == 'vit_b_32':
            self.backbone = models.vit_b_32(pretrained=pretrained)
            hidden_dim = 768
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace head
        self.backbone.heads = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
