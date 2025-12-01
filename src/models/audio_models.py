import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AudioStegCNN(nn.Module):
    """CNN-based audio steganalysis model"""

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        in_channels: int = 1,
        **kwargs
    ):
        super(AudioStegCNN, self).__init__()

        # Convolutional layers for spectrogram features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class RNNAudioSteg(nn.Module):
    """RNN-based audio steganalysis model"""

    def __init__(
        self,
        num_classes: int = 2,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        **kwargs
    ):
        super(RNNAudioSteg, self).__init__()

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.classifier(lstm_out)
        return out


class HybridAudioSteg(nn.Module):
    """Hybrid CNN+RNN audio steganalysis model"""

    def __init__(
        self,
        num_classes: int = 2,
        cnn_channels: int = 64,
        rnn_hidden_size: int = 128,
        num_rnn_layers: int = 2,
        dropout: float = 0.5,
        **kwargs
    ):
        super(HybridAudioSteg, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.rnn = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.cnn(x)
        batch, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).reshape(batch, time, -1)
        x = x.view(batch, time, channels, freq).mean(dim=3)
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out[:, -1, :]
        out = self.classifier(rnn_out)
        return out


class Wav2VecSteg(nn.Module):
    """Wav2Vec-inspired audio steganalysis model"""

    def __init__(
        self,
        num_classes: int = 2,
        hidden_size: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
        **kwargs
    ):
        super(Wav2VecSteg, self).__init__()

        self.feature_encoder = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=10, stride=5, padding=5),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=2),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.feature_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.classifier(x)
        return out
