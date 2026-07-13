"""
2-D CNN for Human Activity Recognition (UCI HAR dataset).

ConvBlock: a Conv2d → BatchNorm2d → ReLU → MaxPool building block shared
  by both HARCNN and KWSCNN (imported from kws_cnn.py).
HARCNN: stacks ConvBlocks over (time × sensor) input and classifies activity.
"""

import torch
import torch.nn as nn

_Stride = int | tuple[int, int]


class ConvBlock(nn.Module):
    decision_head: nn.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        padding: int | tuple[int, int] | str,
        stride1: _Stride = 1,
        stride2: _Stride = 1,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride1,
        )
        self.relu1 = nn.ReLU()
        # Dropout2d(p=0) is a no-op; only train_baseline.py (Stage 1, no
        # channel-gating yet) passes a nonzero dropout_prob. No learnable
        # parameters, so it can't affect checkpoint compatibility.
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride2,
        )
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class har_cnn(nn.Module):
    def __init__(
        self, n_channels: int = 9, n_classes: int = 6, dropout_prob: float = 0.0
    ) -> None:
        super().__init__()
        # (batch, 9, 128) -> (batch, 18, 64)
        self.conv1 = ConvBlock(
            in_channels=n_channels,
            out_channels=18,
            kernel_size=(1, 2),
            padding="same",
            dropout_prob=dropout_prob,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), padding=0)
        # (batch, 18, 64) -> (batch, 36, 32)
        self.conv2 = ConvBlock(
            in_channels=18,
            out_channels=36,
            kernel_size=(1, 2),
            padding="same",
            dropout_prob=dropout_prob,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), padding=0)

        self.ip1 = nn.Linear(16 * 72, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.unsqueeze(x, 2)

        # (batch, 9, 1, 128)
        x = self.conv1(x)
        # (batch, 18, 128)
        x = self.pool1(x)
        # (batch, 18, 1, 64)
        x = self.conv2(x)
        # (batch, 36, 64)
        x = self.pool2(x)

        x = x.view(x.size(0), 16 * 72)

        x = self.ip1(
            x,
        )
        return x
