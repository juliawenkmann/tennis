from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        *,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True),
            nn.ReLU(),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BallTrackNet(nn.Module):
    """Minimal TrackNet-style network for tennis ball heatmaps."""

    def __init__(self, out_channels: int = 2, use_batch_norm: bool = True) -> None:
        super().__init__()
        self.out_channels = out_channels

        self.encoder = nn.Sequential(
            ConvBlock(9, 64, use_batch_norm=use_batch_norm),
            ConvBlock(64, 64, use_batch_norm=use_batch_norm),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, use_batch_norm=use_batch_norm),
            ConvBlock(128, 128, use_batch_norm=use_batch_norm),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, use_batch_norm=use_batch_norm),
            ConvBlock(256, 256, use_batch_norm=use_batch_norm),
            ConvBlock(256, 256, use_batch_norm=use_batch_norm),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 512, use_batch_norm=use_batch_norm),
            ConvBlock(512, 512, use_batch_norm=use_batch_norm),
            ConvBlock(512, 512, use_batch_norm=use_batch_norm),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(512, 256, use_batch_norm=use_batch_norm),
            ConvBlock(256, 256, use_batch_norm=use_batch_norm),
            ConvBlock(256, 256, use_batch_norm=use_batch_norm),
            nn.Upsample(scale_factor=2),
            ConvBlock(256, 128, use_batch_norm=use_batch_norm),
            ConvBlock(128, 128, use_batch_norm=use_batch_norm),
            nn.Upsample(scale_factor=2),
            ConvBlock(128, 64, use_batch_norm=use_batch_norm),
            ConvBlock(64, 64, use_batch_norm=use_batch_norm),
            ConvBlock(64, out_channels, use_batch_norm=use_batch_norm),
        )
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x: torch.Tensor, *, testing: bool = False) -> torch.Tensor:
        batch_size = x.size(0)
        features = self.encoder(x)
        scores_map = self.decoder(features)
        output = scores_map.reshape(batch_size, self.out_channels, -1)
        if testing:
            output = self.softmax(output)
        return output

    def inference(self, frames: torch.Tensor) -> tuple[int | None, int | None]:
        self.eval()
        with torch.inference_mode():
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)
            output = self(frames, testing=True)
            output = output.argmax(dim=1).detach().cpu().numpy()
            if self.out_channels == 2:
                output *= 255
        return self._extract_ball_center(output)

    def _extract_ball_center(self, output: np.ndarray) -> tuple[int | None, int | None]:
        heatmap = output.reshape((360, 640)).astype(np.uint8)
        _, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(heatmap, connectivity=8)
        if component_count <= 1:
            return None, None

        best_index = None
        best_score = None
        for index in range(1, component_count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < 2 or area > 150:
                continue

            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            size_penalty = abs(area - 16)
            aspect_penalty = abs(width - height)
            score = (size_penalty, aspect_penalty)
            if best_score is None or score < best_score:
                best_score = score
                best_index = index

        if best_index is None:
            return None, None

        x = int(round(float(centroids[best_index][0])))
        y = int(round(float(centroids[best_index][1])))
        return x, y

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def stack_three_frames(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    frame_c: np.ndarray,
    *,
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    frames = []
    for frame in (frame_a, frame_b, frame_c):
        resized = cv2.resize(frame, (width, height)).astype(np.float32)
        frames.append(resized)
    merged = np.concatenate(frames, axis=2)
    return np.rollaxis(merged, 2, 0)
