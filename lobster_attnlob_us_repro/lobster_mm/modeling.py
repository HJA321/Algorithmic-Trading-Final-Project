from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Conv2dSame(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], stride: tuple[int, int] = (1, 1)):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_h, in_w = x.shape[-2:]
        stride_h, stride_w = self.stride
        kernel_h, kernel_w = self.kernel_size

        out_h = (in_h + stride_h - 1) // stride_h
        out_w = (in_w + stride_w - 1) // stride_w
        pad_h = max((out_h - 1) * stride_h + kernel_h - in_h, 0)
        pad_w = max((out_w - 1) * stride_w + kernel_w - in_w, 0)

        if pad_h or pad_w:
            x = F.pad(
                x,
                (
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                ),
            )
        return self.conv(x)


class AuthorMultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int = 192, num_heads: int = 10, key_dim: int = 16, output_dim: int = 64):
        super().__init__()
        self.model_dim = num_heads * key_dim
        self.q_proj = nn.Linear(input_dim, self.model_dim)
        self.k_proj = nn.Linear(input_dim, self.model_dim)
        self.v_proj = nn.Linear(input_dim, self.model_dim)
        self.attention = nn.MultiheadAttention(embed_dim=self.model_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(self.model_dim, output_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        attended, weights = self.attention(query, key, value)
        return self.out_proj(attended), weights


class AttnLOBEncoder(nn.Module):
    def __init__(self, time_window: int, feature_dim: int = 64):
        super().__init__()
        self.time_window = time_window
        self.feature_dim = feature_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            Conv2dSame(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            Conv2dSame(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(1, 5), stride=(1, 5)),
            nn.LeakyReLU(0.01),
            Conv2dSame(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            Conv2dSame(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(1, 4)),
            nn.LeakyReLU(0.01),
            Conv2dSame(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            Conv2dSame(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
        )
        self.inception_1 = nn.Sequential(
            Conv2dSame(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            Conv2dSame(64, 64, kernel_size=(3, 1)),
            nn.LeakyReLU(0.01),
        )
        self.inception_2 = nn.Sequential(
            Conv2dSame(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            Conv2dSame(64, 64, kernel_size=(5, 1)),
            nn.LeakyReLU(0.01),
        )
        self.inception_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            Conv2dSame(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
        )
        self.attention = AuthorMultiHeadAttention(input_dim=192, num_heads=10, key_dim=16, output_dim=64)
        self.proj = nn.Identity() if feature_dim == 64 else nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.cat([self.inception_1(x), self.inception_2(x), self.inception_3(x)], dim=1)
        x = x.squeeze(-1).transpose(1, 2)
        query = x[:, -1:, :]
        attended, _ = self.attention(query, x, x)
        return self.proj(attended.squeeze(1))


class MidPriceClassifier(nn.Module):
    def __init__(self, encoder: AttnLOBEncoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.feature_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class MultiInputLOBExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        time_window: int,
        encoder_path: Optional[str] = None,
        freeze_encoder: bool = True,
        output_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim=output_dim)
        self.encoder = AttnLOBEncoder(time_window=time_window, feature_dim=64)
        if encoder_path and Path(encoder_path).exists():
            state_dict = torch.load(encoder_path, map_location="cpu")
            self.encoder.load_state_dict(state_dict, strict=True)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        obs_dim = 64 + observation_space["market_state"].shape[0] + observation_space["agent_state"].shape[0]
        self.mlp = nn.Sequential(nn.Linear(obs_dim, 256), nn.LeakyReLU(0.01), nn.Linear(256, output_dim), nn.LeakyReLU(0.01))

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.encoder(observations["lob_state"])
        fused = torch.cat([encoded, observations["market_state"], observations["agent_state"]], dim=1)
        return self.mlp(fused)
