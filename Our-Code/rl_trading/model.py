from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from attn_lob_pytorch.model import AttnLOBEncoder

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    class BaseFeaturesExtractor(nn.Module):
        def __init__(self, observation_space, features_dim: int):
            super().__init__()
            self.features_dim = features_dim


class MultiInputLOBExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        time_window: int,
        encoder_path: str | None = None,
        freeze_encoder: bool = True,
        output_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim=output_dim)
        self.encoder = AttnLOBEncoder(time_window=time_window, embedding_dim=64)
        if encoder_path:
            checkpoint_path = Path(encoder_path)
            if checkpoint_path.exists():
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                self.encoder.load_state_dict(state_dict, strict=True)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        market_dim = int(observation_space["market_state"].shape[0])
        agent_dim = int(observation_space["agent_state"].shape[0])
        fusion_dim = 64 + market_dim + agent_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, output_dim),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        lob_state = observations["lob_state"]
        if lob_state.ndim == 3:
            lob_state = lob_state.unsqueeze(1)
        encoded = self.encoder(lob_state)
        fused = torch.cat([encoded, observations["market_state"], observations["agent_state"]], dim=1)
        return self.fusion(fused)
