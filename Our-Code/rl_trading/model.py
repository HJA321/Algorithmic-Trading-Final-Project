from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from attn_lob_pytorch.model import AttnLOBEncoder

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
    from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    class BaseFeaturesExtractor(nn.Module):
        def __init__(self, observation_space, features_dim: int):
            super().__init__()
            self.features_dim = features_dim

    create_mlp = None
    DQNPolicy = None
    QNetwork = None


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility
        return torch.load(path, map_location="cpu")


def _read_encoder_hyperparameters(metadata_path: Path) -> dict[str, int]:
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hyperparameters = metadata.get("model", {}).get("hyperparameters", {})
    allowed = {"embedding_dim", "conv_channels", "inception_channels", "num_heads", "head_dim"}
    return {key: int(value) for key, value in hyperparameters.items() if key in allowed}


class MultiInputLOBExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        time_window: int,
        encoder_path: str | None = None,
        encoder_metadata_path: str | None = None,
        freeze_encoder: bool = True,
        output_dim: int = 128,
        embedding_dim: int | None = None,
        conv_channels: int | None = None,
        inception_channels: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
    ):
        super().__init__(observation_space, features_dim=output_dim)
        checkpoint_path = Path(encoder_path) if encoder_path else None
        metadata_path = (
            Path(encoder_metadata_path)
            if encoder_metadata_path
            else checkpoint_path.parent / "metadata.json"
            if checkpoint_path is not None
            else None
        )
        encoder_kwargs = _read_encoder_hyperparameters(metadata_path) if metadata_path is not None else {}
        explicit_kwargs = {
            "embedding_dim": embedding_dim,
            "conv_channels": conv_channels,
            "inception_channels": inception_channels,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }
        encoder_kwargs.update({key: value for key, value in explicit_kwargs.items() if value is not None})
        encoder_kwargs["time_window"] = time_window

        self.encoder = AttnLOBEncoder(**encoder_kwargs)
        if encoder_path:
            if checkpoint_path is not None and checkpoint_path.exists():
                state_dict = _load_checkpoint(checkpoint_path)
                self.encoder.load_state_dict(state_dict, strict=True)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        market_dim = int(observation_space["market_state"].shape[0])
        agent_dim = int(observation_space["agent_state"].shape[0])
        fusion_dim = self.encoder.embedding_dim + market_dim + agent_dim
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


if QNetwork is not None and DQNPolicy is not None and create_mlp is not None:
    class DuelingQNetwork(QNetwork):
        def __init__(
            self,
            observation_space,
            action_space,
            features_extractor,
            features_dim: int,
            net_arch: list[int] | None = None,
            activation_fn: type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
        ) -> None:
            super().__init__(
                observation_space=observation_space,
                action_space=action_space,
                features_extractor=features_extractor,
                features_dim=features_dim,
                net_arch=net_arch,
                activation_fn=activation_fn,
                normalize_images=normalize_images,
            )
            if net_arch is None:
                net_arch = [64, 64]
            action_dim = int(self.action_space.n)
            del self.q_net
            self.value_net = nn.Sequential(*create_mlp(features_dim, 1, net_arch, activation_fn))
            self.advantage_net = nn.Sequential(*create_mlp(features_dim, action_dim, net_arch, activation_fn))

        def forward(self, obs) -> torch.Tensor:
            features = self.extract_features(obs, self.features_extractor)
            value = self.value_net(features)
            advantage = self.advantage_net(features)
            return value + advantage - advantage.mean(dim=1, keepdim=True)


    class DuelingDQNPolicy(DQNPolicy):
        def make_q_net(self) -> DuelingQNetwork:
            net_args = self._update_features_extractor(self.net_args, features_extractor=None)
            return DuelingQNetwork(**net_args).to(self.device)

else:  # pragma: no cover - only used when stable-baselines3 is unavailable
    class DuelingDQNPolicy:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("stable-baselines3 is required for DuelingDQNPolicy")
