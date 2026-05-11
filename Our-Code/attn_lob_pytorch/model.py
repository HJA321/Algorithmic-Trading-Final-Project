from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Conv2dSame(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], stride: tuple[int, int] = (1, 1)):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_height, input_width = inputs.shape[-2:]
        stride_height, stride_width = self.stride
        kernel_height, kernel_width = self.kernel_size

        out_height = math.ceil(input_height / stride_height)
        out_width = math.ceil(input_width / stride_width)

        pad_height = max((out_height - 1) * stride_height + kernel_height - input_height, 0)
        pad_width = max((out_width - 1) * stride_width + kernel_width - input_width, 0)

        if pad_height or pad_width:
            inputs = F.pad(
                inputs,
                [pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2],
            )
        return self.conv(inputs)


class MaxPool2dSame(nn.Module):
    def __init__(self, kernel_size: tuple[int, int], stride: tuple[int, int]):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_height, input_width = inputs.shape[-2:]
        stride_height, stride_width = self.stride
        kernel_height, kernel_width = self.kernel_size

        out_height = math.ceil(input_height / stride_height)
        out_width = math.ceil(input_width / stride_width)

        pad_height = max((out_height - 1) * stride_height + kernel_height - input_height, 0)
        pad_width = max((out_width - 1) * stride_width + kernel_width - input_width, 0)

        if pad_height or pad_width:
            inputs = F.pad(
                inputs,
                [pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2],
                value=float("-inf"),
            )
        return self.pool(inputs)


class AttnLOBEncoder(nn.Module):
    def __init__(
        self,
        time_window: int = 50,
        embedding_dim: int = 64,
        conv_channels: int = 32,
        inception_channels: int = 64,
        num_heads: int = 10,
        head_dim: int = 16,
    ):
        super().__init__()
        self.time_window = time_window
        self.embedding_dim = embedding_dim
        self.conv_channels = conv_channels
        self.inception_channels = inception_channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        attn_dim = num_heads * head_dim
        inception_dim = 3 * inception_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            Conv2dSame(conv_channels, conv_channels, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            Conv2dSame(conv_channels, conv_channels, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(1, 5), stride=(1, 5)),
            nn.LeakyReLU(negative_slope=0.01),
            Conv2dSame(conv_channels, conv_channels, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            Conv2dSame(conv_channels, conv_channels, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(1, 4)),
            nn.LeakyReLU(negative_slope=0.01),
            Conv2dSame(conv_channels, conv_channels, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            Conv2dSame(conv_channels, conv_channels, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.inception_branch_1 = nn.Sequential(
            Conv2dSame(conv_channels, inception_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            Conv2dSame(inception_channels, inception_channels, kernel_size=(3, 1)),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.inception_branch_2 = nn.Sequential(
            Conv2dSame(conv_channels, inception_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            Conv2dSame(inception_channels, inception_channels, kernel_size=(5, 1)),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.inception_branch_3 = nn.Sequential(
            MaxPool2dSame(kernel_size=(3, 1), stride=(1, 1)),
            Conv2dSame(conv_channels, inception_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.query_projection = nn.Linear(inception_dim, attn_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            kdim=inception_dim,
            vdim=inception_dim,
            batch_first=True,
        )
        self.output_projection = nn.Linear(attn_dim, embedding_dim)

    def forward(self, inputs: torch.Tensor, return_attention: bool = False):
        features = self.feature_extractor(inputs)

        branch_1 = self.inception_branch_1(features)
        branch_2 = self.inception_branch_2(features)
        branch_3 = self.inception_branch_3(features)
        features = torch.cat([branch_1, branch_2, branch_3], dim=1)

        sequence = features.squeeze(-1).transpose(1, 2)
        query = self.query_projection(sequence[:, -1:, :])
        attention_output, attention_weights = self.attention(query=query, key=sequence, value=sequence, need_weights=return_attention)
        embedding = self.output_projection(attention_output).flatten(start_dim=1)

        if return_attention:
            return embedding, attention_weights
        return embedding


class AttnLOBClassifier(nn.Module):
    def __init__(
        self,
        time_window: int = 50,
        embedding_dim: int = 64,
        conv_channels: int = 32,
        inception_channels: int = 64,
        num_heads: int = 10,
        head_dim: int = 16,
        num_classes: int = 3,
        aux_feature_dim: int = 0,
    ):
        super().__init__()
        self.encoder = AttnLOBEncoder(
            time_window=time_window,
            embedding_dim=embedding_dim,
            conv_channels=conv_channels,
            inception_channels=inception_channels,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.aux_feature_dim = aux_feature_dim
        if aux_feature_dim > 0:
            self.aux_projection = nn.Sequential(
                nn.LayerNorm(aux_feature_dim),
                nn.Linear(aux_feature_dim, embedding_dim),
                nn.LeakyReLU(negative_slope=0.01),
            )
            classifier_input_dim = embedding_dim * 2
        else:
            self.aux_projection = None
            classifier_input_dim = embedding_dim
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

    def forward(
        self,
        inputs: torch.Tensor,
        aux_features: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        if return_attention:
            embedding, attention_weights = self.encoder(inputs, return_attention=True)
            logits = self.classifier(self._merge_features(embedding, aux_features))
            return logits, attention_weights
        embedding = self.encoder(inputs)
        return self.classifier(self._merge_features(embedding, aux_features))

    def _merge_features(self, embedding: torch.Tensor, aux_features: torch.Tensor | None) -> torch.Tensor:
        if self.aux_projection is None:
            return embedding
        if aux_features is None:
            raise ValueError("aux_features must be provided when aux_feature_dim > 0")
        aux_embedding = self.aux_projection(aux_features.to(dtype=embedding.dtype))
        return torch.cat([embedding, aux_embedding], dim=1)
