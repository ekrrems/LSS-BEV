import torch
from torch import nn


class BevDecoder(nn.Module):
	"""Convert pooled BEV features into segmentation logits."""

	def __init__(
		self,
		*,
		input_channels: int = 64,
		hidden_channels: int = 96,
		output_channels: int = 1,
	) -> None:
		super().__init__()

		self.network = nn.Sequential(
			nn.Conv2d(
				input_channels,
				hidden_channels,
				kernel_size=3,
				padding=1,
				bias=False,
			),
			nn.GroupNorm(
				num_groups=8,
				num_channels=hidden_channels,
			),
			nn.SiLU(inplace=True),

			nn.Conv2d(
				hidden_channels,
				hidden_channels,
				kernel_size=3,
				padding=1,
				bias=False,
			),
			nn.GroupNorm(
				num_groups=8,
				num_channels=hidden_channels,
			),
			nn.SiLU(inplace=True),

			nn.Conv2d(
				hidden_channels,
				output_channels,
				kernel_size=1,
			),
		)

	def forward(
		self,
		bev_features: torch.Tensor,
	) -> torch.Tensor:
		"""
		bev_features:
		    [B, C, X, Y]

		return:
		    [B, output_channels, X, Y]
		"""

		if bev_features.ndim != 4:
			raise ValueError(
				"bev_features must have shape [B, C, X, Y]"
			)

		return self.network(bev_features)