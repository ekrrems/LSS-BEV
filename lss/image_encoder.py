from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormActivation(nn.Sequential):
	"""Convolution followed by batch normalization and SiLU."""

	def __init__(
		self,
		input_channels: int,
		output_channels: int,
		*,
		kernel_size: int = 3,
		stride: int = 1,
	) -> None:
		padding = kernel_size // 2
		super().__init__(
			nn.Conv2d(
				input_channels,
				output_channels,
				kernel_size=kernel_size,
				stride=stride,
				padding=padding,
				bias=False,
			),
			nn.BatchNorm2d(output_channels),
			nn.SiLU(inplace=True),
		)

class ResidualBlock(nn.Module):
	"""A small residual block used by the image backbone."""

	def __init__(
		self,
		input_channels: int,
		output_channels: int,
		*,
		stride: int = 1,
	) -> None:
		super().__init__()
		self.convolutions = nn.Sequential(
			ConvNormActivation(
				input_channels,
				output_channels,
				stride=stride,
			),
			nn.Conv2d(
				output_channels,
				output_channels,
				kernel_size=3,
				padding=1,
				bias=False,
			),
			nn.BatchNorm2d(output_channels),
		)

		if stride != 1 or input_channels != output_channels:
			self.skip = nn.Sequential(
				nn.Conv2d(
					input_channels,
					output_channels,
					kernel_size=1,
					stride=stride,
					bias=False,
				),
				nn.BatchNorm2d(output_channels),
			)
		else:
			self.skip = nn.Identity()

		self.activation = nn.SiLU(inplace=True)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.activation(
			self.convolutions(inputs) + self.skip(inputs)
		)


class UpFuseBlock(nn.Module):
	"""Fuse deep semantics with a higher-resolution lateral feature map."""

	def __init__(
		self,
		lateral_channels: int,
		deep_channels: int,
		output_channels: int,
	) -> None:
		super().__init__()
		self.fusion = nn.Sequential(
			ConvNormActivation(
				lateral_channels + deep_channels,
				output_channels,
			),
			ConvNormActivation(
				output_channels,
				output_channels,
			),
		)

	def forward(
		self,
		lateral: torch.Tensor,
		deep: torch.Tensor,
	) -> torch.Tensor:
		deep = F.interpolate(
			deep,
			size=lateral.shape[-2:],
			mode="bilinear",
			align_corners=False,
		)
		return self.fusion(torch.cat((lateral, deep), dim=1))


@dataclass
class ImageEncoderOutput:
	"""Outputs kept separate until the Lift operation.

	Shapes use B=batch, N=cameras, D=depth bins, C=context channels.
	"""

	context: torch.Tensor
	depth_logits: torch.Tensor
	depth_probabilities: torch.Tensor

	def lifted_features(self) -> torch.Tensor:
		"""Return `[B, N, D, C, Hf, Wf]` weighted frustum features."""
		return (
			self.depth_probabilities.unsqueeze(3)
			* self.context.unsqueeze(2)
		)


class LssImageEncoder(nn.Module):
	"""Lightweight U-Net-style image encoder for Lift-Splat-Shoot.

	The same weights process every camera. The network keeps a 1/8-resolution
	lateral feature and fuses it with a 1/16-resolution semantic feature. Two
	heads then predict a context vector and a categorical depth distribution at
	every feature location.
	"""

	output_downsample = 8

	def __init__(
		self,
		*,
		depth_bins: int = 41,
		context_channels: int = 64,
		base_channels: int = 32,
	) -> None:
		super().__init__()
		if depth_bins <= 1:
			raise ValueError("depth_bins must be larger than one")
		if context_channels <= 0 or base_channels <= 0:
			raise ValueError("channel counts must be positive")

		self.depth_bins = depth_bins
		self.context_channels = context_channels

		self.stem = ConvNormActivation(
			3,
			base_channels,
			kernel_size=5,
			stride=2,
		)
		self.stage_1 = ResidualBlock(
			base_channels,
			base_channels * 2,
			stride=2,
		)
		self.stage_2 = ResidualBlock(
			base_channels * 2,
			base_channels * 4,
			stride=2,
		)
		self.stage_3 = ResidualBlock(
			base_channels * 4,
			base_channels * 8,
			stride=2,
		)

		self.up_fuse = UpFuseBlock(
			lateral_channels=base_channels * 4,
			deep_channels=base_channels * 8,
			output_channels=base_channels * 4,
		)
		self.prediction_head = nn.Conv2d(
			base_channels * 4,
			depth_bins + context_channels,
			kernel_size=1,
		)

	def forward(self, images: torch.Tensor) -> ImageEncoderOutput:
		"""Encode RGB images shaped `[B, N, 3, H, W]`.

		Images should already be converted to floating point and normalized.
		"""
		if images.ndim != 5 or images.shape[2] != 3:
			raise ValueError(
				"images must have shape [batch, cameras, 3, height, width]"
			)

		batch_size, camera_count, _, height, width = images.shape
		if height % 16 != 0 or width % 16 != 0:
			raise ValueError("image height and width must be divisible by 16")

		# Treat cameras as extra batch items so all cameras share one CNN.
		features = images.reshape(batch_size * camera_count, 3, height, width)
		# print(f"HERE ARE THE FEATURES ==> {features.shape}")
		features = self.stem(features)
		features = self.stage_1(features)
		lateral = self.stage_2(features)
		deep = self.stage_3(lateral)
		fused = self.up_fuse(lateral, deep)
		prediction = self.prediction_head(fused)

		depth_logits = prediction[:, : self.depth_bins]
		context = prediction[:, self.depth_bins :]
		depth_probabilities = torch.softmax(depth_logits, dim=1)

		feature_height, feature_width = context.shape[-2:]
		context = context.reshape(
			batch_size,
			camera_count,
			self.context_channels,
			feature_height,
			feature_width,
		)
		depth_logits = depth_logits.reshape(
			batch_size,
			camera_count,
			self.depth_bins,
			feature_height,
			feature_width,
		)
		depth_probabilities = depth_probabilities.reshape_as(depth_logits)

		return ImageEncoderOutput(
			context=context,
			depth_logits=depth_logits,
			depth_probabilities=depth_probabilities,
		)
