from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from lss.image_encoder import ImageEncoderOutput

@dataclass
class LiftOutput:
	# Candidate 3D positions:
	# [B, N, D, Hf, Wf, 3]
	points_camera: torch.Tensor

	# Probability-weighted CNN features:
	# [B, N, D, Hf, Wf, C]
	features: torch.Tensor

	# Actual depth values:
	# [D]
	depth_values: torch.Tensor


class LiftGeometry(nn.Module):
	def __init__(
		self,
		*,
		depth_minimum: float = 4.0,
		depth_maximum: float = 45.0,
		depth_step: float = 1.0,
	) -> None:
		super().__init__()

		if depth_minimum <= 0:
			raise ValueError(
				"depth_minimum must be positive"
			)

		if depth_maximum <= depth_minimum:
			raise ValueError(
				"depth_maximum must be greater than depth_minimum"
			)

		if depth_step <= 0:
			raise ValueError(
				"depth_step must be positive"
			)

		self.depth_minimum = depth_minimum
		self.depth_maximum = depth_maximum
		self.depth_step = depth_step

	def create_frustum(
		self,
		*,
		image_height: int,
		image_width: int,
		feature_height: int,
		feature_width: int,
		device: torch.device,
		dtype: torch.dtype,
	) -> torch.Tensor:
		"""
		Create [D, Hf, Wf, 3].

		The final dimension contains:

		    [u, v, d]
		"""

		depth_values = torch.arange(
			self.depth_minimum,
			self.depth_maximum,
			self.depth_step,
			device=device,
			dtype=dtype,
		)

		# Connect feature-map columns to processed-image columns.
		u_values = torch.linspace(
			0,
			image_width - 1,
			feature_width,
			device=device,
			dtype=dtype,
		)

		# Connect feature-map rows to processed-image rows.
		v_values = torch.linspace(
			0,
			image_height - 1,
			feature_height,
			device=device,
			dtype=dtype,
		)

		d_grid, v_grid, u_grid = torch.meshgrid(
			depth_values,
			v_values,
			u_values,
			indexing="ij",
		)

		frustum = torch.stack(
			(
				u_grid,
				v_grid,
				d_grid,
			),
			dim=-1,
		)

		return frustum

	def forward(
		self,
		encoder_output: ImageEncoderOutput,
		intrinsics: torch.Tensor,
		*,
		image_height: int,
		image_width: int,
	) -> LiftOutput:
		"""
		intrinsics: [B, N, 3, 3]

		The intrinsics must correspond to the processed images,
		not necessarily the original nuScenes image resolution.
		"""

		context = encoder_output.context
		depth_probabilities = (
			encoder_output.depth_probabilities
		)

		batch_size = context.shape[0]
		camera_count = context.shape[1]
		context_channels = context.shape[2]
		feature_height = context.shape[3]
		feature_width = context.shape[4]

		if intrinsics.shape != (
			batch_size,
			camera_count,
			3,
			3,
		):
			raise ValueError(
				"intrinsics must have shape [B, N, 3, 3]"
			)

		frustum = self.create_frustum(
			image_height=image_height,
			image_width=image_width,
			feature_height=feature_height,
			feature_width=feature_width,
			device=context.device,
			dtype=context.dtype,
		)

		depth_count = frustum.shape[0]

		if depth_probabilities.shape[2] != depth_count:
			raise ValueError(
				"The encoder depth-bin count does not match "
				"the Lift depth-bin count"
			)

		# frustum[..., 0] is u.
		u = frustum[..., 0]

		# frustum[..., 1] is v.
		v = frustum[..., 1]

		# frustum[..., 2] is d.
		d = frustum[..., 2]

		pixels_homogeneous = torch.stack(
			(
				u,
				v,
				torch.ones_like(u),
			),
			dim=-1,
		)
		# [D, Hf, Wf, 3]

		inverse_intrinsics = torch.linalg.inv(
			intrinsics
		)
		# [B, N, 3, 3]

		# Apply K^-1 to every [u, v, 1] vector.
		rays_camera = torch.einsum(
			"bnij,dhwj->bndhwi",
			inverse_intrinsics,
			pixels_homogeneous,
		)
		# [B, N, D, Hf, Wf, 3]

		# Evaluate every ray at every candidate depth.
		points_camera = (
			rays_camera
			* d[None, None, ..., None]
		)
		# [B, N, D, Hf, Wf, 3]

		# Give the context tensor a depth dimension:
		#
		# [B, N, C, Hf, Wf]
		# becomes
		# [B, N, 1, C, Hf, Wf]
		context_with_depth = context.unsqueeze(2)

		# Give depth probabilities a context-channel dimension:
		#
		# [B, N, D, Hf, Wf]
		# becomes
		# [B, N, D, 1, Hf, Wf]
		probabilities_with_channels = (
			depth_probabilities.unsqueeze(3)
		)

		weighted_features = (
			probabilities_with_channels
			* context_with_depth
		)
		# [B, N, D, C, Hf, Wf]

		# Reorder it to match the geometry indices.
		weighted_features = weighted_features.permute(
			0,
			1,
			2,
			4,
			5,
			3,
		).contiguous()
		# [B, N, D, Hf, Wf, C]

		depth_values = torch.arange(
			self.depth_minimum,
			self.depth_maximum,
			self.depth_step,
			device=context.device,
			dtype=context.dtype,
		)

		return LiftOutput(
			points_camera=points_camera,
			features=weighted_features,
			depth_values=depth_values,
		)