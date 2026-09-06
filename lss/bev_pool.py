from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BevPoolOutput:
	# [B, C, X, Y]
	features: torch.Tensor

	# Number of lifted points in each BEV cell:
	# [B, 1, X, Y]
	counts: torch.Tensor


class BevPool(nn.Module):
	def __init__(
		self,
		*,
		x_minimum: float,
		x_maximum: float,
		y_minimum: float,
		y_maximum: float,
		z_minimum: float,
		z_maximum: float,
		resolution: float,
	) -> None:
		super().__init__()

		if x_maximum <= x_minimum:
			raise ValueError(
				"x_maximum must be greater than x_minimum"
			)

		if y_maximum <= y_minimum:
			raise ValueError(
				"y_maximum must be greater than y_minimum"
			)

		if z_maximum <= z_minimum:
			raise ValueError(
				"z_maximum must be greater than z_minimum"
			)

		if resolution <= 0.0:
			raise ValueError(
				"resolution must be positive"
			)

		self.x_minimum = float(x_minimum)
		self.x_maximum = float(x_maximum)

		self.y_minimum = float(y_minimum)
		self.y_maximum = float(y_maximum)

		self.z_minimum = float(z_minimum)
		self.z_maximum = float(z_maximum)

		self.resolution = float(resolution)

		self.x_cells = int(round(
			(self.x_maximum - self.x_minimum)
			/ self.resolution
		))

		self.y_cells = int(round(
			(self.y_maximum - self.y_minimum)
			/ self.resolution
		))

	def forward(
		self,
		points_ego: torch.Tensor,
		lifted_features: torch.Tensor,
	) -> BevPoolOutput:
		"""
		points_ego:
		    [B, N, D, Hf, Wf, 3]

		lifted_features:
		    [B, N, D, Hf, Wf, C]
		"""

		if points_ego.ndim != 6:
			raise ValueError(
				"points_ego must have shape "
				"[B, N, D, Hf, Wf, 3]"
			)

		if lifted_features.ndim != 6:
			raise ValueError(
				"lifted_features must have shape "
				"[B, N, D, Hf, Wf, C]"
			)

		if points_ego.shape[:-1] != lifted_features.shape[:-1]:
			raise ValueError(
				"Point and feature indices do not match"
			)

		if points_ego.shape[-1] != 3:
			raise ValueError(
				"Point coordinates must contain x, y and z"
			)

		batch_size = points_ego.shape[0]
		context_channels = lifted_features.shape[-1]

		# Flatten all cameras, depths and feature locations.
		points_flat = points_ego.reshape(-1, 3)
		features_flat = lifted_features.reshape(
			-1,
			context_channels,
		)

		# Create a batch index for every flattened point.
		points_per_batch = (
			points_ego.shape[1]
			* points_ego.shape[2]
			* points_ego.shape[3]
			* points_ego.shape[4]
		)

		batch_indices = torch.arange(
			batch_size,
			device=points_ego.device,
			dtype=torch.long,
		).repeat_interleave(points_per_batch)

		x = points_flat[:, 0]
		y = points_flat[:, 1]
		z = points_flat[:, 2]

		x_indices = torch.floor(
			(x - self.x_minimum)
			/ self.resolution
		).long()

		y_indices = torch.floor(
			(y - self.y_minimum)
			/ self.resolution
		).long()

		# Remove points outside the chosen BEV area.
		valid = (
			torch.isfinite(points_flat).all(dim=1)
			& torch.isfinite(features_flat).all(dim=1)
			& (x_indices >= 0)
			& (x_indices < self.x_cells)
			& (y_indices >= 0)
			& (y_indices < self.y_cells)
			& (z >= self.z_minimum)
			& (z < self.z_maximum)
		)

		x_indices = x_indices[valid]
		y_indices = y_indices[valid]
		valid_batches = batch_indices[valid]
		valid_features = features_flat[valid]

		# Convert [batch, x, y] into one flat cell index.
		linear_indices = (
			valid_batches
			* self.x_cells
			* self.y_cells
			+ x_indices * self.y_cells
			+ y_indices
		)

		total_cells = (
			batch_size
			* self.x_cells
			* self.y_cells
		)

		bev_flat = torch.zeros(
			total_cells,
			context_channels,
			device=lifted_features.device,
			dtype=lifted_features.dtype,
		)

		# Add all features belonging to the same BEV cell.
		bev_flat = bev_flat.index_add(
			0,
			linear_indices,
			valid_features,
		)

		counts_flat = torch.zeros(
			total_cells,
			1,
			device=lifted_features.device,
			dtype=lifted_features.dtype,
		)

		counts_flat = counts_flat.index_add(
			0,
			linear_indices,
			torch.ones(
				len(linear_indices),
				1,
				device=lifted_features.device,
				dtype=lifted_features.dtype,
			),
		)

		bev_features = bev_flat.reshape(
			batch_size,
			self.x_cells,
			self.y_cells,
			context_channels,
		).permute(
			0,
			3,
			1,
			2,
		).contiguous()

		counts = counts_flat.reshape(
			batch_size,
			self.x_cells,
			self.y_cells,
			1,
		).permute(
			0,
			3,
			1,
			2,
		).contiguous()

		return BevPoolOutput(
			features=bev_features,
			counts=counts,
		)