from __future__ import annotations

import torch
from torch import nn


class CameraToEgo(nn.Module):
	"""Transform camera-frame points into the ego frame."""

	def forward(
		self,
		points_camera: torch.Tensor,
		rotations_ego_camera: torch.Tensor,
		translations_ego_camera: torch.Tensor,
	) -> torch.Tensor:
		"""
		points_camera:
		    [B, N, D, Hf, Wf, 3]

		rotations_ego_camera:
		    [B, N, 3, 3]

		translations_ego_camera:
		    [B, N, 3]
		"""

		if points_camera.ndim != 6:
			raise ValueError(
				"points_camera must have shape "
				"[B, N, D, Hf, Wf, 3]"
			)

		points_ego = torch.einsum(
			"bnij,bndhwj->bndhwi",
			rotations_ego_camera,
			points_camera,
		)

		points_ego = (
			points_ego
			+ translations_ego_camera[
				:,
				:,
				None,
				None,
				None,
				:,
			]
		)

		return points_ego