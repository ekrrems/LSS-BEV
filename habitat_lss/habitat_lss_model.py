from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lss.image_encoder import LssImageEncoder
from lss.lift import LiftGeometry


@dataclass
class HabitatLssOutput:
    """Complete model output used during depth-supervised training."""

    logits: torch.Tensor
    depth_logits: torch.Tensor
    depth_probabilities: torch.Tensor
    bev_features: torch.Tensor


class BevDecoder(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 96, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, output_channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class HabitatLiftSplatShoot(nn.Module):
    """Six RGB cameras to an agent-relative BEV prediction."""

    def __init__(
        self,
        *,
        image_height: int = 256,
        image_width: int = 352,
        context_channels: int = 64,
        base_channels: int = 32,
        output_channels: int = 14,
        depth_minimum: float = 0.2,
        depth_maximum: float = 7.5,
        depth_step: float = 0.25,
        right_minimum: float = -5.0,
        right_maximum: float = 5.0,
        forward_minimum: float = -5.0,
        forward_maximum: float = 5.0,
        resolution: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.depth_minimum = depth_minimum
        self.depth_maximum = depth_maximum
        self.depth_step = depth_step
        self.right_minimum = right_minimum
        self.right_maximum = right_maximum
        self.forward_minimum = forward_minimum
        self.forward_maximum = forward_maximum
        self.resolution = resolution

        self.forward_cells = int(
            round((forward_maximum - forward_minimum) / resolution)
        )
        self.right_cells = int(round((right_maximum - right_minimum) / resolution))
        depth_bins = int(
            torch.arange(depth_minimum, depth_maximum, depth_step).numel()
        )

        self.image_encoder = LssImageEncoder(
            depth_bins=depth_bins,
            context_channels=context_channels,
            base_channels=base_channels,
        )
        self.lift_geometry = LiftGeometry(
            depth_minimum=depth_minimum,
            depth_maximum=depth_maximum,
            depth_step=depth_step,
        )
        self.bev_decoder = BevDecoder(
            input_channels=context_channels,
            output_channels=output_channels,
        )

    def camera_to_agent(
        self,
        points_camera_cv: torch.Tensor,
        camera_rotations: torch.Tensor,
        camera_translations: torch.Tensor,
    ) -> torch.Tensor:
        # LiftGeometry reconstructs conventional CV coordinates:
        # +X right, +Y down, +Z forward. Habitat camera transforms expect:
        # +X right, +Y up, -Z forward.
        points_camera_habitat = torch.stack(
            (
                points_camera_cv[..., 0],
                -points_camera_cv[..., 1],
                -points_camera_cv[..., 2],
            ),
            dim=-1,
        )
        points_agent = torch.einsum(
            "bnij,bndhwj->bndhwi",
            camera_rotations,
            points_camera_habitat,
        )
        return points_agent + camera_translations[:, :, None, None, None, :]

    def pool_to_bev(
        self,
        points_agent: torch.Tensor,
        lifted_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = points_agent.shape[0]
        context_channels = lifted_features.shape[-1]
        points_flat = points_agent.reshape(-1, 3)
        features_flat = lifted_features.reshape(-1, context_channels)

        points_per_batch = (
            points_agent.shape[1]
            * points_agent.shape[2]
            * points_agent.shape[3]
            * points_agent.shape[4]
        )
        batch_indices = torch.arange(
            batch_size,
            device=points_agent.device,
            dtype=torch.long,
        ).repeat_interleave(points_per_batch)

        right = points_flat[:, 0]
        height = points_flat[:, 1]
        forward = -points_flat[:, 2]
        forward_indices = torch.floor(
            (forward - self.forward_minimum) / self.resolution
        ).long()
        right_indices = torch.floor(
            (right - self.right_minimum) / self.resolution
        ).long()

        valid = (
            torch.isfinite(points_flat).all(dim=1)
            & torch.isfinite(features_flat).all(dim=1)
            & (forward_indices >= 0)
            & (forward_indices < self.forward_cells)
            & (right_indices >= 0)
            & (right_indices < self.right_cells)
            & (height >= -0.25)
            & (height <= 2.0)
        )
        forward_indices = forward_indices[valid]
        right_indices = right_indices[valid]
        valid_batches = batch_indices[valid]
        valid_features = features_flat[valid]
        linear_indices = (
            valid_batches * self.forward_cells * self.right_cells
            + forward_indices * self.right_cells
            + right_indices
        )
        total_cells = batch_size * self.forward_cells * self.right_cells

        pooled = torch.zeros(
            total_cells,
            context_channels,
            device=lifted_features.device,
            dtype=lifted_features.dtype,
        ).index_add(0, linear_indices, valid_features)
        counts = torch.zeros(
            total_cells,
            1,
            device=lifted_features.device,
            dtype=lifted_features.dtype,
        ).index_add(
            0,
            linear_indices,
            torch.ones(
                len(linear_indices),
                1,
                device=lifted_features.device,
                dtype=lifted_features.dtype,
            ),
        )
        pooled = pooled / counts.clamp_min(1.0)
        return pooled.reshape(
            batch_size,
            self.forward_cells,
            self.right_cells,
            context_channels,
        ).permute(0, 3, 1, 2).contiguous()

    def forward_with_depth(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_rotations: torch.Tensor,
        camera_translations: torch.Tensor,
    ) -> HabitatLssOutput:
        encoded = self.image_encoder(images)
        lifted = self.lift_geometry(
            encoded,
            intrinsics,
            image_height=self.image_height,
            image_width=self.image_width,
        )
        points_agent = self.camera_to_agent(
            lifted.points_camera,
            camera_rotations,
            camera_translations,
        )
        bev_features = self.pool_to_bev(points_agent, lifted.features)
        return HabitatLssOutput(
            logits=self.bev_decoder(bev_features),
            depth_logits=encoded.depth_logits,
            depth_probabilities=encoded.depth_probabilities,
            bev_features=bev_features,
        )

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_rotations: torch.Tensor,
        camera_translations: torch.Tensor,
    ) -> torch.Tensor:
        # Preserve the original API for existing inference and preview code.
        return self.forward_with_depth(
            images,
            intrinsics,
            camera_rotations,
            camera_translations,
        ).logits
