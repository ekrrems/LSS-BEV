"""Fine-tune Habitat LSS with a navigation-focused BEV loss.

Place this file in PROJECT_ROOT/habitat_lss/ and run it as
``python -m habitat_lss.habitat_train_depth_v2``.  It reuses the existing
training pipeline and checkpoint format; only the BEV objective is replaced.
"""

from __future__ import annotations

# Keep OpenCV before NumPy/PyTorch. Some macOS conda wheels otherwise load a
# second OpenMP runtime and abort before argument parsing begins.
import cv2  # noqa: F401
import torch
from torch.nn import functional as F

from habitat_lss import habitat_train_depth as trainer
from habitat_lss.habitat_train import (
    DENSITY_CHANNEL,
    FREE_SPACE_CHANNEL,
    HEIGHT_CHANNELS,
    OBSERVED_CHANNEL,
    OCCUPANCY_CHANNEL,
    masked_mean,
)


def _soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    probability = torch.sigmoid(logits) * mask
    target = target * mask
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _target_boundary(target: torch.Tensor) -> torch.Tensor:
    """Three-cell morphological boundary around occupied target cells."""
    dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def calculate_navigation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    observed = target[:, OBSERVED_CHANNEL : OBSERVED_CHANNEL + 1]
    occupancy_target = target[:, OCCUPANCY_CHANNEL : OCCUPANCY_CHANNEL + 1]
    free_target = target[:, FREE_SPACE_CHANNEL : FREE_SPACE_CHANNEL + 1]
    vertical_target = target[:, 3:10]

    occupancy_logits = logits[:, OCCUPANCY_CHANNEL : OCCUPANCY_CHANNEL + 1]
    occupancy_bce_map = F.binary_cross_entropy_with_logits(
        occupancy_logits,
        occupancy_target,
        reduction="none",
    )
    # Mistakes around wall and furniture edges are especially important for
    # navigation, so give those cells more influence without changing labels.
    boundary = _target_boundary(occupancy_target) * observed
    occupancy_loss = masked_mean(
        occupancy_bce_map * (1.0 + 1.5 * boundary),
        observed,
    )
    dice_loss = _soft_dice_loss(occupancy_logits, occupancy_target, observed)

    free_loss = masked_mean(
        F.binary_cross_entropy_with_logits(
            logits[:, FREE_SPACE_CHANNEL : FREE_SPACE_CHANNEL + 1],
            free_target,
            reduction="none",
        ),
        observed,
    )
    vertical_loss = masked_mean(
        F.binary_cross_entropy_with_logits(
            logits[:, 3:10],
            vertical_target,
            reduction="none",
        ),
        observed.expand_as(vertical_target),
    )
    observed_loss = F.binary_cross_entropy_with_logits(
        logits[:, OBSERVED_CHANNEL : OBSERVED_CHANNEL + 1],
        observed,
    )

    predicted_heights = torch.sigmoid(logits[:, HEIGHT_CHANNELS])
    target_heights = target[:, HEIGHT_CHANNELS]
    height_loss = masked_mean(
        F.smooth_l1_loss(
            predicted_heights,
            target_heights,
            reduction="none",
        ),
        occupancy_target.expand_as(target_heights),
    )

    predicted_density = torch.sigmoid(
        logits[:, DENSITY_CHANNEL : DENSITY_CHANNEL + 1]
    )
    target_density = target[:, DENSITY_CHANNEL : DENSITY_CHANNEL + 1]
    density_loss = masked_mean(
        F.smooth_l1_loss(
            predicted_density,
            target_density,
            reduction="none",
        ),
        observed,
    )

    # Occupancy is deliberately the dominant term because this is the output
    # consumed by collision avoidance and mapping.
    total_loss = (
        occupancy_loss
        + 0.35 * dice_loss
        + 0.25 * free_loss
        + 0.35 * vertical_loss
        + 0.20 * observed_loss
        + 0.50 * height_loss
        + 0.10 * density_loss
    )
    binary_summary = (
        occupancy_loss + 0.35 * dice_loss + 0.25 * free_loss + 0.35 * vertical_loss
    )
    components = {
        "binary": float(binary_summary.item()),
        "occupancy": float(occupancy_loss.item()),
        "dice": float(dice_loss.item()),
        "free": float(free_loss.item()),
        "vertical": float(vertical_loss.item()),
        "observed": float(observed_loss.item()),
        "height": float(height_loss.item()),
        "density": float(density_loss.item()),
    }
    return total_loss, components


def main() -> None:
    print("Using navigation-focused occupancy + boundary + Dice BEV loss")
    # run_epoch resolves calculate_loss from habitat_train_depth's module
    # globals, so replacing it here preserves the rest of the tested trainer.
    trainer.calculate_loss = calculate_navigation_loss
    trainer.main()


if __name__ == "__main__":
    main()
