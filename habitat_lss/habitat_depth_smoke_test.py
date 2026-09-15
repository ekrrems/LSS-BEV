from __future__ import annotations

import argparse
from pathlib import Path

# Keep OpenCV before PyTorch. This matches the working Habitat trainer and avoids
# a duplicate OpenMP initialization seen with some macOS wheel combinations.
import cv2  # noqa: F401
import torch
from torch.utils.data import DataLoader

from habitat_lss.habitat_lss_model import HabitatLiftSplatShoot
from habitat_lss.habitat_train import (
    TARGET_CHANNELS,
    HabitatBevDataset,
    calculate_loss,
    find_samples,
    move_batch,
    select_device,
)
from habitat_lss.habitat_train_depth import calculate_depth_supervision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/habitat_dataset/scene_102344280"),
    )
    parser.add_argument("--depth-loss-weight", type=float, default=0.20)
    arguments = parser.parse_args()

    device = select_device()
    samples = find_samples(arguments.dataset_root.expanduser().resolve())
    loader = DataLoader(HabitatBevDataset(samples[:1]), batch_size=1, shuffle=False)
    batch = move_batch(next(iter(loader)), device)
    if "depths" not in batch:
        raise KeyError("HabitatBevDataset must return a 'depths' tensor")

    model = HabitatLiftSplatShoot(output_channels=TARGET_CHANNELS).to(device)
    model.train()
    output = model.forward_with_depth(
        batch["images"],
        batch["intrinsics"],
        batch["rotations"],
        batch["translations"],
    )
    bev_loss, _ = calculate_loss(output.logits, batch["target"])
    depth_loss, valid_count, error_sum, near_count = calculate_depth_supervision(
        output.depth_logits,
        batch["depths"],
        depth_minimum=model.depth_minimum,
        depth_maximum=model.depth_maximum,
        depth_step=model.depth_step,
    )
    total_loss = bev_loss + arguments.depth_loss_weight * depth_loss
    total_loss.backward()

    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    possible_count = (
        output.depth_logits.shape[0]
        * output.depth_logits.shape[1]
        * output.depth_logits.shape[3]
        * output.depth_logits.shape[4]
    )
    print(f"Device:              {device}")
    print(f"Images:              {tuple(batch['images'].shape)}")
    print(f"Raw depth:           {tuple(batch['depths'].shape)}")
    print(f"BEV logits:          {tuple(output.logits.shape)}")
    print(f"Depth logits:        {tuple(output.depth_logits.shape)}")
    print(f"Valid depth:         {valid_count / max(1, possible_count):.2%}")
    print(f"BEV loss:            {bev_loss.item():.5f}")
    print(f"Depth loss:          {depth_loss.item():.5f}")
    print(f"Depth MAE:           {error_sum / max(1, valid_count):.3f} m")
    print(f"Within one bin:      {near_count / max(1, valid_count):.2%}")
    print(f"Combined loss:       {total_loss.item():.5f}")
    print(f"Finite gradients:    {finite_gradients}")
    if valid_count == 0:
        raise RuntimeError("No depth values fall inside the model depth range")
    if not finite_gradients:
        raise RuntimeError("Backward pass produced a non-finite gradient")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
