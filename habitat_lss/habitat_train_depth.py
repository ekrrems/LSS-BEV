from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

# Keep OpenCV before NumPy/PyTorch. This matches habitat_train.py and avoids a
# duplicate OpenMP initialization seen with some macOS wheel combinations.
import cv2  # noqa: F401
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from habitat.habitat_training_diagnostics import (
    LiveValidationPreview,
    audit_splits,
    randomly_limit_samples,
    split_by_stratified_blocks,
)
from habitat_lss.habitat_lss_model import HabitatLiftSplatShoot
from habitat_lss.habitat_train import (
    TARGET_CHANNELS,
    HabitatBevDataset,
    Metrics,
    calculate_loss,
    find_samples,
    metrics_from_counts,
    move_batch,
    select_device,
)


@dataclass
class EpochResult:
    bev: Metrics
    total_loss: float
    depth_loss: float
    depth_mae: float
    depth_within_one_bin: float
    valid_depth_fraction: float


class RgbPhotometricAugmentation(Dataset):
    """Geometry-safe RGB augmentation; depth, calibration, and BEV stay unchanged."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = dict(self.dataset[index])
        images = sample["images"].clone()
        # One lighting transform is shared across cameras to preserve consistency.
        brightness = 0.85 + 0.30 * torch.rand(())
        contrast = 0.85 + 0.30 * torch.rand(())
        gamma = 0.90 + 0.20 * torch.rand(())
        mean = images.mean(dim=(-2, -1), keepdim=True)
        images = (images - mean) * contrast + mean
        images = (images * brightness).clamp(0.0, 1.0).pow(gamma)
        if torch.rand(()) < 0.35:
            noise_scale = 0.015 * torch.rand(())
            images = images + noise_scale * torch.randn_like(images)
        sample["images"] = images.clamp(0.0, 1.0)
        return sample


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Habitat LSS with metric-depth supervision."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/habitat_dataset/scene_102344280"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/habitat_lss_depth"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--depth-loss-weight", type=float, default=0.20)
    parser.add_argument("--split-block-size", type=int, default=25)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--validation-previews", type=int, default=3)
    parser.add_argument(
        "--live-preview",
        action="store_true",
        help="Update fixed validation samples in one GUI window each epoch.",
    )
    parser.add_argument(
        "--keep-preview-open",
        action="store_true",
        help="Keep the best-model preview open after testing until the window closes.",
    )
    parser.add_argument(
        "--augment-rgb",
        action="store_true",
        help="Apply mild brightness, contrast, gamma, and noise augmentation to train RGB.",
    )
    parser.add_argument(
        "--save-best-loss",
        action="store_true",
        help="Also retain the lowest-total-validation-loss checkpoint.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to resume. Starting fresh is recommended when enabling depth.",
    )
    return parser.parse_args()


def sample_depth_at_feature_locations(
    depths: torch.Tensor,
    feature_height: int,
    feature_width: int,
) -> torch.Tensor:
    """Match the exact linspace pixel locations used by LiftGeometry."""
    if depths.ndim != 4:
        raise ValueError(f"depths must be [B, N, H, W], got {tuple(depths.shape)}")
    image_height, image_width = depths.shape[-2:]
    rows = torch.linspace(
        0,
        image_height - 1,
        feature_height,
        device=depths.device,
    ).round().long()
    columns = torch.linspace(
        0,
        image_width - 1,
        feature_width,
        device=depths.device,
    ).round().long()
    return depths.index_select(-2, rows).index_select(-1, columns)


def calculate_depth_supervision(
    depth_logits: torch.Tensor,
    ground_truth_depth: torch.Tensor,
    *,
    depth_minimum: float,
    depth_maximum: float,
    depth_step: float,
) -> tuple[torch.Tensor, int, float, int]:
    """Return CE loss, valid count, absolute-error sum, and near-bin count."""
    if depth_logits.ndim != 5:
        raise ValueError(
            f"depth_logits must be [B, N, D, Hf, Wf], got {tuple(depth_logits.shape)}"
        )
    batch_size, cameras, depth_bins, feature_height, feature_width = (
        depth_logits.shape
    )
    sampled_depth = sample_depth_at_feature_locations(
        ground_truth_depth,
        feature_height,
        feature_width,
    )
    valid = (
        torch.isfinite(sampled_depth)
        & (sampled_depth >= depth_minimum)
        & (sampled_depth < depth_maximum)
    )
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return depth_logits.sum() * 0.0, 0, 0.0, 0

    depth_targets = torch.floor(
        (sampled_depth - depth_minimum) / depth_step
    ).long().clamp(min=0, max=depth_bins - 1)
    depth_targets = depth_targets.masked_fill(~valid, -100)
    depth_loss = F.cross_entropy(
        depth_logits.reshape(
            batch_size * cameras,
            depth_bins,
            feature_height,
            feature_width,
        ),
        depth_targets.reshape(
            batch_size * cameras,
            feature_height,
            feature_width,
        ),
        ignore_index=-100,
    )

    probabilities = torch.softmax(depth_logits, dim=2)
    depth_values = depth_minimum + depth_step * torch.arange(
        depth_bins,
        device=depth_logits.device,
        dtype=depth_logits.dtype,
    )
    predicted_depth = (
        probabilities * depth_values[None, None, :, None, None]
    ).sum(dim=2)
    absolute_error_sum = float(
        (predicted_depth[valid] - sampled_depth[valid]).abs().sum().item()
    )
    predicted_bins = depth_logits.argmax(dim=2)
    within_one_bin = int(
        ((predicted_bins - depth_targets.clamp_min(0)).abs() <= 1)[valid]
        .sum()
        .item()
    )
    return depth_loss, valid_count, absolute_error_sum, within_one_bin


def occupancy_counts_at_threshold(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> tuple[int, int, int]:
    prediction = torch.sigmoid(logits[:, 0]) >= threshold
    truth = target[:, 0] >= 0.5
    observed = target[:, 2] >= 0.5
    prediction &= observed
    truth &= observed
    return (
        int((prediction & truth).sum().item()),
        int((prediction & ~truth).sum().item()),
        int((~prediction & truth).sum().item()),
    )


def run_epoch(
    model: HabitatLiftSplatShoot,
    loader: DataLoader,
    device: torch.device,
    *,
    depth_loss_weight: float,
    prediction_threshold: float,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    epochs: int,
    print_every: int,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    total_loss_sum = 0.0
    bev_loss_sum = 0.0
    depth_loss_sum = 0.0
    batches = 0
    true_positive = false_positive = false_negative = 0
    valid_depth = possible_depth = within_one_bin = 0
    absolute_depth_error = 0.0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            if "depths" not in batch:
                raise KeyError(
                    "Dataset batch has no 'depths'. Use the supplied HabitatBevDataset change."
                )
            if training:
                optimizer.zero_grad(set_to_none=True)

            output = model.forward_with_depth(
                batch["images"],
                batch["intrinsics"],
                batch["rotations"],
                batch["translations"],
            )
            bev_loss, components = calculate_loss(output.logits, batch["target"])
            depth_loss, valid, error_sum, near_count = calculate_depth_supervision(
                output.depth_logits,
                batch["depths"],
                depth_minimum=model.depth_minimum,
                depth_maximum=model.depth_maximum,
                depth_step=model.depth_step,
            )
            total_loss = bev_loss + depth_loss_weight * depth_loss

            if training:
                total_loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0
                )
                optimizer.step()
            else:
                gradient_norm = torch.tensor(0.0)

            tp, fp, fn = occupancy_counts_at_threshold(
                output.logits.detach(),
                batch["target"],
                prediction_threshold,
            )
            batches += 1
            total_loss_sum += float(total_loss.item())
            bev_loss_sum += float(bev_loss.item())
            depth_loss_sum += float(depth_loss.item())
            true_positive += tp
            false_positive += fp
            false_negative += fn
            valid_depth += valid
            possible_depth += (
                output.depth_logits.shape[0]
                * output.depth_logits.shape[1]
                * output.depth_logits.shape[3]
                * output.depth_logits.shape[4]
            )
            absolute_depth_error += error_sum
            within_one_bin += near_count

            if training and (batch_index == 1 or batch_index % print_every == 0):
                print(
                    f"epoch={epoch:03d}/{epochs:03d} "
                    f"batch={batch_index:04d}/{len(loader):04d} "
                    f"total={total_loss.item():.5f} "
                    f"bev={bev_loss.item():.5f} "
                    f"depth={depth_loss.item():.5f} "
                    f"binary={components['binary']:.5f} "
                    f"gradient={float(gradient_norm):.3f}"
                )

    bev_metrics = metrics_from_counts(
        bev_loss_sum,
        batches,
        true_positive,
        false_positive,
        false_negative,
    )
    return EpochResult(
        bev=bev_metrics,
        total_loss=total_loss_sum / max(1, batches),
        depth_loss=depth_loss_sum / max(1, batches),
        depth_mae=absolute_depth_error / max(1, valid_depth),
        depth_within_one_bin=within_one_bin / max(1, valid_depth),
        valid_depth_fraction=valid_depth / max(1, possible_depth),
    )


def build_loaders(arguments: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    all_samples = find_samples(arguments.dataset_root.expanduser().resolve())
    train_samples, validation_samples, test_samples = split_by_stratified_blocks(
        all_samples,
        seed=arguments.seed,
        block_size=arguments.split_block_size,
    )
    train_samples = randomly_limit_samples(
        train_samples, arguments.max_train_samples, seed=arguments.seed
    )
    validation_samples = randomly_limit_samples(
        validation_samples,
        arguments.max_validation_samples,
        seed=arguments.seed + 1,
    )
    test_samples = randomly_limit_samples(
        test_samples, arguments.max_test_samples, seed=arguments.seed + 2
    )
    audit_splits(train_samples, validation_samples, test_samples)
    train_dataset: Dataset = HabitatBevDataset(train_samples)
    if arguments.augment_rgb:
        train_dataset = RgbPhotometricAugmentation(train_dataset)
    datasets = (
        train_dataset,
        HabitatBevDataset(validation_samples),
        HabitatBevDataset(test_samples),
    )
    return tuple(
        DataLoader(
            dataset,
            batch_size=arguments.batch_size,
            shuffle=index == 0,
            num_workers=arguments.num_workers,
        )
        for index, dataset in enumerate(datasets)
    )  # type: ignore[return-value]


def main() -> None:
    arguments = parse_arguments()
    if arguments.batch_size < 1 or arguments.epochs < 1:
        raise ValueError("--batch-size and --epochs must be positive")
    if arguments.depth_loss_weight < 0:
        raise ValueError("--depth-loss-weight cannot be negative")
    if not 0.0 < arguments.prediction_threshold < 1.0:
        raise ValueError("--prediction-threshold must be between 0 and 1")
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    device = select_device()
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, test_loader = build_loaders(arguments)
    model = HabitatLiftSplatShoot(output_channels=TARGET_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.3,
        patience=3,
    )

    start_epoch = 1
    global_step = 0
    best_validation_iou = -1.0
    best_validation_loss = float("inf")
    if arguments.resume is not None:
        checkpoint = torch.load(arguments.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_validation_iou = float(checkpoint.get("best_validation_iou", -1.0))
        best_validation_loss = float(
            checkpoint.get("best_validation_loss", float("inf"))
        )
        print(f"Resumed {arguments.resume} at epoch {start_epoch}")

    first_depth = train_loader.dataset[0]["depths"]
    valid_first_depth = torch.isfinite(first_depth) & (first_depth > 0)
    if not valid_first_depth.any():
        raise RuntimeError("The first sample contains no valid metric depth")
    print(f"Device: {device}")
    print(
        f"Samples: train={len(train_loader.dataset)}, "
        f"validation={len(validation_loader.dataset)}, test={len(test_loader.dataset)}"
    )
    print(
        "Depth check: "
        f"shape={tuple(first_depth.shape)} "
        f"min={first_depth[valid_first_depth].min().item():.3f}m "
        f"max={first_depth[valid_first_depth].max().item():.3f}m "
        f"mean={first_depth[valid_first_depth].mean().item():.3f}m"
    )
    print(
        f"Depth bins: [{model.depth_minimum:.2f}, {model.depth_maximum:.2f}) "
        f"step={model.depth_step:.2f}, weight={arguments.depth_loss_weight:.2f}"
    )
    print(f"RGB augmentation: {'enabled' if arguments.augment_rgb else 'disabled'}")

    live_preview = None
    if arguments.live_preview:
        live_preview = LiveValidationPreview(
            validation_loader.dataset,
            count=arguments.validation_previews,
            threshold=arguments.prediction_threshold,
        )

    history: list[dict[str, float | int]] = []
    for epoch in range(start_epoch, arguments.epochs + 1):
        train_result = run_epoch(
            model,
            train_loader,
            device,
            depth_loss_weight=arguments.depth_loss_weight,
            prediction_threshold=arguments.prediction_threshold,
            optimizer=optimizer,
            epoch=epoch,
            epochs=arguments.epochs,
            print_every=arguments.print_every,
        )
        global_step += len(train_loader)
        validation_result = run_epoch(
            model,
            validation_loader,
            device,
            depth_loss_weight=arguments.depth_loss_weight,
            prediction_threshold=arguments.prediction_threshold,
            optimizer=None,
            epoch=epoch,
            epochs=arguments.epochs,
            print_every=arguments.print_every,
        )
        scheduler.step(validation_result.bev.iou)

        print(f"\nEpoch {epoch} complete")
        print(
            f"  train total/BEV/depth loss: {train_result.total_loss:.5f} / "
            f"{train_result.bev.loss:.5f} / {train_result.depth_loss:.5f}"
        )
        print(f"  train IoU:                 {train_result.bev.iou:.4f}")
        print(
            f"  validation total/BEV/depth: {validation_result.total_loss:.5f} / "
            f"{validation_result.bev.loss:.5f} / {validation_result.depth_loss:.5f}"
        )
        print(f"  validation IoU:            {validation_result.bev.iou:.4f}")
        print(f"  precision:                 {validation_result.bev.precision:.4f}")
        print(f"  recall:                    {validation_result.bev.recall:.4f}")
        print(f"  depth MAE:                 {validation_result.depth_mae:.3f} m")
        print(
            f"  depth within one bin:      "
            f"{validation_result.depth_within_one_bin:.2%}"
        )
        print(
            f"  valid depth coverage:      "
            f"{validation_result.valid_depth_fraction:.2%}"
        )
        print(f"  learning rate:             {optimizer.param_groups[0]['lr']:.2e}")

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_iou": validation_result.bev.iou,
            "validation_bev_loss": validation_result.bev.loss,
            "validation_total_loss": validation_result.total_loss,
            "depth_loss_weight": arguments.depth_loss_weight,
            "prediction_threshold": arguments.prediction_threshold,
            "best_validation_iou": max(
                best_validation_iou, validation_result.bev.iou
            ),
            "best_validation_loss": min(
                best_validation_loss, validation_result.total_loss
            ),
        }
        torch.save(checkpoint, output_directory / "habitat_lss_depth_last.pt")
        if validation_result.bev.iou > best_validation_iou:
            best_validation_iou = validation_result.bev.iou
            torch.save(checkpoint, output_directory / "habitat_lss_depth_best_iou.pt")
            print("  Saved best-IoU checkpoint.")
        if (
            arguments.save_best_loss
            and validation_result.total_loss < best_validation_loss
        ):
            best_validation_loss = validation_result.total_loss
            torch.save(checkpoint, output_directory / "habitat_lss_depth_best_loss.pt")
            print("  Saved best-loss checkpoint.")

        if live_preview is not None:
            live_preview.update(model, device, epoch=epoch)
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_result.total_loss,
                "train_bev_loss": train_result.bev.loss,
                "train_depth_loss": train_result.depth_loss,
                "train_iou": train_result.bev.iou,
                "validation_total_loss": validation_result.total_loss,
                "validation_bev_loss": validation_result.bev.loss,
                "validation_depth_loss": validation_result.depth_loss,
                "validation_depth_mae": validation_result.depth_mae,
                "validation_iou": validation_result.bev.iou,
                "validation_precision": validation_result.bev.precision,
                "validation_recall": validation_result.bev.recall,
            }
        )
        with (output_directory / "training_history_depth.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(history, file, indent=2)

    best_path = output_directory / "habitat_lss_depth_best_iou.pt"
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    if live_preview is not None:
        live_preview.update(
            model,
            device,
            epoch=int(best_checkpoint["epoch"]),
            title_suffix="best-IoU checkpoint",
        )
        live_preview.save(output_directory / "best_validation_preview.png")
    test_result = run_epoch(
        model,
        test_loader,
        device,
        depth_loss_weight=arguments.depth_loss_weight,
        prediction_threshold=arguments.prediction_threshold,
        optimizer=None,
        epoch=best_checkpoint["epoch"],
        epochs=arguments.epochs,
        print_every=arguments.print_every,
    )
    print(f"\nHeld-out test result from best-IoU epoch {best_checkpoint['epoch']}")
    print(f"  total loss: {test_result.total_loss:.5f}")
    print(f"  BEV loss:   {test_result.bev.loss:.5f}")
    print(f"  depth loss: {test_result.depth_loss:.5f}")
    print(f"  depth MAE:  {test_result.depth_mae:.3f} m")
    print(f"  IoU:        {test_result.bev.iou:.4f}")
    print(f"  precision:  {test_result.bev.precision:.4f}")
    print(f"  recall:     {test_result.bev.recall:.4f}")
    if live_preview is not None and arguments.keep_preview_open:
        live_preview.keep_open()


if __name__ == "__main__":
    main()
