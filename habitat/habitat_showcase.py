from __future__ import annotations

import argparse
import json
from pathlib import Path

# Preserve the working macOS native-library import order.
import cv2  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

from habitat.habitat_training_diagnostics import split_by_stratified_blocks
from habitat_lss.habitat_lss_model import HabitatLiftSplatShoot
from habitat_lss.habitat_train import (
    CAMERAS,
    TARGET_CHANNELS,
    HabitatBevDataset,
    find_samples,
    move_batch,
    select_device,
)
from habitat_lss.habitat_train_depth import sample_depth_at_feature_locations


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained Habitat LSS checkpoint."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/habitat_dataset/scene_102344280"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/habitat_lss_depth/habitat_lss_depth_best_iou.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/habitat_lss_depth/showcase"),
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Subset to report and visualize. Threshold tuning always uses validation.",
    )
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-block-size", type=int, default=25)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold. If omitted, select it using validation IoU.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[HabitatLiftSplatShoot, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = HabitatLiftSplatShoot(output_channels=TARGET_CHANNELS).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def make_datasets(
    dataset_root: Path,
    *,
    seed: int,
    block_size: int,
) -> tuple[HabitatBevDataset, HabitatBevDataset]:
    samples = find_samples(dataset_root)
    _, validation_samples, test_samples = split_by_stratified_blocks(
        samples,
        seed=seed,
        block_size=block_size,
    )
    return HabitatBevDataset(validation_samples), HabitatBevDataset(test_samples)


def collect_occupancy(
    model: HabitatLiftSplatShoot,
    dataset: HabitatBevDataset,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    probabilities: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    observed_masks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(
                batch["images"],
                batch["intrinsics"],
                batch["rotations"],
                batch["translations"],
            )
            probabilities.append(torch.sigmoid(logits[:, 0]).cpu().numpy())
            truths.append((batch["target"][:, 0] >= 0.5).cpu().numpy())
            observed_masks.append((batch["target"][:, 2] >= 0.5).cpu().numpy())
    return (
        np.concatenate(probabilities),
        np.concatenate(truths),
        np.concatenate(observed_masks),
    )


def binary_metrics(
    probabilities: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    prediction = (probabilities >= threshold) & observed
    truth = truth & observed
    true_positive = int(np.count_nonzero(prediction & truth))
    false_positive = int(np.count_nonzero(prediction & ~truth))
    false_negative = int(np.count_nonzero(~prediction & truth))
    denominator = true_positive + false_positive + false_negative
    return {
        "threshold": threshold,
        "iou": true_positive / max(1, denominator),
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, true_positive + false_negative),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def choose_validation_threshold(
    model: HabitatLiftSplatShoot,
    validation_dataset: HabitatBevDataset,
    device: torch.device,
    output_directory: Path,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[float, list[dict[str, float | int]]]:
    probabilities, truth, observed = collect_occupancy(
        model,
        validation_dataset,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    thresholds = np.arange(0.10, 0.81, 0.05)
    results = [
        binary_metrics(probabilities, truth, observed, float(threshold))
        for threshold in thresholds
    ]
    best = max(results, key=lambda result: float(result["iou"]))

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        thresholds,
        [result["iou"] for result in results],
        marker="o",
        label="IoU",
    )
    axis.plot(
        thresholds,
        [result["precision"] for result in results],
        marker="o",
        label="Precision",
    )
    axis.plot(
        thresholds,
        [result["recall"] for result in results],
        marker="o",
        label="Recall",
    )
    axis.axvline(float(best["threshold"]), color="black", linestyle="--")
    axis.set(xlabel="Occupancy threshold", ylabel="Score", ylim=(0, 1))
    axis.set_title("Validation threshold selection")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "validation_threshold_sweep.png", dpi=160)
    plt.close(figure)
    return float(best["threshold"]), results


def plot_history(history_path: Path, output_path: Path) -> None:
    if not history_path.exists():
        print(f"History not found; skipped learning curves: {history_path}")
        return
    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)
    if not history:
        return
    epochs = [record["epoch"] for record in history]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, [r["train_iou"] for r in history], label="train")
    axes[0].plot(epochs, [r["validation_iou"] for r in history], label="validation")
    axes[0].set(title="Occupancy IoU", xlabel="Epoch", ylabel="IoU")
    axes[0].legend()

    axes[1].plot(epochs, [r["train_bev_loss"] for r in history], label="train")
    axes[1].plot(epochs, [r["validation_bev_loss"] for r in history], label="validation")
    axes[1].set(title="BEV loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()

    axes[2].plot(
        epochs,
        [r["validation_depth_mae"] for r in history],
        label="validation",
    )
    axes[2].set(title="Depth MAE", xlabel="Epoch", ylabel="Metres")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_sample_panel(
    model: HabitatLiftSplatShoot,
    dataset: HabitatBevDataset,
    dataset_index: int,
    device: torch.device,
    output_path: Path,
    *,
    threshold: float,
) -> dict[str, float | int]:
    sample = dataset[dataset_index]
    batch = {
        name: value.unsqueeze(0).to(device=device, dtype=torch.float32)
        for name, value in sample.items()
    }
    with torch.inference_mode():
        output = model.forward_with_depth(
            batch["images"],
            batch["intrinsics"],
            batch["rotations"],
            batch["translations"],
        )
    occupancy_probability = torch.sigmoid(output.logits[0, 0]).cpu().numpy()
    target = batch["target"][0].cpu().numpy()
    truth = target[0] >= 0.5
    observed = target[2] >= 0.5
    prediction = (occupancy_probability >= threshold) & observed
    metrics = binary_metrics(
        occupancy_probability[None],
        truth[None],
        observed[None],
        threshold,
    )

    errors = np.zeros_like(truth, dtype=np.uint8)
    errors[prediction & truth] = 1
    errors[prediction & ~truth] = 2
    errors[~prediction & truth & observed] = 3
    error_cmap = ListedColormap(["#202020", "#2fb344", "#e03131", "#1c7ed6"])

    depth_probabilities = output.depth_probabilities[0, 1]
    depth_values = model.depth_minimum + model.depth_step * torch.arange(
        depth_probabilities.shape[0],
        device=device,
        dtype=depth_probabilities.dtype,
    )
    predicted_depth = (
        depth_probabilities * depth_values[:, None, None]
    ).sum(dim=0).cpu().numpy()
    sampled_depth = sample_depth_at_feature_locations(
        batch["depths"],
        depth_probabilities.shape[-2],
        depth_probabilities.shape[-1],
    )[0, 1].cpu().numpy()

    figure, axes = plt.subplots(3, 4, figsize=(17, 12))
    for camera_index, camera_name in enumerate(CAMERAS):
        axis = axes.flat[camera_index]
        image = batch["images"][0, camera_index].permute(1, 2, 0).cpu().numpy()
        axis.imshow(image)
        axis.set_title(camera_name.replace("camera_", ""))

    axes[1, 2].imshow(truth, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("Target occupancy")
    probability_image = axes[1, 3].imshow(
        occupancy_probability,
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    axes[1, 3].set_title("Predicted occupancy probability")
    figure.colorbar(probability_image, ax=axes[1, 3], fraction=0.046)

    axes[2, 0].imshow(prediction, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[2, 0].set_title(f"Thresholded occupancy ({threshold:.2f})")
    axes[2, 1].imshow(errors, origin="lower", cmap=error_cmap, vmin=0, vmax=3)
    axes[2, 1].set_title("Occupancy errors")
    axes[2, 1].legend(
        handles=[
            Patch(color="#2fb344", label="TP"),
            Patch(color="#e03131", label="FP"),
            Patch(color="#1c7ed6", label="FN"),
        ],
        loc="lower right",
    )
    axes[2, 2].imshow(
        sampled_depth,
        cmap="turbo",
        vmin=model.depth_minimum,
        vmax=model.depth_maximum,
    )
    axes[2, 2].set_title("Front ground-truth depth")
    axes[2, 3].imshow(
        predicted_depth,
        cmap="turbo",
        vmin=model.depth_minimum,
        vmax=model.depth_maximum,
    )
    axes[2, 3].set_title("Front RGB-predicted depth")
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(
        f"Sample {dataset_index} — IoU {metrics['iou']:.3f}, "
        f"precision {metrics['precision']:.3f}, recall {metrics['recall']:.3f}",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return metrics


def main() -> None:
    arguments = parse_arguments()
    if arguments.samples < 1 or arguments.batch_size < 1:
        raise ValueError("--samples and --batch-size must be positive")
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.checkpoint.expanduser().resolve()
    device = select_device()
    model, checkpoint = load_model(checkpoint_path, device)
    validation_dataset, test_dataset = make_datasets(
        arguments.dataset_root.expanduser().resolve(),
        seed=arguments.seed,
        block_size=arguments.split_block_size,
    )
    selected_dataset = validation_dataset if arguments.split == "validation" else test_dataset

    plot_history(
        checkpoint_path.parent / "training_history_depth.json",
        output_directory / "learning_curves.png",
    )
    if arguments.threshold is None:
        threshold, threshold_results = choose_validation_threshold(
            model,
            validation_dataset,
            device,
            output_directory,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
        )
    else:
        if not 0.0 < arguments.threshold < 1.0:
            raise ValueError("--threshold must be between 0 and 1")
        threshold = arguments.threshold
        threshold_results = []

    probabilities, truth, observed = collect_occupancy(
        model,
        selected_dataset,
        device,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
    )
    aggregate = binary_metrics(probabilities, truth, observed, threshold)
    indices = np.linspace(
        0,
        len(selected_dataset) - 1,
        min(arguments.samples, len(selected_dataset)),
        dtype=int,
    )
    sample_results = []
    for number, dataset_index in enumerate(indices):
        metrics = save_sample_panel(
            model,
            selected_dataset,
            int(dataset_index),
            device,
            output_directory / f"{arguments.split}_sample_{number:02d}.png",
            threshold=threshold,
        )
        sample_results.append({"dataset_index": int(dataset_index), **metrics})

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_validation_iou": float(checkpoint.get("validation_iou", -1.0)),
        "split": arguments.split,
        "samples_in_split": len(selected_dataset),
        "selected_threshold": threshold,
        "aggregate": aggregate,
        "validation_threshold_sweep": threshold_results,
        "displayed_samples": sample_results,
    }
    with (output_directory / "showcase_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Checkpoint epoch: {summary['checkpoint_epoch']}")
    print(f"Selected validation threshold: {threshold:.2f}")
    print(f"{arguments.split.title()} samples: {len(selected_dataset)}")
    print(f"IoU:       {aggregate['iou']:.4f}")
    print(f"Precision: {aggregate['precision']:.4f}")
    print(f"Recall:    {aggregate['recall']:.4f}")
    print(f"Saved showcase to: {output_directory}")


if __name__ == "__main__":
    main()
