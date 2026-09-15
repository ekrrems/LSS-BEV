from __future__ import annotations

import argparse
import time
from pathlib import Path

# Match the macOS import order used by the working trainer.
import cv2  # noqa: F401
import numpy as np
import torch

from habitat_lss.habitat_lss_model import HabitatLiftSplatShoot
from habitat_lss.habitat_train import (
    TARGET_CHANNELS,
    HabitatBevDataset,
    find_samples,
    select_device,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark six-camera Habitat LSS inference latency."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/habitat_dataset/scene_102344280"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/habitat_lss_depth_augmented/habitat_lss_depth_best_iou.pt"
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def report(name: str, timings: list[float]) -> None:
    milliseconds = np.asarray(timings, dtype=np.float64) * 1_000.0
    mean_ms = float(milliseconds.mean())
    print(f"\n{name}")
    print(f"  mean:   {mean_ms:8.2f} ms  ({1_000.0 / mean_ms:6.2f} FPS)")
    print(f"  median: {np.percentile(milliseconds, 50):8.2f} ms")
    print(f"  p90:    {np.percentile(milliseconds, 90):8.2f} ms")
    print(f"  p99:    {np.percentile(milliseconds, 99):8.2f} ms")
    print(f"  min:    {milliseconds.min():8.2f} ms")
    print(f"  max:    {milliseconds.max():8.2f} ms")


def to_device(
    sample: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: value.unsqueeze(0).to(device=device, dtype=torch.float32)
        for name, value in sample.items()
    }


def predict(
    model: HabitatLiftSplatShoot,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    return model(
        batch["images"],
        batch["intrinsics"],
        batch["rotations"],
        batch["translations"],
    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.warmup < 1 or arguments.iterations < 1:
        raise ValueError("--warmup and --iterations must be positive")

    device = select_device()
    checkpoint = torch.load(
        arguments.checkpoint.expanduser().resolve(),
        map_location=device,
    )
    model = HabitatLiftSplatShoot(output_channels=TARGET_CHANNELS).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    samples = find_samples(arguments.dataset_root.expanduser().resolve())
    dataset = HabitatBevDataset(samples)
    batch = to_device(dataset[0], device)
    print(f"Device:           {device}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Input shape:      {tuple(batch['images'].shape)}")
    print(f"Iterations:       {arguments.iterations}")

    with torch.inference_mode():
        for _ in range(arguments.warmup):
            output = predict(model, batch)
        synchronize(device)
        if not torch.isfinite(output).all():
            raise RuntimeError("Prediction contains NaN or infinity")

        model_timings: list[float] = []
        for _ in range(arguments.iterations):
            synchronize(device)
            started = time.perf_counter()
            output = predict(model, batch)
            synchronize(device)
            model_timings.append(time.perf_counter() - started)
    report("Model only: six RGB cameras to BEV", model_timings)

    # Disk access is not representative of live Habitat observations, but it
    # exposes preprocessing or dataset-loader bottlenecks separately.
    end_to_end_timings: list[float] = []
    measured = min(arguments.iterations, len(dataset))
    with torch.inference_mode():
        for index in range(measured):
            synchronize(device)
            started = time.perf_counter()
            disk_batch = to_device(dataset[index], device)
            output = predict(model, disk_batch)
            synchronize(device)
            end_to_end_timings.append(time.perf_counter() - started)
    report("Disk loading + preprocessing + model", end_to_end_timings)

    if device.type == "mps":
        memory = torch.mps.current_allocated_memory() / (2**20)
        print(f"\nMPS tensor memory: {memory:.1f} MiB")
    print(
        "\nFor a live Habitat loop, use model-only p90 plus simulator "
        "rendering, mapping, and planning latency."
    )


if __name__ == "__main__":
    main()
