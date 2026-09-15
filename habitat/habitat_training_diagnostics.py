"""Dataset splitting, auditing, saved previews, and live validation previews."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


Sample = tuple[Path, int, Path]
OCCUPANCY_CHANNEL = 0
FREE_SPACE_CHANNEL = 1
OBSERVED_CHANNEL = 2


@dataclass
class TargetStatistics:
    samples: int
    observed_cells: int
    total_cells: int
    occupied_cells: int
    free_cells: int

    @property
    def observed_fraction(self) -> float:
        return self.observed_cells / max(1, self.total_cells)

    @property
    def occupancy_fraction(self) -> float:
        return self.occupied_cells / max(1, self.observed_cells)

    @property
    def free_fraction(self) -> float:
        return self.free_cells / max(1, self.observed_cells)


@dataclass
class _Block:
    sequence: Path
    order: int
    samples: list[Sample]
    occupancy_fraction: float


def _sample_counts(sample: Sample) -> tuple[int, int, int, int]:
    with np.load(sample[2], allow_pickle=False) as archive:
        target = archive["target"]
    if target.ndim != 3 or target.shape[0] < 3:
        raise ValueError(f"Invalid target {target.shape} in {sample[2]}")
    observed = target[OBSERVED_CHANNEL] >= 0.5
    occupied = (target[OCCUPANCY_CHANNEL] >= 0.5) & observed
    free = (target[FREE_SPACE_CHANNEL] >= 0.5) & observed
    return (
        target.shape[1] * target.shape[2],
        int(observed.sum()),
        int(occupied.sum()),
        int(free.sum()),
    )


def target_statistics(samples: Sequence[Sample]) -> TargetStatistics:
    total = observed = occupied = free = 0
    for sample in samples:
        values = _sample_counts(sample)
        total += values[0]
        observed += values[1]
        occupied += values[2]
        free += values[3]
    return TargetStatistics(len(samples), observed, total, occupied, free)


def _quota_counts(size: int, fractions: Sequence[float]) -> list[int]:
    raw = [size * fraction for fraction in fractions]
    result = [int(value) for value in raw]
    for index in sorted(
        range(len(raw)),
        key=lambda item: raw[item] - result[item],
        reverse=True,
    )[: size - sum(result)]:
        result[index] += 1
    return result


def split_by_stratified_blocks(
    samples: list[Sample],
    *,
    seed: int = 42,
    block_size: int = 25,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Keep neighbouring frames together and balance block occupancy rates."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    test_fraction = 1.0 - train_fraction - validation_fraction
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be positive")

    grouped: dict[Path, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample[0], []).append(sample)
    blocks: list[_Block] = []
    for sequence, sequence_samples in sorted(grouped.items(), key=lambda item: str(item[0])):
        ordered = sorted(sequence_samples, key=lambda sample: sample[1])
        for start in range(0, len(ordered), block_size):
            selected = ordered[start : start + block_size]
            observed = occupied = 0
            for sample in selected:
                _, sample_observed, sample_occupied, _ = _sample_counts(sample)
                observed += sample_observed
                occupied += sample_occupied
            blocks.append(
                _Block(
                    sequence,
                    start // block_size,
                    selected,
                    occupied / max(1, observed),
                )
            )
    if len(blocks) < 3:
        raise ValueError("At least three temporal blocks are required")

    blocks.sort(key=lambda block: block.occupancy_fraction)
    fractions = (train_fraction, validation_fraction, test_fraction)
    remaining = _quota_counts(len(blocks), fractions)
    selected_blocks: list[list[_Block]] = [[], [], []]
    generator = random.Random(seed)
    for start in range(0, len(blocks), 10):
        stratum = blocks[start : start + 10]
        generator.shuffle(stratum)
        remaining_total = len(blocks) - start
        if len(stratum) == remaining_total:
            counts = remaining.copy()
        else:
            counts = _quota_counts(
                len(stratum),
                [count / remaining_total for count in remaining],
            )
        labels = [index for index, count in enumerate(counts) for _ in range(count)]
        generator.shuffle(labels)
        for block, label in zip(stratum, labels):
            selected_blocks[label].append(block)
        remaining = [old - used for old, used in zip(remaining, counts)]

    results: list[list[Sample]] = []
    for subset in selected_blocks:
        subset.sort(key=lambda block: (str(block.sequence), block.order))
        results.append([sample for block in subset for sample in block.samples])
    if any(not result for result in results):
        raise ValueError("The dataset is too small to produce three non-empty splits")
    return results[0], results[1], results[2]


def randomly_limit_samples(
    samples: list[Sample], maximum: int, *, seed: int
) -> list[Sample]:
    if maximum <= 0 or maximum >= len(samples):
        return samples
    selected = random.Random(seed).sample(samples, maximum)
    return sorted(selected, key=lambda sample: (str(sample[0]), sample[1]))


def audit_splits(
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
    *,
    expected_fractions: Sequence[float] = (0.70, 0.15, 0.15),
) -> dict[str, TargetStatistics]:
    named = {
        "train": train_samples,
        "validation": validation_samples,
        "test": test_samples,
    }
    identities = {
        name: {(str(sample[0]), sample[1]) for sample in subset}
        for name, subset in named.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if identities[left] & identities[right]:
            raise RuntimeError(f"Data leakage between {left} and {right}")
    statistics = {name: target_statistics(subset) for name, subset in named.items()}
    print("\nSplit target distribution (within observed BEV cells)")
    total_samples = sum(stat.samples for stat in statistics.values())
    for (name, stat), expected in zip(statistics.items(), expected_fractions):
        print(
            f"  {name:10s} samples={stat.samples:5d} "
            f"sequences={len({sample[0] for sample in named[name]}):3d} "
            f"observed={stat.observed_fraction:7.2%} "
            f"occupied={stat.occupancy_fraction:7.2%} free={stat.free_fraction:7.2%}"
        )
        actual = stat.samples / max(1, total_samples)
        if abs(actual - expected) > 0.03:
            print(f"  WARNING: {name} is {actual:.1%}; expected about {expected:.1%}")
    train = statistics["train"]
    for name in ("validation", "test"):
        stat = statistics[name]
        if abs(stat.observed_fraction - train.observed_fraction) > 0.05:
            print(f"  WARNING: {name} observed coverage differs from train by >5%")
        relative = abs(stat.occupancy_fraction - train.occupancy_fraction) / max(
            train.occupancy_fraction, 1e-9
        )
        if relative > 0.25:
            print(f"  WARNING: {name} occupancy prevalence differs by {relative:.0%}")
    return statistics


class LiveValidationPreview:
    """Reuse one GUI window to compare fixed validation samples across epochs."""

    def __init__(self, dataset: Any, *, count: int = 3, threshold: float = 0.5) -> None:
        if len(dataset) == 0 or count < 1:
            raise ValueError("A live preview requires a non-empty dataset and count")
        self.dataset = dataset
        self.threshold = threshold
        self.indices = np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int)
        self.figure: Any | None = None
        self.axes: Any | None = None

    def update(
        self,
        model: Any,
        device: Any,
        *,
        epoch: int,
        title_suffix: str = "",
    ) -> None:
        import matplotlib.pyplot as plt
        import torch
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        if self.figure is None:
            plt.ion()
            self.figure, self.axes = plt.subplots(
                len(self.indices), 4, figsize=(15, 4 * len(self.indices)), squeeze=False
            )
        was_training = model.training
        model.eval()
        error_cmap = ListedColormap(["#202020", "#2fb344", "#e03131", "#1c7ed6"])
        with torch.inference_mode():
            for row, dataset_index in enumerate(self.indices):
                sample = self.dataset[int(dataset_index)]
                batch = {
                    name: value.unsqueeze(0).to(device=device, dtype=torch.float32)
                    for name, value in sample.items()
                }
                logits = model(
                    batch["images"],
                    batch["intrinsics"],
                    batch["rotations"],
                    batch["translations"],
                )
                probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
                target = batch["target"][0].cpu().numpy()
                truth = target[0] >= 0.5
                observed = target[2] >= 0.5
                prediction = (probability >= self.threshold) & observed
                errors = np.zeros_like(truth, dtype=np.uint8)
                errors[prediction & truth] = 1
                errors[prediction & ~truth] = 2
                errors[~prediction & truth & observed] = 3
                tp = np.count_nonzero(prediction & truth)
                fp = np.count_nonzero(prediction & ~truth)
                fn = np.count_nonzero(~prediction & truth & observed)
                iou = tp / max(1, tp + fp + fn)
                for axis in self.axes[row]:
                    axis.clear()
                    axis.axis("off")
                front = batch["images"][0, 1].permute(1, 2, 0).cpu().numpy()
                self.axes[row, 0].imshow(front)
                self.axes[row, 0].set_title(f"Validation {int(dataset_index)} RGB")
                self.axes[row, 1].imshow(truth, origin="lower", cmap="gray")
                self.axes[row, 1].set_title("Target occupancy")
                self.axes[row, 2].imshow(
                    probability, origin="lower", cmap="magma", vmin=0, vmax=1
                )
                if prediction.any() and (~prediction).any():
                    self.axes[row, 2].contour(
                        prediction.astype(float),
                        levels=[0.5],
                        colors=["cyan"],
                        linewidths=0.7,
                        origin="lower",
                    )
                self.axes[row, 2].set_title(f"Probability + threshold {self.threshold:.2f}")
                self.axes[row, 3].imshow(errors, origin="lower", cmap=error_cmap, vmin=0, vmax=3)
                self.axes[row, 3].set_title(f"Errors — IoU {iou:.3f}")
        self.axes[0, 3].legend(
            handles=[
                Patch(color="#2fb344", label="TP"),
                Patch(color="#e03131", label="FP"),
                Patch(color="#1c7ed6", label="FN"),
            ],
            loc="lower right",
        )
        suffix = f" — {title_suffix}" if title_suffix else ""
        self.figure.suptitle(f"Live validation — epoch {epoch}{suffix}")
        self.figure.tight_layout()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.show(block=False)
        plt.pause(0.05)
        model.train(was_training)

    def save(self, output_path: Path) -> None:
        if self.figure is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.figure.savefig(output_path, dpi=160, bbox_inches="tight")

    def keep_open(self) -> None:
        if self.figure is not None:
            import matplotlib.pyplot as plt

            plt.ioff()
            plt.show(block=True)


def save_validation_previews(
    model: Any,
    dataset: Any,
    device: Any,
    output_directory: Path,
    *,
    epoch: int,
    count: int = 3,
    threshold: float = 0.5,
) -> list[Path]:
    """Compatibility helper: save fixed examples as one combined panel."""
    preview = LiveValidationPreview(dataset, count=count, threshold=threshold)
    preview.update(model, device, epoch=epoch)
    path = output_directory / f"epoch_{epoch:03d}.png"
    preview.save(path)
    return [path]
