from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from habitat.read_dataset import load_rgb_and_depth
from habitat_lss.habitat_lss_model import HabitatLiftSplatShoot
from habitat.habitat_training_diagnostics import (
    audit_splits,
    randomly_limit_samples,
    save_validation_previews,
    split_by_stratified_blocks,
)


CAMERAS = (
	"camera_front_left",
	"camera_front",
	"camera_front_right",
	"camera_back_left",
	"camera_back",
	"camera_back_right",
)

TARGET_CHANNELS = 14
OCCUPANCY_CHANNEL = 0
FREE_SPACE_CHANNEL = 1
OBSERVED_CHANNEL = 2
HEIGHT_CHANNELS = (10, 11, 12)
DENSITY_CHANNEL = 13


@dataclass
class Metrics:
	loss: float = 0.0
	iou: float = 0.0
	precision: float = 0.0
	recall: float = 0.0


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument("--split-block-size", type=int, default=25)
	parser.add_argument("--validation-previews", type=int, default=3)
	parser.add_argument("--prediction-threshold", type=float, default=0.5)

	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=Path("outputs/habitat_dataset/scene_102344280"),
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("outputs/habitat_lss"),
	)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument(
		"--learning-rate",
		type=float,
		default=3e-4,
	)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--print-every", type=int, default=25)
	parser.add_argument("--preview-every", type=int, default=50)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--max-train-samples",
		type=int,
		default=0,
	)
	parser.add_argument(
		"--max-validation-samples",
		type=int,
		default=0,
	)
	parser.add_argument(
		"--max-test-samples",
		type=int,
		default=0,
	)

	return parser.parse_args()


def select_device() -> torch.device:
	if torch.backends.mps.is_available():
		return torch.device("mps")

	if torch.cuda.is_available():
		return torch.device("cuda")

	return torch.device("cpu")


def limit_samples(
	samples: list[tuple[Path, int, Path]],
	maximum: int,
) -> list[tuple[Path, int, Path]]:
	if maximum <= 0:
		return samples

	return samples[:maximum]


def find_samples(
	dataset_root: Path,
) -> list[tuple[Path, int, Path]]:
	samples: list[tuple[Path, int, Path]] = []

	for manifest_path in sorted(
		dataset_root.rglob("manifest.jsonl")
	):
		sequence_directory = manifest_path.parent
		target_directory = sequence_directory / "bev_targets"

		for target_path in sorted(
			target_directory.glob("frame_*.npz")
		):
			frame_index = int(
				target_path.stem.split("_")[-1]
			)

			samples.append(
				(
					sequence_directory,
					frame_index,
					target_path,
				)
			)

	if not samples:
		raise RuntimeError(
			"No BEV targets found. Run "
			"`python -m habitat.build_bev_dataset` first."
		)

	return samples


def split_by_sequence(
	samples: list[tuple[Path, int, Path]],
) -> tuple[
	list[tuple[Path, int, Path]],
	list[tuple[Path, int, Path]],
	list[tuple[Path, int, Path]],
]:
	grouped: dict[Path, list[tuple[Path, int, Path]]] = {}

	for sample in samples:
		grouped.setdefault(sample[0], []).append(sample)

	train_samples = []
	validation_samples = []
	test_samples = []

	for sequence_samples in grouped.values():
		sequence_samples = sorted(
			sequence_samples,
			key=lambda value: value[1],
		)

		count = len(sequence_samples)
		train_end = max(1, int(count * 0.70))
		validation_end = max(
			train_end + 1,
			int(count * 0.85),
		)

		train_samples.extend(
			sequence_samples[:train_end]
		)
		validation_samples.extend(
			sequence_samples[
				train_end:validation_end
			]
		)
		test_samples.extend(
			sequence_samples[validation_end:]
		)

	return (
		train_samples,
		validation_samples,
		test_samples,
	)


class HabitatBevDataset(Dataset):
	def __init__(
		self,
		samples: list[tuple[Path, int, Path]],
	) -> None:
		self.samples = samples

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(
		self,
		index: int,
	) -> dict[str, torch.Tensor]:
		(
			sequence_directory,
			frame_index,
			target_path,
		) = self.samples[index]

		images = []
		depths = []
		intrinsics = []
		rotations = []
		translations = []

		for camera_name in CAMERAS:
			(
				rgb,
				depth,
				intrinsic,
				agent_from_camera,
			) = load_rgb_and_depth(
				sequence_directory,
				frame_index=frame_index,
				camera_name=camera_name,
			)

			image_tensor = (
				torch.from_numpy(rgb.copy())
				.permute(2, 0, 1)
				.float()
				/ 255.0
			)

			depth = np.asarray(depth, dtype=np.float32)

			if depth.ndim == 3 and depth.shape[-1] == 1:
				depth = depth[..., 0]

			if depth.ndim != 2:
				raise ValueError(
					f"Expected depth [H, W], got {depth.shape}"
				)

			images.append(image_tensor)
			depths.append(torch.from_numpy(depth.copy()))
			intrinsics.append(
				torch.from_numpy(
					np.asarray(
						intrinsic,
						dtype=np.float32,
					)
				)
			)
			rotations.append(
				torch.from_numpy(
					np.asarray(
						agent_from_camera[:3, :3],
						dtype=np.float32,
					)
				)
			)
			translations.append(
				torch.from_numpy(
					np.asarray(
						agent_from_camera[:3, 3],
						dtype=np.float32,
					)
				)
			)

		with np.load(
			target_path,
			allow_pickle=False,
		) as archive:
			target = archive["target"].astype(
				np.float32
			)

		if target.shape != (TARGET_CHANNELS, 100, 100):
			raise ValueError(
				f"Unexpected target shape {target.shape} "
				f"in {target_path}"
			)

		return {
			"images": torch.stack(images),
			"depths": torch.stack(depths),  # [6, H, W]
			"intrinsics": torch.stack(intrinsics),
			"rotations": torch.stack(rotations),
			"translations": torch.stack(translations),
			"target": torch.from_numpy(target),
		}


def move_batch(
	batch: dict[str, torch.Tensor],
	device: torch.device,
) -> dict[str, torch.Tensor]:
	return {
		name: value.to(
			device=device,
			dtype=torch.float32,
		)
		for name, value in batch.items()
	}


def masked_mean(
	values: torch.Tensor,
	mask: torch.Tensor,
) -> torch.Tensor:
	return (
		(values * mask).sum()
		/ mask.sum().clamp_min(1.0)
	)


def calculate_loss(
	logits: torch.Tensor,
	target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
	observed = target[:, OBSERVED_CHANNEL:OBSERVED_CHANNEL + 1]
	occupancy_target = target[:, OCCUPANCY_CHANNEL:OCCUPANCY_CHANNEL + 1]
	free_target = target[:, FREE_SPACE_CHANNEL:FREE_SPACE_CHANNEL + 1]

	vertical_target = target[:, 3:10]

	binary_logits = torch.cat(
		(
			logits[:, OCCUPANCY_CHANNEL:OCCUPANCY_CHANNEL + 1],
			logits[:, FREE_SPACE_CHANNEL:FREE_SPACE_CHANNEL + 1],
			logits[:, 3:10],
		),
		dim=1,
	)

	binary_target = torch.cat(
		(
			occupancy_target,
			free_target,
			vertical_target,
		),
		dim=1,
	)

	binary_loss_map = F.binary_cross_entropy_with_logits(
		binary_logits,
		binary_target,
		reduction="none",
	)

	observed_binary_mask = observed.expand_as(
		binary_loss_map
	)

	binary_loss = masked_mean(
		binary_loss_map,
		observed_binary_mask,
	)

	observed_loss = F.binary_cross_entropy_with_logits(
		logits[:, OBSERVED_CHANNEL:OBSERVED_CHANNEL + 1],
		observed,
	)

	predicted_heights = torch.sigmoid(
		logits[:, HEIGHT_CHANNELS]
	)
	target_heights = target[:, HEIGHT_CHANNELS]

	height_loss_map = F.smooth_l1_loss(
		predicted_heights,
		target_heights,
		reduction="none",
	)

	height_mask = occupancy_target.expand_as(
		height_loss_map
	)

	height_loss = masked_mean(
		height_loss_map,
		height_mask,
	)

	predicted_density = torch.sigmoid(
		logits[:, DENSITY_CHANNEL:DENSITY_CHANNEL + 1]
	)
	target_density = target[
		:,
		DENSITY_CHANNEL:DENSITY_CHANNEL + 1,
	]

	density_loss_map = F.smooth_l1_loss(
		predicted_density,
		target_density,
		reduction="none",
	)

	density_loss = masked_mean(
		density_loss_map,
		observed,
	)

	total_loss = (
		binary_loss
		+ 0.25 * observed_loss
		+ 0.50 * height_loss
		+ 0.10 * density_loss
	)

	components = {
		"binary": float(binary_loss.item()),
		"observed": float(observed_loss.item()),
		"height": float(height_loss.item()),
		"density": float(density_loss.item()),
	}

	return total_loss, components


def occupancy_counts(
	logits: torch.Tensor,
	target: torch.Tensor,
) -> tuple[int, int, int]:
	prediction = torch.sigmoid(
		logits[:, OCCUPANCY_CHANNEL]
	) >= 0.5

	truth = target[:, OCCUPANCY_CHANNEL] >= 0.5
	observed = target[:, OBSERVED_CHANNEL] >= 0.5

	prediction = prediction & observed
	truth = truth & observed

	true_positive = int(
		(prediction & truth).sum().item()
	)
	false_positive = int(
		(prediction & ~truth).sum().item()
	)
	false_negative = int(
		(~prediction & truth).sum().item()
	)

	return (
		true_positive,
		false_positive,
		false_negative,
	)


def metrics_from_counts(
	loss_sum: float,
	batches: int,
	true_positive: int,
	false_positive: int,
	false_negative: int,
) -> Metrics:
	iou = true_positive / (
		true_positive
		+ false_positive
		+ false_negative
		+ 1e-9
	)

	precision = true_positive / (
		true_positive
		+ false_positive
		+ 1e-9
	)

	recall = true_positive / (
		true_positive
		+ false_negative
		+ 1e-9
	)

	return Metrics(
		loss=loss_sum / max(1, batches),
		iou=iou,
		precision=precision,
		recall=recall,
	)


def evaluate(
	model: HabitatLiftSplatShoot,
	loader: DataLoader,
	device: torch.device,
) -> Metrics:
	model.eval()

	loss_sum = 0.0
	batches = 0
	true_positive = 0
	false_positive = 0
	false_negative = 0

	with torch.inference_mode():
		for batch in loader:
			batch = move_batch(batch, device)

			logits = model(
				batch["images"],
				batch["intrinsics"],
				batch["rotations"],
				batch["translations"],
			)

			loss, _ = calculate_loss(
				logits,
				batch["target"],
			)

			tp, fp, fn = occupancy_counts(
				logits,
				batch["target"],
			)

			loss_sum += float(loss.item())
			batches += 1
			true_positive += tp
			false_positive += fp
			false_negative += fn

	return metrics_from_counts(
		loss_sum,
		batches,
		true_positive,
		false_positive,
		false_negative,
	)


def save_preview(
	model: HabitatLiftSplatShoot,
	dataset: Dataset,
	device: torch.device,
	output_path: Path,
	step: int,
) -> None:
	model.eval()

	sample = dataset[0]

	batch = {
		name: value.unsqueeze(0).to(
			device=device,
			dtype=torch.float32,
		)
		for name, value in sample.items()
	}

	with torch.inference_mode():
		logits = model(
			batch["images"],
			batch["intrinsics"],
			batch["rotations"],
			batch["translations"],
		)

	prediction = torch.sigmoid(logits)[0].cpu().numpy()
	target = batch["target"][0].cpu().numpy()

	front_image = (
		batch["images"][0, 1]
		.permute(1, 2, 0)
		.cpu()
		.numpy()
	)

	figure, axes = plt.subplots(
		2,
		3,
		figsize=(15, 9),
	)

	axes[0, 0].imshow(front_image)
	axes[0, 0].set_title("CAM_FRONT RGB")
	axes[0, 0].axis("off")

	axes[0, 1].imshow(
		target[OCCUPANCY_CHANNEL],
		origin="lower",
		cmap="gray",
		vmin=0,
		vmax=1,
	)
	axes[0, 1].set_title("Target occupancy")

	axes[0, 2].imshow(
		prediction[OCCUPANCY_CHANNEL],
		origin="lower",
		cmap="magma",
		vmin=0,
		vmax=1,
	)
	axes[0, 2].set_title("Predicted occupancy")

	axes[1, 0].imshow(
		target[FREE_SPACE_CHANNEL],
		origin="lower",
		cmap="Blues",
		vmin=0,
		vmax=1,
	)
	axes[1, 0].set_title("Target free-space")

	axes[1, 1].imshow(
		prediction[FREE_SPACE_CHANNEL],
		origin="lower",
		cmap="Blues",
		vmin=0,
		vmax=1,
	)
	axes[1, 1].set_title("Predicted free-space")

	axes[1, 2].imshow(
		prediction[11] * 2.0,
		origin="lower",
		cmap="viridis",
		vmin=0,
		vmax=2,
	)
	axes[1, 2].set_title("Predicted maximum height [m]")

	for axis in axes.flat:
		axis.set_xlabel("Agent right [m]")
		axis.set_ylabel("Agent forward [m]")

	figure.suptitle(
		f"Habitat LSS preview — optimizer step {step}"
	)
	figure.tight_layout()

	output_path.parent.mkdir(
		parents=True,
		exist_ok=True,
	)
	figure.savefig(output_path, dpi=160)
	plt.close(figure)

	model.train()


def main() -> None:
	arguments = parse_arguments()

	if arguments.batch_size <= 0:
		raise ValueError("--batch-size must be positive")

	random.seed(arguments.seed)
	np.random.seed(arguments.seed)
	torch.manual_seed(arguments.seed)

	device = select_device()
	output_directory = arguments.output_dir.expanduser().resolve()
	output_directory.mkdir(
		parents=True,
		exist_ok=True,
	)

	all_samples = find_samples(
		arguments.dataset_root.expanduser().resolve()
	)

	train_samples, validation_samples, test_samples = (
		split_by_stratified_blocks(
			all_samples,
			seed=arguments.seed,
			block_size=arguments.split_block_size,
		)
	)

	train_samples = randomly_limit_samples(
		train_samples,
		arguments.max_train_samples,
		seed=arguments.seed,
	)
	validation_samples = randomly_limit_samples(
		validation_samples,
		arguments.max_validation_samples,
		seed=arguments.seed + 1,
	)
	test_samples = randomly_limit_samples(
		test_samples,
		arguments.max_test_samples,
		seed=arguments.seed + 2,
	)

	audit_splits(
		train_samples,
		validation_samples,
		test_samples,
	)

	train_dataset = HabitatBevDataset(train_samples)
	validation_dataset = HabitatBevDataset(validation_samples)
	test_dataset = HabitatBevDataset(test_samples)

	train_loader = DataLoader(
		train_dataset,
		batch_size=arguments.batch_size,
		shuffle=True,
		num_workers=arguments.num_workers,
	)
	validation_loader = DataLoader(
		validation_dataset,
		batch_size=arguments.batch_size,
		shuffle=False,
		num_workers=arguments.num_workers,
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=arguments.batch_size,
		shuffle=False,
		num_workers=arguments.num_workers,
	)

	model = HabitatLiftSplatShoot(
		output_channels=TARGET_CHANNELS,
	).to(device)

	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=arguments.learning_rate,
		weight_decay=1e-4,
	)

	print(f"Device: {device}")
	print(
		f"Samples: train={len(train_dataset)}, "
		f"validation={len(validation_dataset)}, "
		f"test={len(test_dataset)}"
	)
	print(
		"Trainable parameters:",
		f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
	)

	history: list[dict[str, float]] = []
	best_validation_iou = -1.0
	global_step = 0

	for epoch in range(1, arguments.epochs + 1):
		model.train()

		loss_sum = 0.0
		batches = 0
		true_positive = 0
		false_positive = 0
		false_negative = 0

		for batch_index, batch in enumerate(
			train_loader,
			start=1,
		):
			batch = move_batch(batch, device)

			optimizer.zero_grad(set_to_none=True)

			logits = model(
				batch["images"],
				batch["intrinsics"],
				batch["rotations"],
				batch["translations"],
			)

			loss, components = calculate_loss(
				logits,
				batch["target"],
			)

			loss.backward()

			gradient_norm = torch.nn.utils.clip_grad_norm_(
				model.parameters(),
				max_norm=5.0,
			)

			optimizer.step()

			tp, fp, fn = occupancy_counts(
				logits.detach(),
				batch["target"],
			)

			global_step += 1
			loss_sum += float(loss.item())
			batches += 1
			true_positive += tp
			false_positive += fp
			false_negative += fn

			if (
				batch_index == 1
				or batch_index % arguments.print_every == 0
			):
				print(
					f"epoch={epoch:03d}/{arguments.epochs:03d} "
					f"batch={batch_index:04d}/{len(train_loader):04d} "
					f"loss={loss.item():.5f} "
					f"binary={components['binary']:.5f} "
					f"height={components['height']:.5f} "
					f"gradient={float(gradient_norm):.3f}"
				)

			if (
				arguments.preview_every > 0
				and global_step % arguments.preview_every == 0
			):
				save_preview(
					model,
					validation_dataset,
					device,
					output_directory
					/ f"preview_step_{global_step:06d}.png",
					global_step,
				)

		train_metrics = metrics_from_counts(
			loss_sum,
			batches,
			true_positive,
			false_positive,
			false_negative,
		)

		validation_metrics = evaluate(
			model,
			validation_loader,
			device,
		)

		# preview_paths = save_validation_previews(
		# 	model,
		# 	validation_dataset,
		# 	device,
		# 	output_directory / "validation_previews",
		# 	epoch=epoch,
		# 	count=arguments.validation_previews,
		# 	threshold=arguments.prediction_threshold,
		# )

		# print(f"  Saved {len(preview_paths)} validation previews.")

		history.append(
			{
				"epoch": epoch,
				"train_loss": train_metrics.loss,
				"train_iou": train_metrics.iou,
				"validation_loss": validation_metrics.loss,
				"validation_iou": validation_metrics.iou,
				"validation_precision": validation_metrics.precision,
				"validation_recall": validation_metrics.recall,
			}
		)

		print(f"\nEpoch {epoch} complete")
		print(f"  train loss:      {train_metrics.loss:.5f}")
		print(f"  train IoU:       {train_metrics.iou:.4f}")
		print(
			f"  validation loss: "
			f"{validation_metrics.loss:.5f}"
		)
		print(
			f"  validation IoU:  "
			f"{validation_metrics.iou:.4f}"
		)
		print(
			f"  precision:       "
			f"{validation_metrics.precision:.4f}"
		)
		print(
			f"  recall:          "
			f"{validation_metrics.recall:.4f}"
		)

		checkpoint = {
			"epoch": epoch,
			"global_step": global_step,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"validation_iou": validation_metrics.iou,
		}

		torch.save(
			checkpoint,
			output_directory / "habitat_lss_last.pt",
		)

		if validation_metrics.iou > best_validation_iou:
			best_validation_iou = validation_metrics.iou

			torch.save(
				checkpoint,
				output_directory / "habitat_lss_best.pt",
			)

			print("  Saved new best checkpoint.")

		with (
			output_directory / "training_history.json"
		).open("w", encoding="utf-8") as file:
			json.dump(history, file, indent=2)

	test_metrics = evaluate(
		model,
		test_loader,
		device,
	)

	print("\nHeld-out test result")
	print(f"  loss:      {test_metrics.loss:.5f}")
	print(f"  IoU:       {test_metrics.iou:.4f}")
	print(f"  precision: {test_metrics.precision:.4f}")
	print(f"  recall:    {test_metrics.recall:.4f}")


if __name__ == "__main__":
    main()