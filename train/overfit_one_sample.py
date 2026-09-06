import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from nuscenes.nuscenes import NuScenes
from torch.nn import functional as F

from lss.image_encoder import LssImageEncoder
from lss.lift import LiftGeometry
from lss.camera_to_ego import CameraToEgo
from lss.bev_pool import BevPool
from lss.bev_decoder import BevDecoder

from lss.model_settings import (
	IMAGE_HEIGHT,
	IMAGE_WIDTH,
	DEPTH_MINIMUM,
	DEPTH_MAXIMUM,
	DEPTH_STEP,
	DEPTH_BINS,
	CONTEXT_CHANNELS,
	BASE_CHANNELS,
	BEV_X_MINIMUM,
	BEV_X_MAXIMUM,
	BEV_Y_MINIMUM,
	BEV_Y_MAXIMUM,
	BEV_Z_MINIMUM,
	BEV_Z_MAXIMUM,
	BEV_RESOLUTION,
)

from utils.data.data_input import (
	ordered_scene_samples,
)

from utils.data.camera_geometry import (
	prepare_six_camera_extrinsics,
)

from utils.data.process_image import (
	select_device,
	prepare_six_camera_batch,
)

from utils.data.bev_target import (
	create_vehicle_bev_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Overfit LSS on one nuScenes sample."
	)

	parser.add_argument(
		"--scene-index",
		type=int,
		default=0,
	)

	parser.add_argument(
		"--frame-index",
		type=int,
		default=0,
	)

	parser.add_argument(
		"--steps",
		type=int,
		default=300,
	)

	parser.add_argument(
		"--learning-rate",
		type=float,
		default=1e-3,
	)

	parser.add_argument(
		"--output-directory",
		type=Path,
		default=Path("outputs"),
	)

	return parser.parse_args()


def dice_loss(
	logits: torch.Tensor,
	target: torch.Tensor,
	epsilon: float = 1e-6,
) -> torch.Tensor:
	probabilities = torch.sigmoid(logits)

	intersection = (
		probabilities * target
	).sum(dim=(1, 2, 3))

	probability_sum = probabilities.sum(
		dim=(1, 2, 3)
	)

	target_sum = target.sum(
		dim=(1, 2, 3)
	)

	dice_score = (
		2.0 * intersection + epsilon
	) / (
		probability_sum
		+ target_sum
		+ epsilon
	)

	return 1.0 - dice_score.mean()


def run_pipeline(
	*,
	images: torch.Tensor,
	intrinsics: torch.Tensor,
	camera_rotations: torch.Tensor,
	camera_translations: torch.Tensor,
	image_encoder: LssImageEncoder,
	lift_geometry: LiftGeometry,
	camera_to_ego: CameraToEgo,
	bev_pool: BevPool,
	bev_decoder: BevDecoder,
) -> torch.Tensor:
	encoder_output = image_encoder(
		images
	)

	lift_output = lift_geometry(
		encoder_output,
		intrinsics,
		image_height=IMAGE_HEIGHT,
		image_width=IMAGE_WIDTH,
	)

	points_ego = camera_to_ego(
		lift_output.points_camera,
		camera_rotations,
		camera_translations,
	)

	bev_output = bev_pool(
		points_ego,
		lift_output.features,
	)

	vehicle_logits = bev_decoder(
		bev_output.features
	)

	return vehicle_logits


def main() -> None:
	arguments = parse_arguments()

	if arguments.steps <= 0:
		raise ValueError("--steps must be positive")

	if arguments.learning_rate <= 0:
		raise ValueError("--learning-rate must be positive")

	dataset_value = os.getenv(
		"NUSCENES_ROOT"
	)

	if dataset_value is None:
		raise RuntimeError(
			"NUSCENES_ROOT is missing from .env"
		)

	dataset_root = Path(
		dataset_value
	).expanduser().resolve()

	output_directory = (
		arguments.output_directory
		.expanduser()
		.resolve()
	)

	output_directory.mkdir(
		parents=True,
		exist_ok=True,
	)

	device = select_device()

	torch.manual_seed(42)

	print(f"Device: {device}")
	print(f"Dataset: {dataset_root}")

	nuscenes = NuScenes(
		version="v1.0-mini",
		dataroot=str(dataset_root),
		verbose=True,
	)

	if not 0 <= arguments.scene_index < len(nuscenes.scene):
		raise IndexError(
			"Invalid scene index"
		)

	scene = nuscenes.scene[
		arguments.scene_index
	]

	samples = ordered_scene_samples(
		nuscenes=nuscenes,
		scene=scene,
	)

	if not 0 <= arguments.frame_index < len(samples):
		raise IndexError(
			"Invalid frame index"
		)

	sample = samples[
		arguments.frame_index
	]

	# ---------------------------------------------------------------
	# Prepare one training example.
	# ---------------------------------------------------------------

	images, intrinsics = prepare_six_camera_batch(
		nuscenes=nuscenes,
		sample=sample,
	)

	camera_rotations, camera_translations = (
		prepare_six_camera_extrinsics(
			nuscenes=nuscenes,
			sample=sample,
		)
	)

	vehicle_target = create_vehicle_bev_target(
		nuscenes=nuscenes,
		sample=sample,
	)

	images = images.to(
		device=device,
		dtype=torch.float32,
	)

	intrinsics = intrinsics.to(
		device=device,
		dtype=torch.float32,
	)

	camera_rotations = camera_rotations.to(
		device=device,
		dtype=torch.float32,
	)

	camera_translations = camera_translations.to(
		device=device,
		dtype=torch.float32,
	)

	vehicle_target = vehicle_target.to(
		device=device,
		dtype=torch.float32,
	)

	positive_cells = vehicle_target.sum()

	if positive_cells.item() <= 0:
		raise RuntimeError(
			"The selected frame contains no vehicle "
			"cells inside the BEV area. Choose another frame."
		)

	# ---------------------------------------------------------------
	# Create model components.
	# ---------------------------------------------------------------

	image_encoder = LssImageEncoder(
		depth_bins=DEPTH_BINS,
		context_channels=CONTEXT_CHANNELS,
		base_channels=BASE_CHANNELS,
	).to(device)

	lift_geometry = LiftGeometry(
		depth_minimum=DEPTH_MINIMUM,
		depth_maximum=DEPTH_MAXIMUM,
		depth_step=DEPTH_STEP,
	).to(device)

	camera_to_ego = CameraToEgo().to(
		device
	)

	bev_pool = BevPool(
		x_minimum=BEV_X_MINIMUM,
		x_maximum=BEV_X_MAXIMUM,
		y_minimum=BEV_Y_MINIMUM,
		y_maximum=BEV_Y_MAXIMUM,
		z_minimum=BEV_Z_MINIMUM,
		z_maximum=BEV_Z_MAXIMUM,
		resolution=BEV_RESOLUTION,
	).to(device)

	bev_decoder = BevDecoder(
		input_channels=CONTEXT_CHANNELS,
		hidden_channels=96,
		output_channels=1,
	).to(device)

	# Only these two components contain trainable weights.
	image_encoder.train()
	bev_decoder.train()

	# These are deterministic geometry operations.
	lift_geometry.eval()
	camera_to_ego.eval()
	bev_pool.eval()

	trainable_parameters = [
		*image_encoder.parameters(),
		*bev_decoder.parameters(),
	]

	optimizer = torch.optim.AdamW(
		trainable_parameters,
		lr=arguments.learning_rate,
		weight_decay=1e-4,
	)

	# There are far more background cells than vehicle cells.
	negative_cells = (
		vehicle_target.numel()
		- positive_cells
	)

	pos_weight_value = (
		negative_cells / positive_cells
	).clamp(
		min=1.0,
		max=50.0,
	)

	pos_weight = pos_weight_value.reshape(
		1,
		1,
		1,
	)

	print(
		f"Vehicle cells: {int(positive_cells.item())}/"
		f"{vehicle_target.numel()}"
	)

	print(
		f"Positive-class weight: "
		f"{pos_weight_value.item():.3f}"
	)

	# ---------------------------------------------------------------
	# Prediction before training.
	# ---------------------------------------------------------------

	image_encoder.eval()
	bev_decoder.eval()

	with torch.no_grad():
		initial_logits = run_pipeline(
			images=images,
			intrinsics=intrinsics,
			camera_rotations=camera_rotations,
			camera_translations=camera_translations,
			image_encoder=image_encoder,
			lift_geometry=lift_geometry,
			camera_to_ego=camera_to_ego,
			bev_pool=bev_pool,
			bev_decoder=bev_decoder,
		)

		initial_prediction = torch.sigmoid(
			initial_logits
		).detach().cpu()

	image_encoder.train()
	bev_decoder.train()

	loss_history: list[float] = []

	# ---------------------------------------------------------------
	# Training loop.
	# ---------------------------------------------------------------

	for step in range(1, arguments.steps + 1):
		optimizer.zero_grad(
			set_to_none=True
		)

		vehicle_logits = run_pipeline(
			images=images,
			intrinsics=intrinsics,
			camera_rotations=camera_rotations,
			camera_translations=camera_translations,
			image_encoder=image_encoder,
			lift_geometry=lift_geometry,
			camera_to_ego=camera_to_ego,
			bev_pool=bev_pool,
			bev_decoder=bev_decoder,
		)

		if vehicle_logits.shape != vehicle_target.shape:
			raise RuntimeError(
				"Prediction and target shapes disagree: "
				f"{tuple(vehicle_logits.shape)} versus "
				f"{tuple(vehicle_target.shape)}"
			)

		bce = F.binary_cross_entropy_with_logits(
			vehicle_logits,
			vehicle_target,
			pos_weight=pos_weight,
		)

		dice = dice_loss(
			vehicle_logits,
			vehicle_target,
		)

		loss = (
			0.7 * bce
			+ 0.3 * dice
		)

		if not torch.isfinite(loss):
			raise FloatingPointError(
				f"Non-finite loss at step {step}: {loss.item()}"
			)

		loss.backward()

		gradient_norm = torch.nn.utils.clip_grad_norm_(
			trainable_parameters,
			max_norm=5.0,
		)

		if step == 1:
			has_encoder_gradient = any(
				parameter.grad is not None
				and torch.isfinite(parameter.grad).all()
				and parameter.grad.abs().sum() > 0
				for parameter in image_encoder.parameters()
			)

			has_decoder_gradient = any(
				parameter.grad is not None
				and torch.isfinite(parameter.grad).all()
				and parameter.grad.abs().sum() > 0
				for parameter in bev_decoder.parameters()
			)

			print(
				"Encoder receives gradient:",
				has_encoder_gradient,
			)
			print(
				"Decoder receives gradient:",
				has_decoder_gradient,
			)

			if not has_encoder_gradient:
				raise RuntimeError(
					"No gradient reaches the image encoder. "
					"Check whether BevPool preserves autograd."
				)

		optimizer.step()

		loss_history.append(
			float(loss.item())
		)

		if (
			step == 1
			or step % 10 == 0
			or step == arguments.steps
		):
			with torch.no_grad():
				probabilities = torch.sigmoid(
					vehicle_logits
				)

				predicted_vehicle_cells = (
					probabilities > 0.5
				).sum().item()

			print(
				f"step={step:04d} "
				f"loss={loss.item():.6f} "
				f"bce={bce.item():.6f} "
				f"dice={dice.item():.6f} "
				f"gradient={float(gradient_norm):.4f} "
				f"predicted_cells={predicted_vehicle_cells}"
			)

	# ---------------------------------------------------------------
	# Final prediction.
	# ---------------------------------------------------------------

	image_encoder.eval()
	bev_decoder.eval()

	with torch.no_grad():
		final_logits = run_pipeline(
			images=images,
			intrinsics=intrinsics,
			camera_rotations=camera_rotations,
			camera_translations=camera_translations,
			image_encoder=image_encoder,
			lift_geometry=lift_geometry,
			camera_to_ego=camera_to_ego,
			bev_pool=bev_pool,
			bev_decoder=bev_decoder,
		)

		final_prediction = torch.sigmoid(
			final_logits
		).detach().cpu()

	# ---------------------------------------------------------------
	# Save checkpoint.
	# ---------------------------------------------------------------

	checkpoint_path = (
		output_directory
		/ "lss_one_sample_overfit.pt"
	)

	torch.save(
		{
			"image_encoder": image_encoder.state_dict(),
			"bev_decoder": bev_decoder.state_dict(),
			"optimizer": optimizer.state_dict(),
			"scene_index": arguments.scene_index,
			"frame_index": arguments.frame_index,
			"steps": arguments.steps,
			"loss_history": loss_history,
		},
		checkpoint_path,
	)

	print(f"Checkpoint saved to: {checkpoint_path}")

	# ---------------------------------------------------------------
	# Visualize the learning result.
	# ---------------------------------------------------------------

	target_image = (
		vehicle_target[0, 0]
		.detach()
		.cpu()
		.numpy()
	)

	initial_image = (
		initial_prediction[0, 0]
		.numpy()
	)

	final_image = (
		final_prediction[0, 0]
		.numpy()
	)

	extent = [
		BEV_Y_MINIMUM,
		BEV_Y_MAXIMUM,
		BEV_X_MINIMUM,
		BEV_X_MAXIMUM,
	]

	figure, axes = plt.subplots(
		1,
		4,
		figsize=(20, 5),
	)

	axes[0].imshow(
		target_image,
		origin="lower",
		extent=extent,
		cmap="gray",
		vmin=0.0,
		vmax=1.0,
	)
	axes[0].set_title("Ground-truth vehicle target")

	axes[1].imshow(
		initial_image,
		origin="lower",
		extent=extent,
		cmap="magma",
		vmin=0.0,
		vmax=1.0,
	)
	axes[1].set_title("Before training")

	axes[2].imshow(
		final_image,
		origin="lower",
		extent=extent,
		cmap="magma",
		vmin=0.0,
		vmax=1.0,
	)
	axes[2].set_title("After training")

	axes[3].plot(
		loss_history,
		color="blue",
	)
	axes[3].set_title("Training loss")
	axes[3].set_xlabel("Training step")
	axes[3].set_ylabel("Loss")
	axes[3].grid(True)

	for axis in axes[:3]:
		axis.scatter(
			0.0,
			0.0,
			color="cyan",
			marker="^",
			s=70,
		)
		axis.set_xlabel("Ego Y — left [m]")
		axis.set_ylabel("Ego X — forward [m]")

	figure.suptitle(
		f"One-sample LSS overfit — {scene['name']} — "
		f"frame {arguments.frame_index}"
	)

	figure.tight_layout()

	plot_path = (
		output_directory
		/ "lss_one_sample_overfit.png"
	)

	figure.savefig(
		plot_path,
		dpi=160,
		bbox_inches="tight",
	)

	print(f"Plot saved to: {plot_path}")

	plt.show()


if __name__ == "__main__":
	main()