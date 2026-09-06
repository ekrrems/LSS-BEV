import os
from pathlib import Path

import cv2
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
	CAMERA_LAYOUT,
	ordered_scene_samples,
	load_camera_image,
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

from utils.command_utils import parse_arguments


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


def count_trainable_parameters(
	model: torch.nn.Module,
) -> int:
	return sum(
		parameter.numel()
		for parameter in model.parameters()
		if parameter.requires_grad
	)


def main() -> None:
	arguments = parse_arguments()

	# ---------------------------------------------------------------
	# Load dataset location.
	# ---------------------------------------------------------------

	dataset_value = os.getenv("NUSCENES_ROOT")

	if dataset_value is None:
		raise RuntimeError(
			"NUSCENES_ROOT is missing from .env"
		)

	dataset_root = Path(
		dataset_value
	).expanduser().resolve()

	if not dataset_root.exists():
		raise FileNotFoundError(
			f"nuScenes dataset does not exist: {dataset_root}"
		)

	device = select_device()

	print(f"Device: {device}")
	print(f"Dataset: {dataset_root}")

	# ---------------------------------------------------------------
	# Load nuScenes.
	# ---------------------------------------------------------------

	nuscenes = NuScenes(
		version="v1.0-mini",
		dataroot=str(dataset_root),
		verbose=True,
	)

	if not 0 <= arguments.scene_index < len(nuscenes.scene):
		raise IndexError(
			f"Scene index must be between 0 and "
			f"{len(nuscenes.scene) - 1}"
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
			f"Frame index must be between 0 and "
			f"{len(samples) - 1}"
		)

	sample = samples[
		arguments.frame_index
	]

	camera_names = [
		camera_name
		for camera_name, _ in CAMERA_LAYOUT
	]

	if arguments.camera not in camera_names:
		raise ValueError(
			f"Unknown camera: {arguments.camera}. "
			f"Available cameras: {camera_names}"
		)

	camera_index = camera_names.index(
		arguments.camera
	)

	raw_image_bgr = load_camera_image(
		nuscenes=nuscenes,
		sample=sample,
		camera_name=arguments.camera,
	)

	raw_image_rgb = cv2.cvtColor(
		raw_image_bgr,
		cv2.COLOR_BGR2RGB,
	)

	# ---------------------------------------------------------------
	# Construct the complete LSS pipeline.
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

	camera_to_ego = CameraToEgo().to(device)

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

	# This script inspects the pipeline. It does not train it.
	image_encoder.eval()
	lift_geometry.eval()
	camera_to_ego.eval()
	bev_pool.eval()
	bev_decoder.eval()

	# ---------------------------------------------------------------
	# Prepare images and calibration.
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

	# Ground-truth vehicle occupancy:
	#
	# [B, 1, BEV height, BEV width]
	vehicle_target = create_vehicle_bev_target(
		nuscenes=nuscenes,
		sample=sample,
	)

	vehicle_target = vehicle_target.to(
		device=device,
		dtype=torch.float32,
	)

	# ---------------------------------------------------------------
	# Complete forward pipeline.
	# ---------------------------------------------------------------

	with torch.inference_mode():
		# 1. Six RGB images -> context and depth probabilities.
		encoder_output = image_encoder(
			images
		)

		# 2. Construct camera rays and distribute context over depth.
		lift_output = lift_geometry(
			encoder_output,
			intrinsics,
			image_height=IMAGE_HEIGHT,
			image_width=IMAGE_WIDTH,
		)

		# 3. Transform every candidate point into the ego frame.
		points_ego = camera_to_ego(
			lift_output.points_camera,
			camera_rotations,
			camera_translations,
		)

		# 4. Accumulate the lifted features in the BEV grid.
		bev_output = bev_pool(
			points_ego,
			lift_output.features,
		)

		# 5. Predict vehicle occupancy from the BEV features.
		vehicle_logits = bev_decoder(
			bev_output.features
		)

		vehicle_probabilities = torch.sigmoid(
			vehicle_logits
		)

		if vehicle_logits.shape != vehicle_target.shape:
			raise ValueError(
				"Vehicle prediction and target shapes disagree: "
				f"prediction={tuple(vehicle_logits.shape)}, "
				f"target={tuple(vehicle_target.shape)}"
			)

		# This loss is only inspected here.
		# During training it will be followed by backward().
		inspection_loss = F.binary_cross_entropy_with_logits(
			vehicle_logits,
			vehicle_target,
		)

		# Probability-weighted mean depth for every feature cell.
		depth_values = lift_output.depth_values.view(
			1,
			1,
			-1,
			1,
			1,
		)

		expected_depth = (
			encoder_output.depth_probabilities
			* depth_values
		).sum(dim=2)

		# Combine all context channels into one visualization.
		context_energy = torch.linalg.vector_norm(
			encoder_output.context,
			dim=2,
		)

	# ---------------------------------------------------------------
	# Shape diagnostics.
	# ---------------------------------------------------------------

	print("\nPipeline tensor shapes")
	print(f"Images:              {tuple(images.shape)}")
	print(
		"Context:             ",
		tuple(encoder_output.context.shape),
	)
	print(
		"Depth logits:        ",
		tuple(encoder_output.depth_logits.shape),
	)
	print(
		"Depth probabilities: ",
		tuple(encoder_output.depth_probabilities.shape),
	)
	print(
		"Camera points:       ",
		tuple(lift_output.points_camera.shape),
	)
	print(
		"Lifted features:     ",
		tuple(lift_output.features.shape),
	)
	print(
		"Ego points:          ",
		tuple(points_ego.shape),
	)
	print(
		"BEV features:        ",
		tuple(bev_output.features.shape),
	)
	print(
		"BEV counts:          ",
		tuple(bev_output.counts.shape),
	)
	print(
		"Vehicle logits:      ",
		tuple(vehicle_logits.shape),
	)
	print(
		"Vehicle target:      ",
		tuple(vehicle_target.shape),
	)

	# ---------------------------------------------------------------
	# Numerical diagnostics.
	# ---------------------------------------------------------------

	probability_error = (
		encoder_output.depth_probabilities
		.sum(dim=2)
		.sub(1.0)
		.abs()
		.max()
		.item()
	)

	nonempty_cells = (
		bev_output.counts > 0
	).sum().item()

	total_cells = bev_output.counts.numel()

	positive_target_cells = (
		vehicle_target > 0.5
	).sum().item()

	print("\nNumerical diagnostics")
	print(
		"Maximum probability sum error:",
		f"{probability_error:.8f}",
	)
	print(
		"Nonempty BEV cells:",
		f"{nonempty_cells}/{total_cells}",
	)
	print(
		"Positive vehicle target cells:",
		positive_target_cells,
	)
	print(
		"Vehicle prediction range:",
		f"{vehicle_probabilities.min().item():.6f} to "
		f"{vehicle_probabilities.max().item():.6f}",
	)
	print(
		"Mean vehicle probability:",
		f"{vehicle_probabilities.mean().item():.6f}",
	)
	print(
		"Inspection BCE loss:",
		f"{inspection_loss.item():.6f}",
	)

	# ---------------------------------------------------------------
	# Parameter diagnostics.
	# ---------------------------------------------------------------

	print("\nTrainable parameters")
	print(
		"Image encoder:",
		f"{count_trainable_parameters(image_encoder):,}",
	)
	print(
		"BEV decoder:",
		f"{count_trainable_parameters(bev_decoder):,}",
	)
	print(
		"Lift geometry:",
		f"{count_trainable_parameters(lift_geometry):,}",
	)
	print(
		"Camera-to-ego:",
		f"{count_trainable_parameters(camera_to_ego):,}",
	)
	print(
		"BEV pool:",
		f"{count_trainable_parameters(bev_pool):,}",
	)

	# ---------------------------------------------------------------
	# Inspect central viewing directions.
	# ---------------------------------------------------------------

	feature_height = points_ego.shape[3]
	feature_width = points_ego.shape[4]

	center_row = feature_height // 2
	center_column = feature_width // 2

	print(
		"\nCentral camera-ray directions "
		"in ego coordinates:"
	)

	for current_camera_index, camera_name in enumerate(
		camera_names
	):
		# Select the first depth candidate of the central ray.
		point_ego = points_ego[
			0,
			current_camera_index,
			0,
			center_row,
			center_column,
		]

		camera_origin = camera_translations[
			0,
			current_camera_index,
		]

		direction = point_ego - camera_origin

		direction = direction / torch.linalg.vector_norm(
			direction
		).clamp_min(1e-8)

		print(
			f"{camera_name:18s}",
			direction.detach().cpu().numpy(),
		)

	# ---------------------------------------------------------------
	# Move inspection images to the CPU.
	# ---------------------------------------------------------------

	context_energy_image = (
		context_energy[
			0,
			camera_index,
		]
		.detach()
		.cpu()
		.numpy()
	)

	expected_depth_image = (
		expected_depth[
			0,
			camera_index,
		]
		.detach()
		.cpu()
		.numpy()
	)

	bev_occupancy_image = (
		torch.log1p(
			bev_output.counts[0, 0]
		)
		.detach()
		.cpu()
		.numpy()
	)

	vehicle_target_image = (
		vehicle_target[0, 0]
		.detach()
		.cpu()
		.numpy()
	)

	vehicle_probability_image = (
		vehicle_probabilities[0, 0]
		.detach()
		.cpu()
		.numpy()
	)

	bev_extent = [
		BEV_Y_MINIMUM,
		BEV_Y_MAXIMUM,
		BEV_X_MINIMUM,
		BEV_X_MAXIMUM,
	]

	# ---------------------------------------------------------------
	# Visualize the complete inspection.
	# ---------------------------------------------------------------

	figure, axes = plt.subplots(
		2,
		3,
		figsize=(18, 11),
	)

	# Original camera image.
	axes[0, 0].imshow(raw_image_rgb)
	axes[0, 0].set_title(arguments.camera)
	axes[0, 0].axis("off")

	# Encoder context.
	context_plot = axes[0, 1].imshow(
		context_energy_image,
		cmap="turbo",
	)
	axes[0, 1].set_title(
		"Context feature energy\n"
		f"{CONTEXT_CHANNELS} channels combined"
	)
	axes[0, 1].axis("off")

	figure.colorbar(
		context_plot,
		ax=axes[0, 1],
		fraction=0.046,
	)

	# Predicted depth.
	depth_plot = axes[0, 2].imshow(
		expected_depth_image,
		cmap="viridis",
		vmin=DEPTH_MINIMUM,
		vmax=DEPTH_MAXIMUM,
	)
	axes[0, 2].set_title(
		"Expected depth\n"
		"Untrained image encoder"
	)
	axes[0, 2].axis("off")

	figure.colorbar(
		depth_plot,
		ax=axes[0, 2],
		label="Depth [m]",
		fraction=0.046,
	)

	# Lifted/splatted geometry occupancy.
	occupancy_plot = axes[1, 0].imshow(
		bev_occupancy_image,
		origin="lower",
		extent=bev_extent,
		cmap="turbo",
		aspect="equal",
	)
	axes[1, 0].scatter(
		0.0,
		0.0,
		color="white",
		edgecolor="black",
		marker="^",
		s=100,
		label="Ego vehicle",
	)
	axes[1, 0].set_title(
		"Lifted samples in BEV"
	)
	axes[1, 0].set_xlabel(
		"Ego Y — left [m]"
	)
	axes[1, 0].set_ylabel(
		"Ego X — forward [m]"
	)
	axes[1, 0].legend()

	figure.colorbar(
		occupancy_plot,
		ax=axes[1, 0],
		label="log(1 + sample count)",
		fraction=0.046,
	)

	# Ground-truth vehicle mask.
	target_plot = axes[1, 1].imshow(
		vehicle_target_image,
		origin="lower",
		extent=bev_extent,
		cmap="gray",
		vmin=0.0,
		vmax=1.0,
		aspect="equal",
	)
	axes[1, 1].scatter(
		0.0,
		0.0,
		color="red",
		marker="^",
		s=80,
	)
	axes[1, 1].set_title(
		"nuScenes vehicle target"
	)
	axes[1, 1].set_xlabel(
		"Ego Y — left [m]"
	)
	axes[1, 1].set_ylabel(
		"Ego X — forward [m]"
	)

	figure.colorbar(
		target_plot,
		ax=axes[1, 1],
		label="Vehicle occupancy",
		fraction=0.046,
	)

	# Untrained decoder prediction.
	prediction_plot = axes[1, 2].imshow(
		vehicle_probability_image,
		origin="lower",
		extent=bev_extent,
		cmap="magma",
		vmin=0.0,
		vmax=1.0,
		aspect="equal",
	)
	axes[1, 2].scatter(
		0.0,
		0.0,
		color="cyan",
		marker="^",
		s=80,
	)
	axes[1, 2].set_title(
		"Vehicle probability\n"
		f"Untrained decoder, BCE={inspection_loss.item():.4f}"
	)
	axes[1, 2].set_xlabel(
		"Ego Y — left [m]"
	)
	axes[1, 2].set_ylabel(
		"Ego X — forward [m]"
	)

	figure.colorbar(
		prediction_plot,
		ax=axes[1, 2],
		label="Predicted probability",
		fraction=0.046,
	)

	figure.suptitle(
		f"LSS pipeline inspection — {scene['name']} — "
		f"frame {arguments.frame_index}",
		fontsize=15,
	)

	figure.tight_layout()
	plt.show()

	print("\nInspection completed")


if __name__ == "__main__":
	main()