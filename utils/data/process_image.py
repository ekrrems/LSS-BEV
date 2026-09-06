import cv2
import numpy as np
import torch
from torch import nn
from pathlib import Path
import os
from nuscenes.nuscenes import NuScenes

from utils.data.data_input import (
	CAMERA_LAYOUT,
	ordered_scene_samples,
	load_camera_image,
	create_camera_tile,
	build_mosaic,

)

from lss.model_settings import (
	IMAGE_HEIGHT,
	IMAGE_WIDTH,
)


def select_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")

	if torch.backends.mps.is_available():
		return torch.device("mps")

	return torch.device("cpu")


def prepare_camera_for_lss(
	nuscenes: NuScenes,
	sample: dict,
	camera_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
	image_bgr = load_camera_image(
		nuscenes=nuscenes,
		sample=sample,
		camera_name=camera_name,
	)

	sample_data = nuscenes.get(
		"sample_data",
		sample["data"][camera_name],
	)

	calibration = nuscenes.get(
		"calibrated_sensor",
		sample_data["calibrated_sensor_token"],
	)

	original_intrinsics = np.asarray(
		calibration["camera_intrinsic"],
		dtype=np.float32,
	).reshape(3, 3)

	original_height, original_width = (
		image_bgr.shape[:2]
	)

	# Resize while preserving the aspect ratio.
	scale = max(
		IMAGE_WIDTH / original_width,
		IMAGE_HEIGHT / original_height,
	)

	resized_width = max(
		IMAGE_WIDTH,
		int(round(original_width * scale)),
	)
	resized_height = max(
		IMAGE_HEIGHT,
		int(round(original_height * scale)),
	)

	resized = cv2.resize(
		image_bgr,
		(resized_width, resized_height),
		interpolation=cv2.INTER_AREA,
	)

	# Center crop to 704 × 256.
	crop_left = (
		resized_width - IMAGE_WIDTH
	) // 2
	crop_top = (
		resized_height - IMAGE_HEIGHT
	) // 2

	processed_bgr = resized[
		crop_top:crop_top + IMAGE_HEIGHT,
		crop_left:crop_left + IMAGE_WIDTH,
	]

	# Actual scale after integer rounding.
	scale_x = resized_width / original_width
	scale_y = resized_height / original_height

	# Update K for resize and crop.
	processed_intrinsics = (
		original_intrinsics.copy()
	)

	processed_intrinsics[0, 0] = (
		original_intrinsics[0, 0] * scale_x
	)
	processed_intrinsics[1, 1] = (
		original_intrinsics[1, 1] * scale_y
	)
	processed_intrinsics[0, 2] = (
		original_intrinsics[0, 2] * scale_x
		- crop_left
	)
	processed_intrinsics[1, 2] = (
		original_intrinsics[1, 2] * scale_y
		- crop_top
	)

	processed_rgb = cv2.cvtColor(
		processed_bgr,
		cv2.COLOR_BGR2RGB,
	)

	image_tensor = torch.from_numpy(
		processed_rgb.copy()
	).permute(2, 0, 1).float() / 255.0

	intrinsics_tensor = torch.from_numpy(
		processed_intrinsics
	).float()

	return image_tensor, intrinsics_tensor


def prepare_six_camera_batch(
	nuscenes: NuScenes,
	sample: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
	image_tensors = []
	intrinsics = []

	for camera_name, _ in CAMERA_LAYOUT:
		image_tensor, camera_intrinsics = (
			prepare_camera_for_lss(
				nuscenes=nuscenes,
				sample=sample,
				camera_name=camera_name,
			)
		)

		image_tensors.append(image_tensor)
		intrinsics.append(camera_intrinsics)

	images = torch.stack(
		image_tensors,
		dim=0,
	).unsqueeze(0)

	camera_intrinsics = torch.stack(
		intrinsics,
		dim=0,
	).unsqueeze(0)

	return images, camera_intrinsics