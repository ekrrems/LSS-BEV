import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

from utils.data.data_input import CAMERA_LAYOUT


def prepare_six_camera_extrinsics(
	nuscenes: NuScenes,
	sample: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Return camera-to-ego transformations.

	rotations:
	    [1, 6, 3, 3]

	translations:
	    [1, 6, 3]
	"""

	rotations = []
	translations = []

	for camera_name, _ in CAMERA_LAYOUT:
		sample_data_token = sample["data"][
			camera_name
		]

		sample_data = nuscenes.get(
			"sample_data",
			sample_data_token,
		)

		calibration = nuscenes.get(
			"calibrated_sensor",
			sample_data["calibrated_sensor_token"],
		)

		rotation_ego_camera = Quaternion(
			calibration["rotation"]
		).rotation_matrix

		translation_ego_camera = np.asarray(
			calibration["translation"],
			dtype=np.float32,
		)

		rotations.append(
			rotation_ego_camera.astype(np.float32)
		)

		translations.append(
			translation_ego_camera
		)

	rotations_tensor = torch.from_numpy(
		np.stack(rotations)
	).unsqueeze(0)

	translations_tensor = torch.from_numpy(
		np.stack(translations)
	).unsqueeze(0)

	return (
		rotations_tensor,
		translations_tensor,
	)