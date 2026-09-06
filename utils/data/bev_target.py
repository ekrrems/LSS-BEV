import cv2
import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

from lss.model_settings import (
	BEV_X_MINIMUM,
	BEV_X_MAXIMUM,
	BEV_Y_MINIMUM,
	BEV_Y_MAXIMUM,
	BEV_RESOLUTION,
)


def create_vehicle_bev_target(
	nuscenes: NuScenes,
	sample: dict,
) -> torch.Tensor:
	"""
	Create a binary vehicle footprint mask.

	Return shape:
	    [1, 1, X, Y]
	"""

	x_cells = int(round(
		(BEV_X_MAXIMUM - BEV_X_MINIMUM)
		/ BEV_RESOLUTION
	))

	y_cells = int(round(
		(BEV_Y_MAXIMUM - BEV_Y_MINIMUM)
		/ BEV_RESOLUTION
	))

	mask = np.zeros(
		(x_cells, y_cells),
		dtype=np.float32,
	)

	# Use LIDAR_TOP to obtain the ego pose at this sample.
	lidar_sample_data = nuscenes.get(
		"sample_data",
		sample["data"]["LIDAR_TOP"],
	)

	ego_pose = nuscenes.get(
		"ego_pose",
		lidar_sample_data["ego_pose_token"],
	)

	translation_global_ego = np.asarray(
		ego_pose["translation"],
		dtype=np.float64,
	)

	rotation_global_ego = Quaternion(
		ego_pose["rotation"]
	)

	for annotation_token in sample["anns"]:
		annotation = nuscenes.get(
			"sample_annotation",
			annotation_token,
		)

		category_name = annotation[
			"category_name"
		]

		if not category_name.startswith("vehicle."):
			continue

		# The returned box is expressed globally.
		box = nuscenes.get_box(
			annotation_token
		)

		# Global -> ego translation.
		box.translate(
			-translation_global_ego
		)

		# Global -> ego rotation.
		box.rotate(
			rotation_global_ego.inverse
		)

		# Shape: [3, 4], then [4, 2].
		corners_ego = (
			box.bottom_corners()[:2]
			.T
		)

		x_coordinates = corners_ego[:, 0]
		y_coordinates = corners_ego[:, 1]

		# Skip boxes completely outside the BEV grid.
		if (
			x_coordinates.max() < BEV_X_MINIMUM
			or x_coordinates.min() >= BEV_X_MAXIMUM
			or y_coordinates.max() < BEV_Y_MINIMUM
			or y_coordinates.min() >= BEV_Y_MAXIMUM
		):
			continue

		x_indices = (
			(x_coordinates - BEV_X_MINIMUM)
			/ BEV_RESOLUTION
		)

		y_indices = (
			(y_coordinates - BEV_Y_MINIMUM)
			/ BEV_RESOLUTION
		)

		# OpenCV expects polygon coordinates as:
		# [horizontal column, vertical row].
		#
		# Our mask uses:
		# row = ego X
		# column = ego Y
		polygon = np.stack(
			(
				y_indices,
				x_indices,
			),
			axis=1,
		)

		polygon = np.rint(
			polygon
		).astype(np.int32)

		cv2.fillPoly(
			mask,
			[polygon],
			color=1.0,
		)

	target = torch.from_numpy(mask)

	# [X, Y] -> [B=1, C=1, X, Y]
	return target.unsqueeze(0).unsqueeze(0)