
import argparse
from pathlib import Path
import os

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from utils.data.data_input import (
	ordered_scene_samples,
	load_camera_image,
	create_camera_tile,
	build_mosaic,

)
import torch
from torch import nn

from lss.image_encoder import ConvNormActivation
from utils.command_utils import (
	parse_arguments
)

from dotenv import load_dotenv

# read the environment variables
ENV_FILE = (
	Path(__file__).resolve().parents[1]
	/ ".env"
)

load_dotenv(ENV_FILE)

NUSCENES_ROOT = os.environ["NUSCENES_ROOT"]


def main() -> None:
	arguments = parse_arguments()
	if arguments.fps <= 0.0:
		raise ValueError("--fps must be greater than zero.")
	if arguments.tile_width < 160:
		raise ValueError("--tile-width must be at least 160 pixels.")

	nuscenes = NuScenes(
		version="v1.0-mini",
		dataroot=str(NUSCENES_ROOT),
		verbose=True,
	)

	if not 0 <= arguments.scene_index < len(nuscenes.scene):
		raise IndexError(
			f"--scene-index must be between 0 and {len(nuscenes.scene) - 1}."
		)

	scene = nuscenes.scene[arguments.scene_index]
	samples = ordered_scene_samples(nuscenes, scene)
	start_timestamp = samples[0]["timestamp"]
	delay_ms = max(1, int(round(1000.0 / arguments.fps)))
	window_name = "nuScenes chronological six-camera scene"

	print(f"Scene:       {scene['name']}")
	print(f"Description: {scene['description']}")
	print(f"Samples:     {len(samples)}")
	print("Controls: SPACE = pause/resume, Q or ESC = quit")

	video_writer: cv2.VideoWriter | None = None
	paused = False
	frame_index = 0

	feature_extractor = ConvNormActivation(
		input_channels=3,
		output_channels=8,
		kernel_size=3,
		stride=1,
	)

	feature_extractor.eval()

	try:
		while True:
			sample = samples[frame_index]
			front_image_bgr = load_camera_image(
				nuscenes=nuscenes,
				sample=sample,
				camera_name="CAM_FRONT",
			)

			front_image_rgb = cv2.cvtColor(
				front_image_bgr,
				cv2.COLOR_BGR2RGB,
			)

			image_tensor = torch.from_numpy(
				front_image_rgb.copy()
			).permute(2, 0, 1).unsqueeze(0).float()

			image_tensor = image_tensor / 255.0

			with torch.no_grad():
				feature_maps = feature_extractor(
					image_tensor
				)
			print("Image tensor:", image_tensor.shape)
			print("Feature maps:", feature_maps.shape)

			first_feature_map = (
				feature_maps[0, 0]
				.detach()
				.cpu()
				.numpy()
			)

			# Convert the arbitrary feature values into [0, 255]
			# so OpenCV can display them.
			first_feature_map_display = cv2.normalize(
				first_feature_map,
				None,
				alpha=0,
				beta=255,
				norm_type=cv2.NORM_MINMAX,
			).astype(np.uint8)

			# Add colors to make the activation easier to inspect.
			first_feature_map_display = cv2.applyColorMap(
				first_feature_map_display,
				cv2.COLORMAP_TURBO,
			)

			cv2.imshow(
				"CAM_FRONT original",
				front_image_bgr,
			)

			cv2.imshow(
				"ConvNormActivation channel 0",
				first_feature_map_display,
			)


			mosaic = build_mosaic(
				nuscenes=nuscenes,
				sample=sample,
				tile_width=arguments.tile_width,
				frame_index=frame_index,
				frame_count=len(samples),
				scene_name=scene["name"],
				start_timestamp=start_timestamp,
			)



			# if arguments.save_video is not None and video_writer is None:
			# 	video_path = arguments.save_video.expanduser().resolve()
			# 	video_path.parent.mkdir(parents=True, exist_ok=True)
			# 	video_writer = cv2.VideoWriter(
			# 		str(video_path),
			# 		cv2.VideoWriter_fourcc(*"mp4v"),
			# 		arguments.fps,
			# 		(mosaic.shape[1], mosaic.shape[0]),
			# 	)
			# 	if not video_writer.isOpened():
			# 		raise RuntimeError(f"Could not create video: {video_path}")

			cv2.imshow(window_name, mosaic)
			# if video_writer is not None:
			# 	video_writer.write(mosaic)

			key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
			if key in (ord("q"), 27):
				break
			if key == ord(" "):
				paused = not paused
				continue

			frame_index += 1
			if frame_index >= len(samples):
				if arguments.loop:
					frame_index = 0
				else:
					break
	finally:
		# if video_writer is not None:
		# 	video_writer.release()
		cv2.destroyAllWindows()

	# if arguments.save_video is not None:
		# print(f"Video saved to: {arguments.save_video.expanduser().resolve()}")


if __name__ == "__main__":
	main()

