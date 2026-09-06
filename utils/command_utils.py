import argparse
from pathlib import Path
from utils.data.data_input import (
	CAMERA_LAYOUT,
)

# def parse_arguments() -> argparse.Namespace:
# 	parser = argparse.ArgumentParser(
# 		description="Play all six nuScenes cameras in chronological scene order."
# 	)
# 	parser.add_argument(
# 		"--scene-index",
# 		type=int,
# 		default=0,
# 		help="Scene index in nuScenes Mini (usually 0 to 9).",
# 	)
# 	parser.add_argument(
# 		"--fps",
# 		type=float,
# 		default=5.0,
# 		help="Display speed. Annotated nuScenes samples are approximately 2 Hz.",
# 	)
# 	parser.add_argument(
# 		"--tile-width",
# 		type=int,
# 		default=480,
# 		help="Width of each camera tile in pixels.",
# 	)
# 	parser.add_argument(
# 		"--save-video",
# 		type=Path,
# 		default=None,
# 		help="Optional MP4 path, for example outputs/scene_00.mp4.",
# 	)
# 	parser.add_argument(
# 		"--loop",
# 		action="store_true",
# 		help="Restart after the final sample.",
# 	)
# 	return parser.parse_args()

def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Inspect LSS encoder weights and activations."
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
		"--camera",
		type=str,
		default="CAM_FRONT",
		choices=[
			camera_name
			for camera_name, _ in CAMERA_LAYOUT
		],
	)

	return parser.parse_args()