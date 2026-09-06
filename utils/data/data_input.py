import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pathlib import Path
import os
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

CAMERA_LAYOUT = (
	("CAM_FRONT_LEFT", "FRONT LEFT"),
	("CAM_FRONT", "FRONT"),
	("CAM_FRONT_RIGHT", "FRONT RIGHT"),
	("CAM_BACK_LEFT", "BACK LEFT"),
	("CAM_BACK", "BACK"),
	("CAM_BACK_RIGHT", "BACK RIGHT"),
)


def required_path(variable_name: str) -> Path:
	value = os.getenv(variable_name)

	if value is None:
		raise RuntimeError(
			f"{variable_name} is missing. "
			"Create a .env file using .env.example."
		)

	path = Path(value).expanduser().resolve()

	if not path.exists():
		raise FileNotFoundError(
			f"{variable_name} does not exist: {path}"
		)

	return path


def ordered_scene_samples(
	nuscenes: NuScenes,
	scene: dict,
) -> list[dict]:
	"""Follow the scene's linked sample list in chronological order."""
	samples: list[dict] = []
	sample_token = scene["first_sample_token"]

	while sample_token:
		sample = nuscenes.get("sample", sample_token)
		samples.append(sample)
		sample_token = sample["next"]

	if not samples:
		raise RuntimeError("The selected scene contains no samples.")
	if samples[-1]["token"] != scene["last_sample_token"]:
		raise RuntimeError("The sample chain did not reach last_sample_token.")

	return samples

def load_camera_image(
	nuscenes: NuScenes,
	sample: dict,
	camera_name: str,
) -> np.ndarray:
	sample_data_token = sample["data"][camera_name]
	image_path = Path(nuscenes.get_sample_data_path(sample_data_token))
	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

	if image is None:
		raise FileNotFoundError(f"Could not read camera image: {image_path}")

	return image

def create_camera_tile(
	image: np.ndarray,
	label: str,
	tile_width: int,
) -> np.ndarray:
	height, width = image.shape[:2]
	tile_height = int(round(height * tile_width / width))
	resized = cv2.resize(
		image,
		(tile_width, tile_height),
		interpolation=cv2.INTER_AREA,
	)

	tile = cv2.copyMakeBorder(
		resized,
		34,
		0,
		0,
		0,
		cv2.BORDER_CONSTANT,
		value=(20, 20, 20),
	)
	cv2.putText(
		tile,
		label,
		(10, 24),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		(255, 255, 255),
		2,
		cv2.LINE_AA,
	)
	return tile

def build_mosaic(
	nuscenes: NuScenes,
	sample: dict,
	tile_width: int,
	frame_index: int,
	frame_count: int,
	scene_name: str,
	start_timestamp: int,
) -> np.ndarray:
	tiles = []
	for camera_name, label in CAMERA_LAYOUT:
		image = load_camera_image(nuscenes, sample, camera_name)
		tiles.append(create_camera_tile(image, label, tile_width))

	mosaic = np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:])))
	mosaic = cv2.copyMakeBorder(
		mosaic,
		54,
		0,
		0,
		0,
		cv2.BORDER_CONSTANT,
		value=(8, 8, 8),
	)

	elapsed_seconds = (sample["timestamp"] - start_timestamp) / 1_000_000.0
	header = (
		f"{scene_name} | frame {frame_index + 1}/{frame_count} | "
		f"t = {elapsed_seconds:.2f} s | SPACE pause | Q quit"
	)
	cv2.putText(
		mosaic,
		header,
		(14, 35),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.75,
		(255, 255, 255),
		2,
		cv2.LINE_AA,
	)
	return mosaic