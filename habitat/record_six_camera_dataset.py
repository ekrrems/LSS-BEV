from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
import quaternion


# This file is intended to live in PROJECT_ROOT/habitat/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_CONFIG = (
	PROJECT_ROOT
	/ "data"
	/ "habitat"
	/ "scene_datasets"
	/ "hssd-hab"
	/ "hssd-hab.scene_dataset_config.json"
)

CAMERA_YAWS_DEGREES = {
	"camera_front": 0.0,
	"camera_front_left": 60.0,
	"camera_back_left": 120.0,
	"camera_back": 180.0,
	"camera_back_right": 240.0,
	"camera_front_right": 300.0,
}

CAMERA_DISPLAY_ORDER = (
	"camera_front_left",
	"camera_front",
	"camera_front_right",
	"camera_back_left",
	"camera_back",
	"camera_back_right",
)


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Record synchronized six-camera Habitat data."
	)
	parser.add_argument("--scene", default="102344280")
	parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_CONFIG)
	parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "habitat_dataset")
	parser.add_argument("--sequence", default="sequence_0000")
	parser.add_argument("--width", type=int, default=352)
	parser.add_argument("--height", type=int, default=256)
	parser.add_argument("--hfov", type=float, default=70.0)
	parser.add_argument("--minimum-distance", type=float, default=0.30)
	parser.add_argument("--minimum-angle", type=float, default=10.0)
	parser.add_argument(
		"--record-semantic",
		action="store_true",
		help="Also save semantic-ID images. HSSD semantic annotations must be loaded.",
	)
	return parser.parse_args()


def create_camera(
	*,
	uuid: str,
	sensor_type: habitat_sim.SensorType,
	yaw_degrees: float,
	width: int,
	height: int,
	hfov: float,
) -> habitat_sim.CameraSensorSpec:
	camera = habitat_sim.CameraSensorSpec()
	camera.uuid = uuid
	camera.sensor_type = sensor_type
	camera.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
	camera.resolution = [height, width]
	camera.position = [0.0, 1.0, 0.0]
	camera.orientation = [0.0, math.radians(yaw_degrees), 0.0]
	camera.hfov = hfov
	return camera


def create_simulator(arguments: argparse.Namespace) -> habitat_sim.Simulator:
	simulator_configuration = habitat_sim.SimulatorConfiguration()
	simulator_configuration.scene_dataset_config_file = str(arguments.dataset.resolve())
	simulator_configuration.scene_id = arguments.scene

	sensors: list[habitat_sim.CameraSensorSpec] = []
	for name, yaw_degrees in CAMERA_YAWS_DEGREES.items():
		sensors.append(
			create_camera(
				uuid=name,
				sensor_type=habitat_sim.SensorType.COLOR,
				yaw_degrees=yaw_degrees,
				width=arguments.width,
				height=arguments.height,
				hfov=arguments.hfov,
			)
		)
		sensors.append(
			create_camera(
				uuid=f"{name}_depth",
				sensor_type=habitat_sim.SensorType.DEPTH,
				yaw_degrees=yaw_degrees,
				width=arguments.width,
				height=arguments.height,
				hfov=arguments.hfov,
			)
		)
		if arguments.record_semantic:
			sensors.append(
				create_camera(
					uuid=f"{name}_semantic",
					sensor_type=habitat_sim.SensorType.SEMANTIC,
					yaw_degrees=yaw_degrees,
					width=arguments.width,
					height=arguments.height,
					hfov=arguments.hfov,
				)
			)

	action_space = {
		"move_forward": habitat_sim.agent.ActionSpec(
			"move_forward", habitat_sim.agent.ActuationSpec(amount=0.15)
		),
		"move_backward": habitat_sim.agent.ActionSpec(
			"move_backward", habitat_sim.agent.ActuationSpec(amount=0.15)
		),
		"turn_left": habitat_sim.agent.ActionSpec(
			"turn_left", habitat_sim.agent.ActuationSpec(amount=5.0)
		),
		"turn_right": habitat_sim.agent.ActionSpec(
			"turn_right", habitat_sim.agent.ActuationSpec(amount=5.0)
		),
	}

	agent_configuration = habitat_sim.agent.AgentConfiguration(
		height=1.2,
		radius=0.22,
		sensor_specifications=sensors,
		action_space=action_space,
		body_type="cylinder",
	)
	return habitat_sim.Simulator(
		habitat_sim.Configuration(simulator_configuration, [agent_configuration])
	)


def ensure_navmesh(simulator: habitat_sim.Simulator) -> None:
	pathfinder = simulator.pathfinder
	if pathfinder.is_loaded:
		return

	print("No navigation mesh was included. Generating one...")
	settings = habitat_sim.NavMeshSettings()
	settings.set_defaults()
	settings.agent_height = 1.20
	settings.agent_radius = 0.22
	settings.agent_max_climb = 0.15
	settings.agent_max_slope = 40.0

	success = simulator.recompute_navmesh(pathfinder, settings)
	if not success or not pathfinder.is_loaded:
		raise RuntimeError("Habitat could not generate a navigation mesh.")

	print(f"Navigation mesh generated: {pathfinder.navigable_area:.2f} m²")


def place_agent_randomly(simulator: habitat_sim.Simulator) -> None:
	ensure_navmesh(simulator)
	position = simulator.pathfinder.get_random_navigable_point()
	if not np.isfinite(position).all():
		raise RuntimeError("Habitat returned an invalid navigable position.")

	state = habitat_sim.AgentState()
	state.position = position
	yaw = np.random.uniform(-np.pi, np.pi)
	state.rotation = quaternion.from_rotation_vector(
		np.array([0.0, yaw, 0.0], dtype=np.float64)
	)
	simulator.get_agent(0).set_state(state, reset_sensors=True)
	print("Agent placed at:", np.round(position, 3))


def rgba_to_bgr(image: np.ndarray) -> np.ndarray:
	image = np.asarray(image)
	if image.shape[-1] == 4:
		return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
	return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
	result = image.copy()
	cv2.rectangle(result, (0, 0), (result.shape[1], 32), (20, 20, 20), -1)
	cv2.putText(
		result,
		label,
		(8, 23),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.55,
		(255, 255, 255),
		1,
		cv2.LINE_AA,
    )
	return result


def build_mosaic(observations: dict) -> np.ndarray:
	tiles = [
		add_label(rgba_to_bgr(observations[name]), name)
		for name in CAMERA_DISPLAY_ORDER
	]
	return np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:])))


def pose_matrix(position: np.ndarray, rotation: np.quaternion) -> np.ndarray:
	transform = np.eye(4, dtype=np.float64)
	transform[:3, :3] = quaternion.as_rotation_matrix(rotation)
	transform[:3, 3] = np.asarray(position, dtype=np.float64)
	return transform


def intrinsic_matrix(width: int, height: int, hfov_degrees: float) -> np.ndarray:
	focal_length = 0.5 * width / math.tan(0.5 * math.radians(hfov_degrees))
	return np.array(
		[
			[focal_length, 0.0, (width - 1.0) / 2.0],
			[0.0, focal_length, (height - 1.0) / 2.0],
			[0.0, 0.0, 1.0],
		],
		dtype=np.float64,
	)


def rotation_difference_degrees(first: np.quaternion, second: np.quaternion) -> float:
	first_rotation = quaternion.as_rotation_matrix(first)
	second_rotation = quaternion.as_rotation_matrix(second)
	relative = first_rotation.T @ second_rotation
	cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
	return float(np.degrees(np.arccos(cosine)))


def prepare_output(arguments: argparse.Namespace) -> Path:
	sequence_directory = (
		arguments.output.expanduser().resolve()
		/ f"scene_{arguments.scene}"
		/ arguments.sequence
	)
	sequence_directory.mkdir(parents=True, exist_ok=True)

	manifest_path = sequence_directory / "manifest.jsonl"
	if manifest_path.exists() and manifest_path.stat().st_size > 0:
		raise FileExistsError(
			f"Sequence already contains recorded frames: {sequence_directory}. "
			"Choose another value with --sequence to avoid overwriting it."
		)

	calibration = {
		"scene": arguments.scene,
		"width": arguments.width,
		"height": arguments.height,
		"horizontal_fov_degrees": arguments.hfov,
		"K": intrinsic_matrix(arguments.width, arguments.height, arguments.hfov).tolist(),
		"camera_yaws_degrees": CAMERA_YAWS_DEGREES,
		"habitat_axes": {"x": "right", "y": "up", "negative_z": "forward"},
		"depth_unit": "metres",
	}
	with (sequence_directory / "calibration.json").open("w", encoding="utf-8") as file:
		json.dump(calibration, file, indent=2)

	return sequence_directory


def save_frame(
	*,
	simulator: habitat_sim.Simulator,
	observations: dict,
	sequence_directory: Path,
	frame_index: int,
	segment_index: int,
	record_semantic: bool,
) -> None:
	frame_name = f"{frame_index:06d}"
	agent_state = simulator.get_agent(0).get_state()
	world_from_agent = pose_matrix(agent_state.position, agent_state.rotation)

	record = {
		"frame_index": frame_index,
		"segment_index": segment_index,
		"world_from_agent": world_from_agent.tolist(),
		"cameras": {},
	}

	for camera_name in CAMERA_DISPLAY_ORDER:
		rgb_directory = sequence_directory / "rgb" / camera_name
		depth_directory = sequence_directory / "depth" / camera_name
		rgb_directory.mkdir(parents=True, exist_ok=True)
		depth_directory.mkdir(parents=True, exist_ok=True)

		rgb_path = rgb_directory / f"{frame_name}.png"
		depth_path = depth_directory / f"{frame_name}.npy"

		rgb = rgba_to_bgr(observations[camera_name])
		depth = np.asarray(observations[f"{camera_name}_depth"], dtype=np.float32)
		cv2.imwrite(str(rgb_path), rgb)
		np.save(depth_path, depth)

		camera_state = agent_state.sensor_states[camera_name]
		world_from_camera = pose_matrix(camera_state.position, camera_state.rotation)
		agent_from_camera = np.linalg.inv(world_from_agent) @ world_from_camera

		camera_record = {
			"rgb": str(rgb_path.relative_to(sequence_directory)),
			"depth": str(depth_path.relative_to(sequence_directory)),
			"world_from_camera": world_from_camera.tolist(),
			"agent_from_camera": agent_from_camera.tolist(),
			"valid_depth_pixels": int(np.isfinite(depth).sum()),
		}

		if record_semantic:
			semantic_directory = sequence_directory / "semantic" / camera_name
			semantic_directory.mkdir(parents=True, exist_ok=True)
			semantic_path = semantic_directory / f"{frame_name}.npy"
			semantic = np.asarray(
				observations[f"{camera_name}_semantic"], dtype=np.int32
			)
			np.save(semantic_path, semantic)
			camera_record["semantic"] = str(
				semantic_path.relative_to(sequence_directory)
			)
			camera_record["semantic_ids"] = np.unique(semantic).tolist()

		record["cameras"][camera_name] = camera_record

	with (sequence_directory / "manifest.jsonl").open("a", encoding="utf-8") as file:
		file.write(json.dumps(record) + "\n")

	print(f"Saved frame {frame_name} (segment {segment_index})")


def main() -> None:
	arguments = parse_arguments()
	arguments.dataset = arguments.dataset.expanduser().resolve()
	if not arguments.dataset.exists():
		raise FileNotFoundError(f"Dataset config does not exist: {arguments.dataset}")

	simulator = create_simulator(arguments)
	sequence_directory = prepare_output(arguments)

	print("\nControls")
	print("  W/S: move forward/backward")
	print("  A/D: rotate left/right")
	print("  E: force-save the current frame")
	print("  R: start a new segment at a random navigable position")
	print("  P: print the agent pose")
	print("  Q or ESC: finish")
	print(f"\nRecording to: {sequence_directory}")

	try:
		place_agent_randomly(simulator)
		observations = simulator.get_sensor_observations()
		frame_index = 0
		segment_index = 0

		state = simulator.get_agent(0).get_state()
		last_saved_position = np.asarray(state.position, dtype=np.float64).copy()
		last_saved_rotation = state.rotation

		save_frame(
			simulator=simulator,
			observations=observations,
			sequence_directory=sequence_directory,
			frame_index=frame_index,
			segment_index=segment_index,
			record_semantic=arguments.record_semantic,
		)
		frame_index += 1

		while True:
			cv2.imshow("HSSD dataset recorder", build_mosaic(observations))
			key = cv2.waitKey(30) & 0xFF

			if key in (ord("q"), 27):
				break

			action = None
			if key == ord("w"):
				action = "move_forward"
			elif key == ord("s"):
				action = "move_backward"
			elif key == ord("a"):
				action = "turn_left"
			elif key == ord("d"):
				action = "turn_right"
			elif key == ord("p"):
				current = simulator.get_agent(0).get_state()
				print("Position:", current.position)
				print("Rotation:", current.rotation)
			elif key == ord("e"):
				save_frame(
					simulator=simulator,
					observations=observations,
					sequence_directory=sequence_directory,
					frame_index=frame_index,
					segment_index=segment_index,
					record_semantic=arguments.record_semantic,
				)
				current = simulator.get_agent(0).get_state()
				last_saved_position = np.asarray(current.position, dtype=np.float64).copy()
				last_saved_rotation = current.rotation
				frame_index += 1
			elif key == ord("r"):
				segment_index += 1
				place_agent_randomly(simulator)
				observations = simulator.get_sensor_observations()
				current = simulator.get_agent(0).get_state()
				last_saved_position = np.asarray(current.position, dtype=np.float64).copy()
				last_saved_rotation = current.rotation
				save_frame(
					simulator=simulator,
					observations=observations,
					sequence_directory=sequence_directory,
					frame_index=frame_index,
					segment_index=segment_index,
					record_semantic=arguments.record_semantic,
				)
				frame_index += 1
				continue

			if action is None:
				continue

			observations = simulator.step(action)
			current = simulator.get_agent(0).get_state()
			distance = float(
				np.linalg.norm(
					np.asarray(current.position, dtype=np.float64)
					- last_saved_position
				)
			)
			angle = rotation_difference_degrees(last_saved_rotation, current.rotation)

			if distance >= arguments.minimum_distance or angle >= arguments.minimum_angle:
				save_frame(
					simulator=simulator,
					observations=observations,
					sequence_directory=sequence_directory,
					frame_index=frame_index,
					segment_index=segment_index,
					record_semantic=arguments.record_semantic,
				)
				last_saved_position = np.asarray(current.position, dtype=np.float64).copy()
				last_saved_rotation = current.rotation
				frame_index += 1

	finally:
		simulator.close()
		cv2.destroyAllWindows()

	print(f"\nFinished. Saved {frame_index} synchronized frames.")
	print(f"Dataset: {sequence_directory}")


if __name__ == "__main__":
	main()
