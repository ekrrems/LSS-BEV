import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

CAMERAS = (
	"camera_front_left",
	"camera_front",
	"camera_front_right",
	"camera_back_left",
	"camera_back",
	"camera_back_right",
)

def load_manifest(sequence_directory: Path) -> list[dict]:
	manifest_path = sequence_directory / "manifest.jsonl"

	with manifest_path.open("r", encoding="utf-8") as file:
		return [
			json.loads(line) for line in file if line.strip()
		]


def load_calibration(sequence_directory: Path) -> dict:
	calibration_path = sequence_directory / "calibration.json"
	with calibration_path.open("r", encoding="utf-8") as file:
		return json.load(file)

def load_rgb_and_depth(
		sequence_directory: Path,
		*,
		frame_index: int,
		camera_name: str = 'camera_front'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

	sequence_directory = sequence_directory.expanduser().resolve()

	records = load_manifest(sequence_directory)
	calibration = load_calibration(sequence_directory)

	if not 0 <= frame_index < len(records):
		raise IndexError(
			f"frame_index must be between 0 and {len(records) - 1}"
		)

	record = records[frame_index]

	if camera_name not in record["cameras"]:
		raise KeyError(f"Unknown camera: {camera_name}")

	camera_record = record["cameras"][camera_name]

	rgb_bgr = cv2.imread(
		str(sequence_directory / camera_record["rgb"]),
		cv2.IMREAD_COLOR,
	)

	if rgb_bgr is None:
		raise FileNotFoundError(camera_record["rgb"])

	rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

	depth = np.load(
		sequence_directory / camera_record["depth"]
	).astype(np.float32)

	intrinsic = np.asarray(
		calibration["K"],
		dtype=np.float64,
	)

	agent_from_camera = np.asarray(
		camera_record["agent_from_camera"],
		dtype=np.float64,
	)

	return rgb, depth, intrinsic, agent_from_camera

def load_agent_pose(
	sequence_directory: Path,
	frame_index: int,
) -> np.ndarray:
	"""Return T_world_from_agent with shape [4, 4]."""

	sequence_directory = (
		sequence_directory.expanduser().resolve()
	)

	records = load_manifest(sequence_directory)

	if not 0 <= frame_index < len(records):
		raise IndexError(
			f"frame_index must be between "
			f"0 and {len(records) - 1}"
		)

	world_from_agent = np.asarray(
		records[frame_index]["world_from_agent"],
		dtype=np.float64,
	)

	if world_from_agent.shape != (4, 4):
		raise ValueError(
			"world_from_agent must have shape [4, 4], "
			f"but received {world_from_agent.shape}"
		)

	return world_from_agent


def depth_to_colored_points(
	rgb: np.ndarray,
	depth: np.ndarray,
	intrinsic: np.ndarray,
	agent_from_camera: np.ndarray,
	*,
	pixel_stride: int = 3,
	minimum_depth: float = 0.2,
	maximum_depth: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Returns:
		points_ego: [M, 3] using:
			X = forward
			Y = left
			Z = up

		colors: [M, 3] in the range [0, 1]
	"""
	rows = np.arange(
		0,
		depth.shape[0],
		pixel_stride,
	)

	columns = np.arange(
		0,
		depth.shape[1],
		pixel_stride,
	)

	u, v = np.meshgrid(columns, rows)

	sampled_depth = depth[
		::pixel_stride,
		::pixel_stride
	]

	sampled_colors = rgb[
		::pixel_stride,
		::pixel_stride
	]

	valid = (
		np.isfinite(sampled_depth)
		& (sampled_depth >= minimum_depth)
		& (sampled_depth <= maximum_depth)
	)

	u = u[valid].astype(np.float64)
	v = v[valid].astype(np.float64)
	z_forward = sampled_depth[valid].astype(np.float64)

	fx = intrinsic[0, 0]
	fy = intrinsic[1, 1]
	cx = intrinsic[0, 2]
	cy = intrinsic[1, 2]

	# Conventional computer-vision camera coordinates.
	x_right = (u - cx) * z_forward / fx
	y_down = (v - cy) * z_forward / fy

	# Habitat camera coordinates:
	# +X right, +Y up, -Z forward.
	points_camera = np.stack(
		(
			x_right,
			-y_down,
			-z_forward,
			np.ones_like(z_forward),
		),
		axis=1,
	)

	# Camera coordinates -> agent coordinates.
	points_agent = (
		agent_from_camera @ points_camera.T
	).T[:, :3]

	# Habitat agent coordinates -> intuitive ego coordinates.
	points_ego = np.stack(
		(
			-points_agent[:, 2],  # forward
			-points_agent[:, 0],  # left
			points_agent[:, 1],   # up
		),
		axis=1,
	)

	colors = (
		sampled_colors[valid].astype(np.float32)
		/ 255.0
	)

	return points_ego, colors

def show_world_point_cloud(
	sequence_directory: Path,
	*,
	frame_index: int = 0,
	camera_name: str = "camera_front",
) -> None:
	points_world, colors = load_world_point_cloud(
		sequence_directory,
		frame_index=frame_index,
		camera_name=camera_name,
		pixel_stride=2,
		minimum_depth=0.2,
		maximum_depth=15.0,
	)

	world_from_agent = load_agent_pose(
		sequence_directory,
		frame_index,
	)

	agent_position_world = world_from_agent[:3, 3]

	figure = plt.figure(figsize=(12, 9))
	axes = figure.add_subplot(
		1,
		1,
		1,
		projection="3d",
	)

	axes.scatter(
		points_world[:, 0],
		points_world[:, 1],
		points_world[:, 2],
		c=colors,
		s=1,
		depthshade=False,
	)

	axes.scatter(
		agent_position_world[0],
		agent_position_world[1],
		agent_position_world[2],
		color="red",
		s=100,
		marker="^",
		label="Agent",
	)

	axes.set_xlabel("World X [m]")
	axes.set_ylabel("World Y [m]")
	axes.set_zlabel("World Z [m]")
	axes.set_title(
		f"{camera_name} world point cloud\n"
		f"frame {frame_index}, {len(points_world):,} points"
	)
	axes.legend()

	plt.tight_layout()
	plt.show()


def load_camera_frame(
	sequence_directory: Path,
	record: dict,
	calibration: dict,
	camera_name: str,
) -> tuple[
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
]:
	"""
	Returns:
		rgb:                 [H, W, 3], RGB uint8
		depth:               [H, W], metric depth in metres
		intrinsic:           [3, 3]
		world_from_agent:    [4, 4]
		agent_from_camera:   [4, 4]
	"""
	camera_record = record["cameras"][camera_name]

	rgb_bgr = cv2.imread(
		str(sequence_directory / camera_record["rgb"]),
		cv2.IMREAD_COLOR,
	)

	if rgb_bgr is None:
		raise FileNotFoundError(camera_record["rgb"])

	rgb = cv2.cvtColor(
		rgb_bgr,
		cv2.COLOR_BGR2RGB,
	)

	depth = np.load(
		sequence_directory / camera_record["depth"]
	).astype(np.float32)

	intrinsic = np.asarray(
		calibration["K"],
		dtype=np.float64,
	)

	world_from_agent = np.asarray(
		record["world_from_agent"],
		dtype=np.float64,
	)

	agent_from_camera = np.asarray(
		camera_record["agent_from_camera"],
		dtype=np.float64,
	)

	return (
		rgb,
		depth,
		intrinsic,
		world_from_agent,
		agent_from_camera,
	)


def rgbd_to_world_points(
	rgb: np.ndarray,
	depth: np.ndarray,
	intrinsic: np.ndarray,
	world_from_agent: np.ndarray,
	agent_from_camera: np.ndarray,
	*,
	minimum_depth: float = 0.2,
	maximum_depth: float = 15.0,
	pixel_stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Convert one Habitat RGB-depth camera observation into colored
	3D points expressed in world coordinates.

	Returns:
		points_world: [M, 3]
		colors:       [M, 3], values in [0, 1]
	"""
	height, width = depth.shape

	v_grid, u_grid = np.meshgrid(
		np.arange(0, height, pixel_stride),
		np.arange(0, width, pixel_stride),
		indexing="ij",
	)

	sampled_depth = depth[
		::pixel_stride,
		::pixel_stride,
	]

	sampled_rgb = rgb[
		::pixel_stride,
		::pixel_stride,
	]

	valid = (
		np.isfinite(sampled_depth)
		& (sampled_depth >= minimum_depth)
		& (sampled_depth <= maximum_depth)
	)

	u = u_grid[valid].astype(np.float64)
	v = v_grid[valid].astype(np.float64)
	z_forward = sampled_depth[valid].astype(np.float64)

	fx = intrinsic[0, 0]
	fy = intrinsic[1, 1]
	cx = intrinsic[0, 2]
	cy = intrinsic[1, 2]

	# Standard image/CV coordinates:
	# +X right, +Y down, +Z forward.
	x_right = (u - cx) * z_forward / fx
	y_down = (v - cy) * z_forward / fy

	# Convert to Habitat camera coordinates:
	# +X right, +Y up, -Z forward.
	points_camera = np.stack(
		(
			x_right,
			-y_down,
			-z_forward,
		),
		axis=1,
	)

	world_from_camera = (
		world_from_agent
		@ agent_from_camera
	)

	rotation_world_from_camera = (
		world_from_camera[:3, :3]
	)

	translation_world_from_camera = (
		world_from_camera[:3, 3]
	)

	points_world = (
		points_camera
		@ rotation_world_from_camera.T
		+ translation_world_from_camera
	)

	colors = (
		sampled_rgb[valid].astype(np.float32)
		/ 255.0
	)

	return points_world, colors

def load_world_point_cloud(
	sequence_directory: Path,
	*,
	frame_index: int,
	camera_name: str = "camera_front",
	pixel_stride: int = 2,
	minimum_depth: float = 0.2,
	maximum_depth: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Load one recorded camera frame and convert its RGB-depth image
	into colored 3D points in Habitat world coordinates.

	Returns:
		points_world: [M, 3]
		colors:       [M, 3], values in [0, 1]
	"""
	sequence_directory = (
		Path(sequence_directory)
		.expanduser()
		.resolve()
	)

	records = load_manifest(sequence_directory)
	calibration = load_calibration(sequence_directory)

	if not 0 <= frame_index < len(records):
		raise IndexError(
			f"frame_index must be between "
			f"0 and {len(records) - 1}"
		)

	# This is the record argument.
	record = records[frame_index]

	(
		rgb,
		depth,
		intrinsic,
		world_from_agent,
		agent_from_camera,
	) = load_camera_frame(
		sequence_directory=sequence_directory,
		record=record,
		calibration=calibration,
		camera_name=camera_name,
	)

	points_world, colors = rgbd_to_world_points(
		rgb=rgb,
		depth=depth,
		intrinsic=intrinsic,
		world_from_agent=world_from_agent,
		agent_from_camera=agent_from_camera,
		pixel_stride=pixel_stride,
		minimum_depth=minimum_depth,
		maximum_depth=maximum_depth,
	)


	return points_world, colors


def get_camera_pose_in_world(
	sequence_directory: Path,
	*,
	frame_index: int,
	camera_name: str,
) -> np.ndarray:
	"""
	Returns:
		world_from_camera: [4, 4]
		rotation_world_camera: [3, 3]
		translation_world_camera: [3]
	"""
	sequence_directory = Path(
		sequence_directory
	).expanduser().resolve()

	records = load_manifest(sequence_directory)

	if not 0 <= frame_index < len(records):
		raise IndexError(
			f"frame_index must be from 0 to {len(records) - 1}"
		)

	record = records[frame_index]

	if camera_name not in record["cameras"]:
		raise KeyError(
			f"Unknown camera {camera_name!r}. "
			f"Available cameras: {list(record['cameras'])}"
		)

	world_from_agent = np.asarray(
		record["world_from_agent"],
		dtype=np.float64,
	)

	agent_from_camera = np.asarray(
		record["cameras"][camera_name][
			"agent_from_camera"
		],
		dtype=np.float64,
	)

	world_from_camera = (
		world_from_agent
		@ agent_from_camera
	)

	# rotation_world_camera = (
	# 	world_from_camera[:3, :3]
	# )

	# translation_world_camera = (
	# 	world_from_camera[:3, 3]
	# )

	return world_from_camera

def world_points_to_agent(
    points_world: np.ndarray,
    world_from_agent: np.ndarray,
) -> np.ndarray:
    rotation_world_from_agent = world_from_agent[:3, :3]
    position_world_agent = world_from_agent[:3, 3]

    # Row-vector form of:
    # p_A = R_W_A.T @ (p_W - t_W_A)
    return (
        points_world - position_world_agent
    ) @ rotation_world_from_agent

def show_agent_top_down(
    points_world: np.ndarray,
    colors: np.ndarray,
    world_from_agent: np.ndarray,
    maximum_points: int = 100_000,
) -> None:
    points_agent = world_points_to_agent(
        points_world,
        world_from_agent,
    )

    if len(points_agent) > maximum_points:
        indices = np.linspace(
            0,
            len(points_agent) - 1,
            maximum_points,
            dtype=np.int64,
        )
        points_agent = points_agent[indices]
        colors = colors[indices]

    # Habitat agent frame:
    # +X = right
    # +Y = up
    # -Z = forward
    right = points_agent[:, 0]
    forward = -points_agent[:, 2]

    figure, axis = plt.subplots(figsize=(10, 10))

    axis.scatter(
        right,
        forward,
        c=colors,
        s=0.4,
        linewidths=0,
    )

    axis.scatter(
        0.0,
        0.0,
        color="cyan",
        marker="^",
        s=100,
        label="Agent",
    )

    axis.set_xlabel("Agent X — right [m]")
    axis.set_ylabel("Agent forward (-Z) [m]")
    axis.set_title("Point cloud in the agent coordinate frame")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True)
    axis.legend()

    plt.show()

def show_six_depth_maps(
	camera_info: dict,
	*,
	maximum_depth: float = 15.0,
) -> None:
	figure, axes = plt.subplots(
		2,
		3,
		figsize=(16, 8),
	)

	depth_image = None

	for axis, camera_name in zip(
		axes.flat,
		CAMERAS,
	):
		depth = camera_info[camera_name]["depth"]

		depth_image = axis.imshow(
			depth,
			cmap="turbo",
			vmin=0.0,
			vmax=maximum_depth,
		)

		axis.set_title(camera_name)
		axis.set_xlabel("Pixel u")
		axis.set_ylabel("Pixel v")

	figure.colorbar(
		depth_image,
		ax=axes.ravel().tolist(),
		label="Depth [m]",
		shrink=0.85,
	)

	figure.suptitle(
		"Six synchronized Habitat depth cameras"
	)

	plt.show()

def set_equal_world_axes(
	axis,
	points_world: np.ndarray,
) -> None:
	if len(points_world) == 0:
		return

	# Matplotlib coordinates:
	#
	# displayed X = world X
	# displayed Y = world Z
	# displayed Z = world Y (height)
	displayed = np.column_stack(
		(
			points_world[:, 0],
			points_world[:, 2],
			points_world[:, 1],
		)
	)

	minimum = displayed.min(axis=0)
	maximum = displayed.max(axis=0)

	center = 0.5 * (minimum + maximum)
	radius = 0.5 * np.max(maximum - minimum)
	radius = max(radius, 1.0)

	axis.set_xlim(
		center[0] - radius,
		center[0] + radius,
	)

	axis.set_ylim(
		center[1] - radius,
		center[1] + radius,
	)

	# Height normally needs a smaller visual range.
	axis.set_zlim(
		minimum[2] - 0.5,
		maximum[2] + 0.5,
	)


def draw_pose_axes(
	axis,
	transformation: np.ndarray,
	*,
	axis_length: float = 0.5,
	label: str = "",
	draw_camera_forward: bool = False,
) -> None:
	rotation = transformation[:3, :3]
	position = transformation[:3, 3]

	# Local axes expressed in world coordinates.
	right_world = rotation[:, 0]
	up_world = rotation[:, 1]
	backward_world = rotation[:, 2]
	forward_world = -backward_world

	def draw_vector(
		vector: np.ndarray,
		color: str,
	) -> None:
		# Plot world X, world Z, world Y.
		axis.quiver(
			position[0],
			position[2],
			position[1],
			vector[0],
			vector[2],
			vector[1],
			length=axis_length,
			normalize=True,
			color=color,
		)

	# Local +X/right.
	draw_vector(right_world, "red")

	# Local +Y/up.
	draw_vector(up_world, "green")

	if draw_camera_forward:
		# Habitat camera forward is local -Z.
		draw_vector(forward_world, "blue")
	else:
		# For the ground agent, show its -Z forward direction.
		draw_vector(forward_world, "blue")

	if label:
		axis.text(
			position[0],
			position[2],
			position[1],
			label,
			fontsize=8,
		)


def show_3d_environment(
	camera_info: dict,
	points_world: np.ndarray,
	colors: np.ndarray,
	world_from_agent: np.ndarray,
	*,
	maximum_display_points: int = 120_000,
) -> None:
	if len(points_world) == 0:
		print("There are no world points to display.")
		return

	if len(points_world) > maximum_display_points:
		indices = np.linspace(
			0,
			len(points_world) - 1,
			maximum_display_points,
			dtype=np.int64,
		)

		displayed_points = points_world[indices]
		displayed_colors = colors[indices]
	else:
		displayed_points = points_world
		displayed_colors = colors

	figure = plt.figure(
		figsize=(14, 10)
	)

	axis = figure.add_subplot(
		111,
		projection="3d",
	)

	axis.scatter(
		displayed_points[:, 0],
		displayed_points[:, 2],
		displayed_points[:, 1],
		c=displayed_colors,
		s=0.4,
		depthshade=False,
	)

	agent_position = world_from_agent[:3, 3]

	axis.scatter(
		agent_position[0],
		agent_position[2],
		agent_position[1],
		color="cyan",
		marker="^",
		s=100,
		label="Agent",
	)

	draw_pose_axes(
		axis,
		world_from_agent,
		axis_length=0.8,
		label="agent",
	)

	for camera_name in CAMERAS:
		world_from_camera = camera_info[
			camera_name
		]["world_from_camera"]

		camera_position = (
			world_from_camera[:3, 3]
		)

		axis.scatter(
			camera_position[0],
			camera_position[2],
			camera_position[1],
			color="yellow",
			marker="o",
			s=25,
		)

		draw_pose_axes(
			axis,
			world_from_camera,
			axis_length=0.35,
			label=camera_name,
			draw_camera_forward=True,
		)

	axis.set_xlabel("World X [m]")
	axis.set_ylabel("World Z [m]")
	axis.set_zlabel("World Y — height [m]")

	axis.set_title(
		"Six-camera RGB-D reconstruction\n"
		f"{len(points_world):,} total points"
	)

	set_equal_world_axes(
		axis,
		displayed_points,
	)

	axis.legend()
	plt.tight_layout()
	plt.show()

def show_rgb_depth_and_3d(
	sequence_directory: Path,
	frame_index: int = 0,
	camera_name: str = "camera_front",
) -> None:
	sequence_directory = Path(sequence_directory)

	rgb, depth, intrinsic, agent_from_camera = (
		load_rgb_and_depth(
			sequence_directory,
			frame_index=frame_index,
			camera_name=camera_name,
		)
	)

	points, colors = depth_to_colored_points(
		rgb,
		depth,
		intrinsic,
		agent_from_camera,
		pixel_stride=3,
		minimum_depth=0.2,
		maximum_depth=10.0,
	)

	figure = plt.figure(figsize=(18, 6))

	rgb_axes = figure.add_subplot(1, 3, 1)
	rgb_axes.imshow(rgb)
	rgb_axes.set_title(f"{camera_name} RGB")
	rgb_axes.axis("off")

	depth_axes = figure.add_subplot(1, 3, 2)
	depth_plot = depth_axes.imshow(
		depth,
		cmap="turbo",
		vmin=0.0,
		vmax=10.0,
	)
	depth_axes.set_title("Metric depth [m]")
	depth_axes.axis("off")
	figure.colorbar(depth_plot, ax=depth_axes)

	cloud_axes = figure.add_subplot(
		1,
		3,
		3,
		projection="3d",
	)

	cloud_axes.scatter(
		points[:, 0],
		points[:, 1],
		points[:, 2],
		c=colors,
		s=1,
		depthshade=False,
	)

	cloud_axes.set_xlabel("Forward X [m]")
	cloud_axes.set_ylabel("Left Y [m]")
	cloud_axes.set_zlabel("Up Z [m]")
	cloud_axes.set_title(
		f"Colored 3D cloud\n{len(points):,} points"
	)

	cloud_axes.set_xlim(0.0, 10.0)
	cloud_axes.set_ylim(-6.0, 6.0)
	cloud_axes.set_zlim(-1.0, 3.0)
	cloud_axes.view_init(elev=20, azim=-70)

	figure.suptitle(
		f"{sequence_directory.name}, frame {frame_index}"
	)

	plt.tight_layout()
	plt.show()