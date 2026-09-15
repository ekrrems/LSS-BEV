from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from habitat.read_dataset import (
	get_camera_pose_in_world,
	load_agent_pose,
	load_rgb_and_depth,
	show_six_depth_maps,
)


CAMERAS = (
	"camera_front_left",
	"camera_front",
	"camera_front_right",
	"camera_back_left",
	"camera_back",
	"camera_back_right",
)


# Heights are measured above the floor.
HEIGHT_EDGES = np.asarray(
	[
		0.00,
		0.15,
		0.30,
		0.50,
		0.75,
		1.00,
		1.50,
		2.00,
	],
	dtype=np.float64,
)


@dataclass
class AgentBev:
	# Binary map: at least one obstacle exists in this cell.
	occupancy: np.ndarray

	# Binary map: a depth ray passed through this cell.
	free_space: np.ndarray

	# Binary map: free space or a measured surface was observed.
	observed: np.ndarray

	# Shape: [vertical_bins, rows, columns].
	vertical_occupancy: np.ndarray

	# Height information in metres above the floor.
	minimum_height: np.ndarray
	maximum_height: np.ndarray
	average_height: np.ndarray

	# Cells for which height regression is valid.
	height_valid: np.ndarray

	# Raw and normalized point density.
	density: np.ndarray
	normalized_density: np.ndarray

	# Complete model target: [channels, rows, columns].
	target: np.ndarray
	channel_names: tuple[str, ...]

	height_edges: np.ndarray

	right_minimum: float
	right_maximum: float
	forward_minimum: float
	forward_maximum: float
	resolution: float
	floor_y_agent: float


def world_points_to_agent(
	points_world: np.ndarray,
	world_from_agent: np.ndarray,
) -> np.ndarray:
	"""Convert row-vector world points into the agent coordinate frame."""

	points_world = np.asarray(
		points_world,
		dtype=np.float64,
	).reshape(-1, 3)

	world_from_agent = np.asarray(
		world_from_agent,
		dtype=np.float64,
	).reshape(4, 4)

	rotation_world_from_agent = world_from_agent[:3, :3]
	translation_world_from_agent = world_from_agent[:3, 3]

	# Column-vector equation:
	#
	# p_A = R_W_A.T @ (p_W - t_W_A)
	#
	# With row vectors, the equivalent equation is:
	#
	# p_A_row = (p_W_row - t_W_A) @ R_W_A
	return (
		points_world - translation_world_from_agent
	) @ rotation_world_from_agent


def camera_points_to_agent(
	points_camera: np.ndarray,
	agent_from_camera: np.ndarray,
) -> np.ndarray:
	"""Convert Habitat-camera points directly into agent coordinates."""

	points_camera = np.asarray(
		points_camera,
		dtype=np.float64,
	).reshape(-1, 3)

	agent_from_camera = np.asarray(
		agent_from_camera,
		dtype=np.float64,
	).reshape(4, 4)

	rotation_agent_from_camera = agent_from_camera[:3, :3]
	translation_agent_from_camera = agent_from_camera[:3, 3]

	return (
		points_camera @ rotation_agent_from_camera.T
		+ translation_agent_from_camera
	)


def metric_to_grid(
	right: np.ndarray,
	forward: np.ndarray,
	*,
	right_minimum: float,
	forward_minimum: float,
	resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
	"""Convert agent-relative metric coordinates into BEV cells."""

	columns = np.floor(
		(right - right_minimum) / resolution
	).astype(np.int64)

	rows = np.floor(
		(forward - forward_minimum) / resolution
	).astype(np.int64)

	return rows, columns


def depth_environment_create(
	sequence_directory: Path,
	frame_index: int,
	*,
	minimum_depth: float = 0.2,
	maximum_depth: float = 15.0,
	pixel_stride: int = 2,
) -> tuple[
	dict[str, dict[str, np.ndarray]],
	np.ndarray,
	np.ndarray,
	np.ndarray,
]:
	"""Reconstruct all six RGB-D cameras into one world point cloud."""

	sequence_directory = sequence_directory.expanduser().resolve()

	camera_info: dict[str, dict[str, np.ndarray]] = {}
	all_world_points: list[np.ndarray] = []
	all_colors: list[np.ndarray] = []

	world_from_agent = load_agent_pose(
		sequence_directory,
		frame_index=frame_index,
	)

	for camera_name in CAMERAS:
		(
			rgb,
			depth,
			intrinsic,
			agent_from_camera,
		) = load_rgb_and_depth(
			sequence_directory,
			frame_index=frame_index,
			camera_name=camera_name,
		)

		world_from_camera = get_camera_pose_in_world(
			sequence_directory,
			frame_index=frame_index,
			camera_name=camera_name,
		)

		rgb = np.asarray(rgb)
		depth = np.asarray(depth, dtype=np.float64)
		intrinsic = np.asarray(intrinsic, dtype=np.float64)
		agent_from_camera = np.asarray(
			agent_from_camera,
			dtype=np.float64,
		)
		world_from_camera = np.asarray(
			world_from_camera,
			dtype=np.float64,
		)

		image_height, image_width = depth.shape

		u_grid, v_grid = np.meshgrid(
			np.arange(
				0,
				image_width,
				pixel_stride,
				dtype=np.float64,
			),
			np.arange(
				0,
				image_height,
				pixel_stride,
				dtype=np.float64,
			),
			indexing="xy",
		)

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

		u = u_grid[valid]
		v = v_grid[valid]
		z_forward = sampled_depth[valid]

		fx = intrinsic[0, 0]
		fy = intrinsic[1, 1]
		cx = intrinsic[0, 2]
		cy = intrinsic[1, 2]

		# Conventional computer-vision camera coordinates:
		#
		# +X: right
		# +Y: down
		# +Z: forward
		x_right = (
			(u - cx) * z_forward / fx
		)

		y_down = (
			(v - cy) * z_forward / fy
		)

		# Habitat/OpenGL camera coordinates:
		#
		# +X: right
		# +Y: up
		# -Z: forward
		points_camera = np.column_stack(
			(
				x_right,
				-y_down,
				-z_forward,
			)
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

		colors = sampled_colors[valid].astype(
			np.float64
		) / 255.0

		camera_info[camera_name] = {
			"rgb": rgb,
			"depth": depth,
			"intrinsic": intrinsic,
			"agent_from_camera": agent_from_camera,
			"world_from_camera": world_from_camera,
			"points_camera": points_camera,
			"points_world": points_world,
			"colors": colors,
			"pixel_coordinates": np.column_stack((u, v)),
			"sampled_depth": z_forward,
		}

		all_world_points.append(points_world)
		all_colors.append(colors)

		print(
			f"{camera_name}: "
			f"{len(points_world):,} valid points"
		)

	if all_world_points:
		combined_world_points = np.vstack(all_world_points)
		combined_colors = np.vstack(all_colors)
	else:
		combined_world_points = np.empty(
			(0, 3),
			dtype=np.float64,
		)
		combined_colors = np.empty(
			(0, 3),
			dtype=np.float64,
		)

	return (
		camera_info,
		combined_world_points,
		combined_colors,
		world_from_agent,
	)


def create_visibility_maps(
	camera_info: dict[str, dict[str, np.ndarray]],
	occupancy: np.ndarray,
	*,
	right_minimum: float,
	right_maximum: float,
	forward_minimum: float,
	forward_maximum: float,
	resolution: float,
	maximum_range: float,
	free_space_ray_stride: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
	"""Use depth rays to distinguish observed free space from unknown space."""

	rows_count, columns_count = occupancy.shape

	free_space = np.zeros(
		(rows_count, columns_count),
		dtype=np.uint8,
	)

	surface_observed = np.zeros(
		(rows_count, columns_count),
		dtype=np.uint8,
	)

	for information in camera_info.values():
		points_camera = information["points_camera"]
		agent_from_camera = information["agent_from_camera"]

		points_agent = camera_points_to_agent(
			points_camera,
			agent_from_camera,
		)

		camera_origin_agent = agent_from_camera[:3, 3]

		# Habitat agent coordinates:
		#
		# +X = right
		# +Y = up
		# -Z = forward
		right = points_agent[:, 0]
		forward = -points_agent[:, 2]

		horizontal_distance = np.sqrt(
			right**2 + forward**2
		)

		valid = (
			np.isfinite(points_agent).all(axis=1)
			& (horizontal_distance <= maximum_range)
			& (right >= right_minimum)
			& (right < right_maximum)
			& (forward >= forward_minimum)
			& (forward < forward_maximum)
		)

		valid_points = points_agent[valid]

		if len(valid_points) == 0:
			continue

		end_right = valid_points[:, 0]
		end_forward = -valid_points[:, 2]

		end_rows, end_columns = metric_to_grid(
			end_right,
			end_forward,
			right_minimum=right_minimum,
			forward_minimum=forward_minimum,
			resolution=resolution,
		)

		surface_observed[
			end_rows,
			end_columns,
		] = 1

		origin_right = float(camera_origin_agent[0])
		origin_forward = float(-camera_origin_agent[2])

		origin_rows, origin_columns = metric_to_grid(
			np.asarray([origin_right]),
			np.asarray([origin_forward]),
			right_minimum=right_minimum,
			forward_minimum=forward_minimum,
			resolution=resolution,
		)

		origin_row = int(origin_rows[0])
		origin_column = int(origin_columns[0])

		if not (
			0 <= origin_row < rows_count
			and 0 <= origin_column < columns_count
		):
			continue

		# Ray tracing every depth pixel is unnecessarily expensive.
		# A subset is sufficient for a dense free-space mask.
		for end_row, end_column in zip(
			end_rows[::free_space_ray_stride],
			end_columns[::free_space_ray_stride],
		):
			cv2.line(
				free_space,
				(origin_column, origin_row),
				(int(end_column), int(end_row)),
				color=1,
				thickness=1,
				lineType=cv2.LINE_8,
			)

	# An obstacle measurement wins over free-space evidence.
	free_space[occupancy > 0.5] = 0

	observed = np.maximum(
		surface_observed,
		free_space,
	)

	observed = np.maximum(
		observed,
		occupancy.astype(np.uint8),
	)

	return (
		free_space.astype(np.float32),
		observed.astype(np.float32),
	)


def create_agent_bev(
	points_world: np.ndarray,
	world_from_agent: np.ndarray,
	camera_info: dict[str, dict[str, np.ndarray]],
	*,
	right_minimum: float = -5.0,
	right_maximum: float = 5.0,
	forward_minimum: float = -5.0,
	forward_maximum: float = 5.0,
	resolution: float = 0.10,
	maximum_range: float = 5.0,
	floor_y_agent: float = -0.20,
	floor_tolerance: float = 0.05,
	height_edges: np.ndarray = HEIGHT_EDGES,
	minimum_points_per_cell: int = 2,
	vertical_minimum_points: int = 1,
	density_reference: float = 32.0,
	free_space_ray_stride: int = 12,
) -> AgentBev:
	"""Create a multi-channel agent-relative BEV training target."""

	if resolution <= 0.0:
		raise ValueError("resolution must be positive")

	if maximum_range <= 0.0:
		raise ValueError("maximum_range must be positive")

	height_edges = np.asarray(
		height_edges,
		dtype=np.float64,
	)

	if (
		height_edges.ndim != 1
		or len(height_edges) < 2
		or np.any(np.diff(height_edges) <= 0.0)
	):
		raise ValueError(
			"height_edges must be a strictly increasing 1D array"
		)

	rows_count = int(
		np.ceil(
			(forward_maximum - forward_minimum)
			/ resolution
		)
	)

	columns_count = int(
		np.ceil(
			(right_maximum - right_minimum)
			/ resolution
		)
	)

	points_agent = world_points_to_agent(
		points_world,
		world_from_agent,
	)

	right = points_agent[:, 0]
	height_agent = points_agent[:, 1]

	# Habitat looks forward along negative agent Z.
	forward = -points_agent[:, 2]

	# Convert agent-relative Y into height above the floor.
	height_above_floor = (
		height_agent - floor_y_agent
	)

	horizontal_distance = np.sqrt(
		right**2 + forward**2
	)

	base_valid = (
		np.isfinite(points_agent).all(axis=1)
		& (horizontal_distance <= maximum_range)
		& (right >= right_minimum)
		& (right < right_maximum)
		& (forward >= forward_minimum)
		& (forward < forward_maximum)
	)

	# Floor points are useful for free-space ray tracing, but should not
	# become obstacles. Ceiling points above the configured range are
	# also excluded from the obstacle target.
	obstacle_valid = (
		base_valid
		& (height_above_floor >= floor_tolerance)
		& (height_above_floor < height_edges[-1])
	)

	valid_right = right[obstacle_valid]
	valid_forward = forward[obstacle_valid]
	valid_height = height_above_floor[obstacle_valid]

	rows, columns = metric_to_grid(
		valid_right,
		valid_forward,
		right_minimum=right_minimum,
		forward_minimum=forward_minimum,
		resolution=resolution,
	)

	density = np.zeros(
		(rows_count, columns_count),
		dtype=np.float32,
	)

	height_sum = np.zeros_like(density)

	minimum_height = np.full(
		(rows_count, columns_count),
		np.inf,
		dtype=np.float32,
	)

	maximum_height = np.full(
		(rows_count, columns_count),
		-np.inf,
		dtype=np.float32,
	)

	if len(rows) > 0:
		np.add.at(
			density,
			(rows, columns),
			1.0,
		)

		np.add.at(
			height_sum,
			(rows, columns),
			valid_height.astype(np.float32),
		)

		np.minimum.at(
			minimum_height,
			(rows, columns),
			valid_height.astype(np.float32),
		)

		np.maximum.at(
			maximum_height,
			(rows, columns),
			valid_height.astype(np.float32),
		)

	height_valid = (
		density >= minimum_points_per_cell
	)

	occupancy = height_valid.astype(np.float32)

	average_height = np.zeros_like(density)

	np.divide(
		height_sum,
		density,
		out=average_height,
		where=density > 0,
	)

	minimum_height[~height_valid] = 0.0
	maximum_height[~height_valid] = 0.0
	average_height[~height_valid] = 0.0

	vertical_bin_count = len(height_edges) - 1

	vertical_counts = np.zeros(
		(
			vertical_bin_count,
			rows_count,
			columns_count,
		),
		dtype=np.float32,
	)

	if len(valid_height) > 0:
		height_bin_indices = np.searchsorted(
			height_edges,
			valid_height,
			side="right",
		) - 1

		height_bin_indices = np.clip(
			height_bin_indices,
			0,
			vertical_bin_count - 1,
		)

		for bin_index in range(vertical_bin_count):
			selected = (
				height_bin_indices == bin_index
			)

			if not np.any(selected):
				continue

			np.add.at(
				vertical_counts[bin_index],
				(
					rows[selected],
					columns[selected],
				),
				1.0,
			)

	vertical_occupancy = (
		vertical_counts >= vertical_minimum_points
	).astype(np.float32)

	free_space, observed = create_visibility_maps(
		camera_info,
		occupancy,
		right_minimum=right_minimum,
		right_maximum=right_maximum,
		forward_minimum=forward_minimum,
		forward_maximum=forward_maximum,
		resolution=resolution,
		maximum_range=maximum_range,
		free_space_ray_stride=free_space_ray_stride,
	)

	height_normalization = float(height_edges[-1])

	minimum_height_normalized = np.clip(
		minimum_height / height_normalization,
		0.0,
		1.0,
	).astype(np.float32)

	maximum_height_normalized = np.clip(
		maximum_height / height_normalization,
		0.0,
		1.0,
	).astype(np.float32)

	average_height_normalized = np.clip(
		average_height / height_normalization,
		0.0,
		1.0,
	).astype(np.float32)

	normalized_density = np.clip(
		np.log1p(density)
		/ np.log1p(density_reference),
		0.0,
		1.0,
	).astype(np.float32)

	vertical_channel_names = tuple(
		(
			f"occupied_{height_edges[index]:.2f}_"
			f"{height_edges[index + 1]:.2f}m"
		)
		for index in range(vertical_bin_count)
	)

	channel_names = (
		"occupancy",
		"free_space",
		"observed",
		*vertical_channel_names,
		"minimum_height_normalized",
		"maximum_height_normalized",
		"average_height_normalized",
		"point_density_normalized",
	)

	target = np.stack(
		(
			occupancy,
			free_space,
			observed,
			*vertical_occupancy,
			minimum_height_normalized,
			maximum_height_normalized,
			average_height_normalized,
			normalized_density,
		),
		axis=0,
	).astype(np.float32)

	return AgentBev(
		occupancy=occupancy,
		free_space=free_space,
		observed=observed,
		vertical_occupancy=vertical_occupancy,
		minimum_height=minimum_height,
		maximum_height=maximum_height,
		average_height=average_height,
		height_valid=height_valid.astype(np.float32),
		density=density,
		normalized_density=normalized_density,
		target=target,
		channel_names=channel_names,
		height_edges=height_edges.copy(),
		right_minimum=right_minimum,
		right_maximum=right_maximum,
		forward_minimum=forward_minimum,
		forward_maximum=forward_maximum,
		resolution=resolution,
		floor_y_agent=floor_y_agent,
	)


def save_agent_bev(
	bev: AgentBev,
	output_path: Path,
) -> Path:
	"""Save the full training target and its metadata."""

	output_path = output_path.expanduser().resolve()
	output_path.parent.mkdir(
		parents=True,
		exist_ok=True,
	)

	np.savez_compressed(
		output_path,
		target=bev.target,
		channel_names=np.asarray(
			bev.channel_names,
			dtype=str,
		),
		occupancy=bev.occupancy,
		free_space=bev.free_space,
		observed=bev.observed,
		vertical_occupancy=bev.vertical_occupancy,
		minimum_height=bev.minimum_height,
		maximum_height=bev.maximum_height,
		average_height=bev.average_height,
		height_valid=bev.height_valid,
		density=bev.density,
		normalized_density=bev.normalized_density,
		height_edges=bev.height_edges,
		right_minimum=bev.right_minimum,
		right_maximum=bev.right_maximum,
		forward_minimum=bev.forward_minimum,
		forward_maximum=bev.forward_maximum,
		resolution=bev.resolution,
		floor_y_agent=bev.floor_y_agent,
	)

	return output_path


def show_agent_bev(
	bev: AgentBev,
	*,
	title: str = "Agent-relative multi-channel BEV target",
) -> None:
	extent = (
		bev.right_minimum,
		bev.right_maximum,
		bev.forward_minimum,
		bev.forward_maximum,
	)

	plots: list[
		tuple[str, np.ndarray, str, float | None, float | None]
	] = [
		(
			"Obstacle occupancy",
			bev.occupancy,
			"gray",
			0.0,
			1.0,
		),
		(
			"Free-space evidence",
			bev.free_space,
			"Blues",
			0.0,
			1.0,
		),
		(
			"Observed area",
			bev.observed,
			"gray",
			0.0,
			1.0,
		),
		(
			"Maximum obstacle height [m]",
			bev.maximum_height,
			"viridis",
			0.0,
			float(bev.height_edges[-1]),
		),
		(
			"Average obstacle height [m]",
			bev.average_height,
			"viridis",
			0.0,
			float(bev.height_edges[-1]),
		),
		(
			"log(1 + points per cell)",
			np.log1p(bev.density),
			"turbo",
			0.0,
			None,
		),
	]

	for bin_index in range(
		bev.vertical_occupancy.shape[0]
	):
		lower = bev.height_edges[bin_index]
		upper = bev.height_edges[bin_index + 1]

		plots.append(
			(
				f"Occupied {lower:.2f}–{upper:.2f} m",
				bev.vertical_occupancy[bin_index],
				"gray",
				0.0,
				1.0,
			)
		)

	column_count = 4
	row_count = int(
		np.ceil(len(plots) / column_count)
	)

	figure, axes = plt.subplots(
		row_count,
		column_count,
		figsize=(18, 4.5 * row_count),
		constrained_layout=True,
	)

	axes = np.asarray(axes).reshape(-1)

	for axis, (
		plot_title,
		image,
		color_map,
		minimum,
		maximum,
	) in zip(axes, plots):
		display = axis.imshow(
			image,
			origin="lower",
			extent=extent,
			cmap=color_map,
			vmin=minimum,
			vmax=maximum,
			interpolation="nearest",
		)

		axis.scatter(
			0.0,
			0.0,
			marker="^",
			s=100,
			color="cyan",
			edgecolor="black",
			label="Agent",
		)

		axis.set_title(plot_title)
		axis.set_xlabel("Agent X — right [m]")
		axis.set_ylabel("Agent forward (-Z) [m]")
		axis.set_aspect("equal")
		axis.legend(loc="upper right")

		if (
			"height" in plot_title.lower()
			or "points" in plot_title.lower()
		):
			figure.colorbar(
				display,
				ax=axis,
				shrink=0.78,
			)

	for axis in axes[len(plots):]:
		axis.axis("off")

	figure.suptitle(
		f"{title}\n"
		f"target shape: {bev.target.shape}",
		fontsize=16,
	)

	plt.show()


def main() -> None:
	sequence_directory = Path(
		"outputs/habitat_dataset/"
		"scene_102344280/sequence_0000"
	)

	frame_index = 400

	(
		camera_info,
		points_world,
		colors,
		world_from_agent,
	) = depth_environment_create(
		sequence_directory,
		frame_index,
		minimum_depth=0.20,
		maximum_depth=5.0,
		pixel_stride=2,
	)

	bev = create_agent_bev(
		points_world,
		world_from_agent,
		camera_info,
		right_minimum=-5.0,
		right_maximum=5.0,
		forward_minimum=-5.0,
		forward_maximum=5.0,
		resolution=0.10,
		maximum_range=5.0,

		# Your recorded agent origin is approximately 20 cm
		# above the floor. Therefore, the floor in agent
		# coordinates is approximately Y=-0.20 m.
		floor_y_agent=-0.20,

		# Ignore depth noise within 5 cm of the floor.
		floor_tolerance=0.05,

		height_edges=HEIGHT_EDGES,
		minimum_points_per_cell=2,
		vertical_minimum_points=1,
		free_space_ray_stride=12,
	)

	output_path = (
		sequence_directory
		/ "bev_targets"
		/ f"frame_{frame_index:06d}.npz"
	)

	saved_path = save_agent_bev(
		bev,
		output_path,
	)

	print("\nBEV target")
	print("  shape:", bev.target.shape)
	# print("  saved:", saved_path)

	for index, name in enumerate(bev.channel_names):
		print(f"  channel {index:02d}: {name}")

	show_agent_bev(bev)

	show_six_depth_maps(
		camera_info,
		maximum_display_depth=5.0,
	)


if __name__ == "__main__":
	main()