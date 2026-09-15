from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

# Keep the native-library import order used by the working macOS environment.
import cv2  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import habitat_sim
import quaternion
import torch
from matplotlib.colors import ListedColormap

from habitat_lss.habitat_lss_model import HabitatLiftSplatShoot
from habitat_lss.habitat_train import CAMERAS, TARGET_CHANNELS, select_device


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move a Habitat agent and display an RGB-predicted local BEV."
    )
    parser.add_argument(
        "--scene",
        required=True,
        help=(
            "Habitat scene handle (for example 102344280) or a direct scene "
            "file path. Scene handles require --scene-dataset-config."
        ),
    )
    parser.add_argument(
        "--reference-sequence",
        type=Path,
        default=Path("outputs/habitat_dataset/scene_102344280/sequence_0000"),
        help="Recorded sequence whose camera calibration and poses match training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/habitat_lss_depth_augmented/habitat_lss_depth_best_iou.pt"
        ),
    )
    parser.add_argument("--scene-dataset-config", type=Path, default=None)
    parser.add_argument("--occupancy-threshold", type=float, default=0.40)
    parser.add_argument("--free-threshold", type=float, default=0.50)
    parser.add_argument("--observed-threshold", type=float, default=0.50)
    parser.add_argument("--move-step", type=float, default=0.15)
    parser.add_argument("--turn-degrees", type=float, default=10.0)
    parser.add_argument("--agent-height", type=float, default=1.20)
    parser.add_argument("--agent-radius", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_reference_rig(
    sequence_directory: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    sequence_directory = sequence_directory.expanduser().resolve()
    with (sequence_directory / "calibration.json").open(
        "r", encoding="utf-8"
    ) as file:
        calibration = json.load(file)
    with (sequence_directory / "manifest.jsonl").open(
        "r", encoding="utf-8"
    ) as file:
        first_record = json.loads(next(line for line in file if line.strip()))

    intrinsic = np.asarray(calibration["K"], dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected K [3,3], got {intrinsic.shape}")
    transforms: dict[str, np.ndarray] = {}
    for camera_name in CAMERAS:
        transform = np.asarray(
            first_record["cameras"][camera_name]["agent_from_camera"],
            dtype=np.float32,
        )
        if transform.shape != (4, 4):
            raise ValueError(f"Invalid transform for {camera_name}: {transform.shape}")
        transforms[camera_name] = transform
    return intrinsic, transforms


def yaw_from_agent_from_camera(transform: np.ndarray) -> float:
    """Extract a horizontal Habitat sensor yaw from a camera transform."""
    rotation = transform[:3, :3]
    yaw = math.atan2(float(rotation[0, 2]), float(rotation[0, 0]))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    reconstructed = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float32,
    )
    error = float(np.max(np.abs(rotation - reconstructed)))
    if error > 0.03:
        raise ValueError(
            "The recorded camera rig contains pitch or roll that this live script "
            f"cannot reproduce safely (rotation error {error:.4f})."
        )
    return yaw


def make_simulator(
    scene: str,
    intrinsic: np.ndarray,
    transforms: dict[str, np.ndarray],
    arguments: argparse.Namespace,
) -> habitat_sim.Simulator:
    scene_value = str(scene)
    scene_candidate = Path(scene_value).expanduser()
    looks_like_path = (
        "/" in scene_value
        or "\\" in scene_value
        or scene_candidate.suffix.lower() in {".glb", ".json"}
    )
    if looks_like_path:
        scene_candidate = scene_candidate.resolve()
        if not scene_candidate.exists():
            raise FileNotFoundError(
                f"Habitat scene does not exist: {scene_candidate}"
            )
        scene_identifier = str(scene_candidate)
    else:
        # Dataset configurations such as HSSD resolve IDs like "102344280".
        scene_identifier = scene_value
        if arguments.scene_dataset_config is None:
            raise ValueError(
                "A scene handle was supplied. Also pass "
                "--scene-dataset-config so Habitat can resolve it."
            )
    if (
        arguments.scene_dataset_config is not None
        and not arguments.scene_dataset_config.expanduser().resolve().exists()
    ):
        raise FileNotFoundError(
            "Scene dataset configuration does not exist: "
            f"{arguments.scene_dataset_config.expanduser().resolve()}"
        )
    image_height = 256
    image_width = 352
    fx = float(intrinsic[0, 0])
    horizontal_fov = math.degrees(2.0 * math.atan(image_width / (2.0 * fx)))

    sensor_specs = []
    for camera_name in CAMERAS:
        transform = transforms[camera_name]
        specification = habitat_sim.CameraSensorSpec()
        specification.uuid = camera_name
        specification.sensor_type = habitat_sim.SensorType.COLOR
        specification.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        specification.resolution = [image_height, image_width]
        specification.position = transform[:3, 3].astype(float).tolist()
        specification.orientation = [
            0.0,
            yaw_from_agent_from_camera(transform),
            0.0,
        ]
        specification.hfov = horizontal_fov
        sensor_specs.append(specification)

    action = habitat_sim.agent.ActionSpec
    actuation = habitat_sim.agent.ActuationSpec
    action_space = {
        "move_forward": action(
            "move_forward", actuation(amount=arguments.move_step)
        ),
        "move_backward": action(
            "move_backward", actuation(amount=arguments.move_step)
        ),
        "turn_left": action(
            "turn_left", actuation(amount=arguments.turn_degrees)
        ),
        "turn_right": action(
            "turn_right", actuation(amount=arguments.turn_degrees)
        ),
    }
    agent_configuration = habitat_sim.agent.AgentConfiguration(
        height=arguments.agent_height,
        radius=arguments.agent_radius,
        sensor_specifications=sensor_specs,
        action_space=action_space,
        body_type="cylinder",
    )
    simulator_configuration = habitat_sim.SimulatorConfiguration()
    simulator_configuration.scene_id = scene_identifier
    simulator_configuration.random_seed = arguments.seed
    if arguments.scene_dataset_config is not None:
        simulator_configuration.scene_dataset_config_file = str(
            arguments.scene_dataset_config.expanduser().resolve()
        )
    return habitat_sim.Simulator(
        habitat_sim.Configuration(simulator_configuration, [agent_configuration])
    )


def ensure_navigation_mesh(
    simulator: habitat_sim.Simulator,
    arguments: argparse.Namespace,
) -> None:
    """Load-free HSSD installs need a navmesh generated at runtime."""
    pathfinder = simulator.pathfinder
    if pathfinder.is_loaded:
        print(
            "Navigation mesh loaded: "
            f"{pathfinder.navigable_area:.2f} square metres"
        )
        return

    print("No navigation mesh was packaged for this scene; generating one...")
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_height = arguments.agent_height
    settings.agent_radius = arguments.agent_radius
    settings.agent_max_climb = 0.15
    settings.agent_max_slope = 40.0

    success = simulator.recompute_navmesh(pathfinder, settings)
    if not success or not pathfinder.is_loaded:
        raise RuntimeError(
            "Habitat could not generate a navigation mesh for this scene."
        )
    print(
        "Navigation mesh generated: "
        f"{pathfinder.navigable_area:.2f} square metres"
    )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[HabitatLiftSplatShoot, dict]:
    checkpoint = torch.load(checkpoint_path.expanduser().resolve(), map_location=device)
    model = HabitatLiftSplatShoot(output_channels=TARGET_CHANNELS).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


class GlobalBevMap:
    """Fuse agent-relative BEV predictions in Habitat world coordinates."""

    def __init__(
        self,
        simulator: habitat_sim.Simulator,
        model: HabitatLiftSplatShoot,
    ) -> None:
        lower, upper = simulator.pathfinder.get_bounds()
        margin = 1.0
        self.resolution = float(model.resolution)
        self.minimum_x = float(lower[0]) - margin
        self.maximum_x = float(upper[0]) + margin
        self.minimum_z = float(lower[2]) - margin
        self.maximum_z = float(upper[2]) + margin
        self.width = int(
            math.ceil((self.maximum_x - self.minimum_x) / self.resolution)
        )
        self.height = int(
            math.ceil((self.maximum_z - self.minimum_z) / self.resolution)
        )
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        self.observations = np.zeros((self.height, self.width), dtype=np.uint16)

        forward = model.forward_minimum + self.resolution * (
            np.arange(model.forward_cells, dtype=np.float32) + 0.5
        )
        right = model.right_minimum + self.resolution * (
            np.arange(model.right_cells, dtype=np.float32) + 0.5
        )
        forward_grid, right_grid = np.meshgrid(forward, right, indexing="ij")
        self.local_points = np.stack(
            (
                right_grid,
                np.zeros_like(right_grid),
                -forward_grid,
            ),
            axis=-1,
        )

    def clear(self) -> None:
        self.log_odds.fill(0.0)
        self.observations.fill(0)

    def update(
        self,
        state: habitat_sim.AgentState,
        occupied: np.ndarray,
        free: np.ndarray,
        observed: np.ndarray,
    ) -> None:
        rotation = quaternion.as_rotation_matrix(state.rotation).astype(np.float32)
        position = np.asarray(state.position, dtype=np.float32)
        world_points = self.local_points @ rotation.T + position
        columns = np.floor(
            (world_points[..., 0] - self.minimum_x) / self.resolution
        ).astype(np.int64)
        rows = np.floor(
            (world_points[..., 2] - self.minimum_z) / self.resolution
        ).astype(np.int64)
        valid = (
            observed
            & (rows >= 0)
            & (rows < self.height)
            & (columns >= 0)
            & (columns < self.width)
        )

        evidence = np.zeros_like(rows, dtype=np.float32)
        # Obstacles are added more strongly; several free observations are
        # required to erase one obstacle observation.
        evidence[free] = -0.45
        evidence[occupied] = 0.90
        np.add.at(self.log_odds, (rows[valid], columns[valid]), evidence[valid])
        np.add.at(self.observations, (rows[valid], columns[valid]), 1)
        np.clip(self.log_odds, -6.0, 6.0, out=self.log_odds)

    def navigation_grid(self) -> np.ndarray:
        grid = np.zeros_like(self.observations, dtype=np.uint8)
        known = self.observations > 0
        grid[known & (self.log_odds <= -0.80)] = 1
        grid[known & (self.log_odds >= 0.80)] = 2
        return grid

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (
            self.minimum_x,
            self.maximum_x,
            self.minimum_z,
            self.maximum_z,
        )


class LiveBevApplication:
    def __init__(
        self,
        simulator: habitat_sim.Simulator,
        model: HabitatLiftSplatShoot,
        intrinsic: np.ndarray,
        transforms: dict[str, np.ndarray],
        device: torch.device,
        arguments: argparse.Namespace,
    ) -> None:
        self.simulator = simulator
        self.agent = simulator.initialize_agent(0)
        self.model = model
        self.device = device
        self.arguments = arguments
        self.step_number = 0
        self.figure, self.axes = plt.subplots(2, 4, figsize=(16, 8))
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.state_cmap = ListedColormap(["#303030", "#59b85c", "#e53935"])
        self.global_map = GlobalBevMap(simulator, model)

        self.intrinsics = torch.from_numpy(intrinsic).unsqueeze(0).repeat(
            len(CAMERAS), 1, 1
        ).unsqueeze(0).to(device=device, dtype=torch.float32)
        self.rotations = torch.stack(
            [torch.from_numpy(transforms[name][:3, :3]) for name in CAMERAS]
        ).unsqueeze(0).to(device=device, dtype=torch.float32)
        self.translations = torch.stack(
            [torch.from_numpy(transforms[name][:3, 3]) for name in CAMERAS]
        ).unsqueeze(0).to(device=device, dtype=torch.float32)

        if not simulator.pathfinder.is_loaded:
            raise RuntimeError("Navigation mesh initialization failed")
        start = simulator.pathfinder.get_random_navigable_point()
        state = self.agent.get_state()
        state.position = start
        self.agent.set_state(state, reset_sensors=True)

    def observations_to_tensor(
        self,
        observations: dict[str, np.ndarray],
    ) -> tuple[torch.Tensor, list[np.ndarray]]:
        images = []
        displayed = []
        for camera_name in CAMERAS:
            image = np.asarray(observations[camera_name])
            if image.ndim != 3 or image.shape[2] < 3:
                raise ValueError(f"Unexpected {camera_name} image shape: {image.shape}")
            rgb = np.ascontiguousarray(image[:, :, :3])
            displayed.append(rgb)
            images.append(
                torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
            )
        tensor = torch.stack(images).unsqueeze(0).to(self.device)
        return tensor, displayed

    def render(self) -> None:
        observations = self.simulator.get_sensor_observations()
        images, displayed = self.observations_to_tensor(observations)
        started = time.perf_counter()
        with torch.inference_mode():
            model_output = self.model.forward_with_depth(
                images,
                self.intrinsics,
                self.rotations,
                self.translations,
            )
            logits = model_output.logits
            probabilities = torch.sigmoid(logits[0]).cpu().numpy()
        inference_ms = (time.perf_counter() - started) * 1_000.0

        occupancy = probabilities[0]
        free_probability = probabilities[1]
        observed_probability = probabilities[2]
        observed = observed_probability >= self.arguments.observed_threshold
        occupied = (
            occupancy >= self.arguments.occupancy_threshold
        ) & observed
        free = (
            free_probability >= self.arguments.free_threshold
        ) & observed & ~occupied
        navigation_grid = np.zeros((100, 100), dtype=np.uint8)
        navigation_grid[free] = 1
        navigation_grid[occupied] = 2
        agent_state = self.agent.get_state()
        self.global_map.update(agent_state, occupied, free, observed)
        global_navigation_grid = self.global_map.navigation_grid()

        for axis in self.axes.flat:
            axis.clear()
            axis.axis("off")
        for camera_index, camera_name in enumerate(CAMERAS):
            axis = self.axes.flat[camera_index]
            axis.imshow(displayed[camera_index])
            axis.set_title(camera_name.replace("camera_", ""))

        self.axes.flat[6].imshow(
            occupancy,
            origin="lower",
            extent=(-5.0, 5.0, -5.0, 5.0),
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        self.axes.flat[6].scatter(0.0, 0.0, marker="^", color="cyan", s=70)
        self.axes.flat[6].set_title("Predicted occupancy probability")
        self.axes.flat[7].imshow(
            global_navigation_grid,
            origin="lower",
            extent=self.global_map.extent,
            cmap=self.state_cmap,
            vmin=0,
            vmax=2,
            interpolation="nearest",
        )
        self.axes.flat[7].scatter(
            float(agent_state.position[0]),
            float(agent_state.position[2]),
            marker="^",
            color="cyan",
            s=70,
        )
        self.axes.flat[7].set_title("Accumulated global BEV map (Habitat pose)")
        self.axes.flat[6].axis("on")
        self.axes.flat[6].set_xlabel("Agent right [m]")
        self.axes.flat[6].set_ylabel("Agent forward [m]")
        self.axes.flat[7].axis("on")
        self.axes.flat[7].set_xlabel("Habitat world X [m]")
        self.axes.flat[7].set_ylabel("Habitat world Z [m]")
        collision = bool(getattr(self.simulator, "previous_step_collided", False))
        self.figure.suptitle(
            f"Live RGB-only Habitat BEV — step {self.step_number} — "
            f"inference {inference_ms:.1f} ms — collision {collision}\n"
            "W/S move, A/D turn, R random start, C clear map, Q quit"
        )
        self.figure.tight_layout()
        self.figure.canvas.draw_idle()

    def random_start(self) -> None:
        state = self.agent.get_state()
        state.position = self.simulator.pathfinder.get_random_navigable_point()
        self.agent.set_state(state, reset_sensors=True)

    def on_key(self, event: object) -> None:
        key = getattr(event, "key", None)
        actions = {
            "w": "move_forward",
            "s": "move_backward",
            "a": "turn_left",
            "d": "turn_right",
        }
        if key in actions:
            self.simulator.step(actions[key])
            self.step_number += 1
            self.render()
        elif key == "r":
            self.random_start()
            self.render()
        elif key == "c":
            self.global_map.clear()
            self.render()
        elif key in ("q", "escape"):
            plt.close(self.figure)

    def run(self) -> None:
        self.render()
        plt.show()


def main() -> None:
    arguments = parse_arguments()
    for name in ("occupancy_threshold", "free_threshold", "observed_threshold"):
        value = getattr(arguments, name)
        if not 0.0 < value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    intrinsic, transforms = load_reference_rig(arguments.reference_sequence)
    device = select_device()
    model, checkpoint = load_model(arguments.checkpoint, device)
    simulator = make_simulator(
        arguments.scene,
        intrinsic,
        transforms,
        arguments,
    )
    ensure_navigation_mesh(simulator, arguments)
    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Scene: {arguments.scene}")
    print("Controls: W/S move, A/D turn, R random start, C clear map, Q quit")
    try:
        LiveBevApplication(
            simulator,
            model,
            intrinsic,
            transforms,
            device,
            arguments,
        ).run()
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
