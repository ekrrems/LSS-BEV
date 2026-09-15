from __future__ import annotations

import argparse
from pathlib import Path

from habitat.create_bev_target import (
    HEIGHT_EDGES,
    create_agent_bev,
    depth_environment_create,
    save_agent_bev,
)
from habitat.read_dataset import load_manifest


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/habitat_dataset/scene_102344280/sequence_0001"),
    )
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--maximum-range",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--max-frames-per-sequence",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    dataset_root = arguments.dataset_root.expanduser().resolve()
    sequence_directories = sorted(
        path.parent
        for path in dataset_root.rglob("manifest.jsonl")
    )

    if not sequence_directories:
        raise FileNotFoundError(
            f"No manifest.json files found inside {dataset_root}"
        )

    total_created = 0

    for sequence_directory in sequence_directories:
        records = load_manifest(sequence_directory)
        target_directory = sequence_directory / "bev_targets"
        target_directory.mkdir(parents=True, exist_ok=True)

        frame_indices = list(range(len(records)))

        if arguments.max_frames_per_sequence > 0:
            frame_indices = frame_indices[
                :arguments.max_frames_per_sequence
            ]

        print(
            f"\nSequence: {sequence_directory} "
            f"({len(frame_indices)} frames)"
        )

        for frame_index in frame_indices:
            output_path = (
                target_directory
                / f"frame_{frame_index:06d}.npz"
            )

            if output_path.exists() and not arguments.overwrite:
                continue

            (
                camera_info,
                points_world,
                _,
                world_from_agent,
            ) = depth_environment_create(
                sequence_directory,
                frame_index,
                minimum_depth=0.2,
                maximum_depth=arguments.maximum_range,
                pixel_stride=arguments.pixel_stride,
            )

            bev = create_agent_bev(
                points_world=points_world,
                world_from_agent=world_from_agent,
                camera_info=camera_info,
                right_minimum=-5.0,
                right_maximum=5.0,
                forward_minimum=-5.0,
                forward_maximum=5.0,
                resolution=0.1,
                maximum_range=arguments.maximum_range,
                floor_y_agent=-0.2,
                floor_tolerance=0.05,
                height_edges=HEIGHT_EDGES,
            )

            save_agent_bev(bev, output_path)

            total_created += 1

            if frame_index % 25 == 0:
                print(
                    f"  saved frame {frame_index:06d}: "
                    f"{output_path.name}"
                )

    print(f"\nCreated {total_created} BEV targets.")


if __name__ == "__main__":
    main()