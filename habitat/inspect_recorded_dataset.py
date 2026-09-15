from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Inspect one recorded Habitat frame.")
	parser.add_argument("sequence", type=Path)
	parser.add_argument("--frame", type=int, default=0)
	parser.add_argument("--camera", default="camera_front")
	return parser.parse_args()


def main() -> None:
	arguments = parse_arguments()
	sequence = arguments.sequence.expanduser().resolve()

	with (sequence / "manifest.jsonl").open("r", encoding="utf-8") as file:
		records = [json.loads(line) for line in file if line.strip()]

	if not 0 <= arguments.frame < len(records):
		raise IndexError(f"Frame must be in [0, {len(records) - 1}]")

	record = records[arguments.frame]
	camera = record["cameras"][arguments.camera]
	rgb = cv2.cvtColor(
		cv2.imread(str(sequence / camera["rgb"])),
		cv2.COLOR_BGR2RGB,
	)
	depth = np.load(sequence / camera["depth"])

	figure, axes = plt.subplots(1, 2, figsize=(14, 5))
	axes[0].imshow(rgb)
	axes[0].set_title(f"{arguments.camera} RGB")
	axes[0].axis("off")

	depth_image = axes[1].imshow(depth, cmap="turbo", vmin=0.0, vmax=10.0)
	axes[1].set_title("Metric depth [m]")
	axes[1].axis("off")
	figure.colorbar(depth_image, ax=axes[1])

	if "semantic" in camera:
		semantic = np.load(sequence / camera["semantic"])
		print("Semantic IDs:", np.unique(semantic))

	world_from_agent = np.asarray(record["world_from_agent"])
	print("Frames:", len(records))
	print("Frame index:", record["frame_index"])
	print("Segment index:", record["segment_index"])
	print("Agent world position:", world_from_agent[:3, 3])
	print("RGB shape:", rgb.shape)
	print("Depth shape:", depth.shape)
	print("Finite depth range:", np.nanmin(depth), np.nanmax(depth))

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
