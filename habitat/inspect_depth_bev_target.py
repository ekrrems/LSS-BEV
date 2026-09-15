from __future__ import annotations

import argparse
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


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("sequence", type=Path)
	parser.add_argument("--frame", type=int, default=0)
	arguments = parser.parse_args()
	sequence = arguments.sequence.expanduser().resolve()

	with (sequence / "manifest.jsonl").open("r", encoding="utf-8") as file:
		records = [json.loads(line) for line in file if line.strip()]
	record = records[arguments.frame]
	frame_name = f"{int(record['frame_index']):06d}.npz"
	target_path = sequence / "targets" / "bev_occupancy" / frame_name
	target = np.load(target_path)
	occupancy = target["occupancy"]
	counts = target["counts"]

	settings_path = target_path.parent / "settings.json"
	with settings_path.open("r", encoding="utf-8") as file:
		settings = json.load(file)
	extent = [
		settings["y_minimum"],
		settings["y_maximum"],
		settings["x_minimum"],
		settings["x_maximum"],
	]

	images = []
	for camera_name in CAMERAS:
		path = sequence / record["cameras"][camera_name]["rgb"]
		image = cv2.imread(str(path))
		images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	mosaic = np.vstack((np.hstack(images[:3]), np.hstack(images[3:])))

	figure, axes = plt.subplots(1, 3, figsize=(19, 6))
	axes[0].imshow(mosaic)
	axes[0].set_title("Six synchronized RGB cameras")
	axes[0].axis("off")

	axes[1].imshow(
		occupancy,
		origin="lower",
		extent=extent,
		cmap="gray",
		vmin=0,
		vmax=1,
	)
	axes[1].scatter([0], [0], marker="^", color="cyan", s=80)
	axes[1].set_title("Depth-derived obstacle target")
	axes[1].set_xlabel("Ego Y — left [m]")
	axes[1].set_ylabel("Ego X — forward [m]")
	axes[1].set_aspect("equal")

	counts_plot = axes[2].imshow(
		np.log1p(counts),
		origin="lower",
		extent=extent,
		cmap="turbo",
	)
	axes[2].scatter([0], [0], marker="^", color="cyan", s=80)
	axes[2].set_title("log(1 + points per cell)")
	axes[2].set_xlabel("Ego Y — left [m]")
	axes[2].set_ylabel("Ego X — forward [m]")
	axes[2].set_aspect("equal")
	figure.colorbar(counts_plot, ax=axes[2])

	figure.suptitle(f"{sequence.name}, frame {record['frame_index']}")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
