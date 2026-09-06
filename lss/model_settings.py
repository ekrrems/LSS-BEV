"""Shared settings for the Lift-Splat-Shoot model."""


# Processed image dimensions
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 704


# Discrete camera-depth range in metres
DEPTH_MINIMUM = 4.0
DEPTH_MAXIMUM = 45.0
DEPTH_STEP = 1.0

DEPTH_BINS = int(
	(DEPTH_MAXIMUM - DEPTH_MINIMUM)
	/ DEPTH_STEP
)


# CNN dimensions
CONTEXT_CHANNELS = 64
BASE_CHANNELS = 32


# Validate the settings immediately when imported.
if IMAGE_HEIGHT % 16 != 0:
	raise ValueError(
		"IMAGE_HEIGHT must be divisible by 16"
	)

if IMAGE_WIDTH % 16 != 0:
	raise ValueError(
		"IMAGE_WIDTH must be divisible by 16"
	)

if DEPTH_BINS <= 1:
	raise ValueError(
		"DEPTH_BINS must be greater than one"
	)

# BEV Configuration
BEV_X_MINIMUM = -50.0
BEV_X_MAXIMUM = 50.0

BEV_Y_MINIMUM = -50.0
BEV_Y_MAXIMUM = 50.0

BEV_Z_MINIMUM = -10.0
BEV_Z_MAXIMUM = 10.0

BEV_RESOLUTION = 0.5