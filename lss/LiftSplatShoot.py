# class LiftSplatShoot(nn.Module):
# 	def __init__(self) -> None:
# 		super().__init__()

# 		self.image_encoder = LssImageEncoder(...)
# 		self.lift_geometry = LiftGeometry(...)
# 		self.camera_to_ego = CameraToEgo()

# 		# We still need to implement these:
# 		self.bev_pool = BevPool(...)
# 		self.bev_decoder = BevDecoder(...)

# 	def forward(
# 		self,
# 		images,
# 		intrinsics,
# 		camera_rotations,
# 		camera_translations,
# 	):
# 		encoded = self.image_encoder(images)

# 		lifted = self.lift_geometry(
# 			encoded,
# 			intrinsics,
# 			image_height=IMAGE_HEIGHT,
# 			image_width=IMAGE_WIDTH,
# 		)

# 		points_ego = self.camera_to_ego(
# 			lifted.points_camera,
# 			camera_rotations,
# 			camera_translations,
# 		)

# 		bev_features = self.bev_pool(
# 			points_ego,
# 			lifted.features,
# 		)

# 		bev_logits = self.bev_decoder(
# 			bev_features
# 		)

# 		return bev_logits