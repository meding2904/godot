extends Label
## Shows what the underwater demo thinks the camera state is, so a wrong
## grade is visible at a glance: SURFACE or UNDERWATER, the camera height, the
## water height under it, and the smoothed submersion.

@export var effect_path: NodePath
@export var water_path: NodePath

var _effect: WaterUnderwaterEffect
var _water: WaterBody3D

func _ready() -> void:
	_effect = get_node_or_null(effect_path)
	_water = get_node_or_null(water_path)

func _process(_delta: float) -> void:
	var camera := get_viewport().get_camera_3d()
	if _effect == null or _water == null or camera == null:
		return
	var position := camera.global_position
	var water_height := _water.sample_height(position)
	var state := "UNDERWATER" if _effect.is_submerged() else "SURFACE"
	text = "%s   camera y=%.2f   water y=%.2f   submersion=%.2f" % [
		state, position.y, water_height, _effect.get_submersion(),
	]
