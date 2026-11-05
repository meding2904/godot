extends Camera3D
## Chase camera for the boat scene.

@export var target_path: NodePath
@export var distance := 12.0
@export var height := 4.0

var _target: Node3D

func _ready() -> void:
	_target = get_node_or_null(target_path)

func _process(_delta: float) -> void:
	if _target == null:
		return
	var xf := _target.global_transform
	# The boat thrusts toward +Z, so +basis.z is behind its stern.
	global_position = xf.origin + xf.basis.z * distance + Vector3(0.0, height, 0.0)
	look_at(xf.origin + Vector3(0.0, 0.6, 0.0), Vector3.UP)
