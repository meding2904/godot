extends SceneTree
## Numeric probe for the underwater demo: prints camera height, water height,
## the effect's smoothed submersion and its fog target, above and below the
## surface. Run headless (water is flat there, still enough to test the logic):
##   ../../bin/godot.linuxbsd.editor.x86_64 --headless --path . --script res://scripts/underwater_probe.gd

var _effect: Node
var _camera: Camera3D
var _water: Node
var _world_environment: WorldEnvironment
var _frame := 0

func _initialize() -> void:
	var scene: PackedScene = load("res://scenes/underwater.tscn")
	var root_node := scene.instantiate()
	root.add_child(root_node)
	_effect = root_node.get_node_or_null("Underwater")
	_camera = root_node.get_node_or_null("Camera3D")
	_water = root_node.get_node_or_null("WaterBody3D")
	_world_environment = root_node.get_node_or_null("WorldEnvironment")

func _process(_delta: float) -> bool:
	_frame += 1
	if _frame == 5:
		_report("start (camera below)")
		_camera.global_position.y = 4.0
	elif _frame == 40:
		_report("raised above the surface")
		quit(0)
	return false

func _report(where: String) -> void:
	var h: float = _water.sample_height(_camera.global_position)
	var environment: Environment = _world_environment.environment
	print("%s: camera y=%.2f water y=%.2f wet=%s submersion=%.2f submerged=%s fog=%s density=%.3f" % [
		where, _camera.global_position.y, h,
		_water.is_wet(_camera.global_position),
		_effect.get_submersion(), _effect.is_submerged(),
		environment.fog_enabled, environment.fog_density,
	])
