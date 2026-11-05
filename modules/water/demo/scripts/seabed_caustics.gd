extends MeshInstance3D
## Feeds the water body's caustics map and projection into this mesh's
## ShaderMaterial (shaders/seabed_caustics.gdshader).

@export var water_path: NodePath

var _water: WaterBody3D
var _material: ShaderMaterial

func _ready() -> void:
	_water = get_node_or_null(water_path)
	_material = material_override as ShaderMaterial

func _process(_delta: float) -> void:
	if _water == null or _material == null:
		return
	_material.set_shader_parameter("caustics_tex", _water.get_caustics_texture())
	_material.set_shader_parameter("caustics_origin", _water.get_caustics_origin())
	_material.set_shader_parameter("caustics_right", _water.get_caustics_light_right())
	_material.set_shader_parameter("caustics_up", _water.get_caustics_light_up())
	_material.set_shader_parameter("caustics_extent", _water.get_caustics_half_extent())
