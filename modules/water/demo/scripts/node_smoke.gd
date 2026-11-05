extends SceneTree
## Node-level check: the ocean must work by instantiating WaterBody3D alone --
## no scene file, no configured properties, no extra nodes beyond a camera.
##   ../../bin/godot.linuxbsd.editor.x86_64 --headless --path . --script res://scripts/node_smoke.gd

var _water: WaterBody3D
var _frames := 0

func _initialize() -> void:
	# A camera is the one thing the engine always needs; nothing else is added.
	var camera := Camera3D.new()
	camera.position = Vector3(0, 6, 18)
	root.add_child(camera)

	_water = WaterBody3D.new()
	root.add_child(_water)

func _process(_delta: float) -> bool:
	_frames += 1
	if _frames < 10:
		return false

	var mesh_instance: MeshInstance3D = null
	for child in _water.get_children(true):
		if child is MeshInstance3D:
			mesh_instance = child
	var material := _water.water_material
	var has_mesh := mesh_instance != null and mesh_instance.mesh != null
	var has_shader := material != null and material.shader != null
	var internal_effect := _water.get_child_count(true) > _water.get_child_count()

	print("bare WaterBody3D:")
	print("  mesh=%s material=%s" % [has_mesh, has_shader])
	print("  render_extent=%.0f  ocean_tile=%s  grids=%d/%d  domain=%.0f" % [
		_water.render_extent, _water.ocean_domain_size,
		_water.grid_resolution, _water.ocean_grid_resolution, _water.domain_size.x,
	])
	print("  sun=%s" % ("scene light" if _water.sun_path != NodePath() else "auto/default"))
	print("  auto_recenter=%s  underwater_effect=%s  internal_effect=%s" % [
		_water.auto_recenter, _water.underwater_effect, internal_effect,
	])
	print("  compute=%s" % (RenderingServer.get_rendering_device() != null))
	# Every sampler must already be bound: no "missing texture" in the material.
	var unbound: Array[String] = []
	for slot in ["ripple_height_tex", "ocean_height_tex", "ocean_fade_tex", "shore_depth_tex",
			"ocean_disp_tex", "ocean_deriv_tex", "ocean_foam_tex", "shore_foam_tex", "swash_tex", "detail_normal"]:
		if material.get_shader_parameter(slot) == null:
			unbound.append(slot)
	print("  unbound textures=%s" % [unbound])
	var ok := has_mesh and has_shader and internal_effect and _water.auto_recenter and unbound.is_empty()
	print("RESULT: %s" % ("PASS" if ok else "FAIL"))
	quit(0 if ok else 1)
	return false
