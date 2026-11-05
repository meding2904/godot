extends SceneTree
## Headless smoke probe: engine boots, module classes registered, then quits.
##   ../../bin/godot.linuxbsd.editor.x86_64 --headless --path . --script res://scripts/probe.gd

func _initialize() -> void:
	print("probe: boot")
	for cls in ["WaterBody3D", "WaterBuoyancy3D", "WaterBoat3D", "WaterWake3D", "WaterSpray3D"]:
		print("probe: %s = %s" % [cls, ClassDB.class_exists(cls)])
	quit(0)
