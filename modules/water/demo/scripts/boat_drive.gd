extends WaterBoat3D
## Keyboard control for the demo boat: arrows drive, Esc quits.

func _physics_process(_delta: float) -> void:
	throttle = Input.get_action_strength("ui_up") - Input.get_action_strength("ui_down")
	steering = Input.get_action_strength("ui_left") - Input.get_action_strength("ui_right")
