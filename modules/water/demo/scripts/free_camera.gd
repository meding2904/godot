extends Camera3D
## Free-look camera for the demo scenes: WASD/Shift to fly, right mouse to look,
## Esc to quit. Aimed at `look_target` on load so the scenes need no hand-tuned
## transforms.

@export var look_target := Vector3.ZERO
@export var move_speed := 8.0

var _yaw := 0.0
var _pitch := 0.0

func _ready() -> void:
	look_at(look_target, Vector3.UP)
	_yaw = rotation.y
	_pitch = rotation.x

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
		_yaw -= event.relative.x * 0.003
		_pitch = clampf(_pitch - event.relative.y * 0.003, -1.5, 1.5)
		rotation = Vector3(_pitch, _yaw, 0.0)
	elif event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_ESCAPE:
		get_tree().quit()

func _process(delta: float) -> void:
	var dir := Vector3.ZERO
	if Input.is_key_pressed(KEY_W):
		dir -= transform.basis.z
	if Input.is_key_pressed(KEY_S):
		dir += transform.basis.z
	if Input.is_key_pressed(KEY_A):
		dir -= transform.basis.x
	if Input.is_key_pressed(KEY_D):
		dir += transform.basis.x
	if dir != Vector3.ZERO:
		var speed := move_speed * (4.0 if Input.is_key_pressed(KEY_SHIFT) else 1.0)
		global_position += dir.normalized() * speed * delta
