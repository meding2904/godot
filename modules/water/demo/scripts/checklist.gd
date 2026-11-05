extends Node3D
## Phase 0 baseline checks for the water module. Run:
##   ../../bin/godot.linuxbsd.editor.x86_64 --path . scenes/checklist.tscn
## Prints PASS/FAIL/SKIP per check and exits non-zero when anything failed.
##
## The CPU-vs-GPU height comparison needs the analytic CPU sampler (Phase 3);
## until then `sample_height()` is itself the GPU readback, so this checks that
## the solver is alive, that waves move, and that the world-anchored FFT does
## not slide when the ripple square is re-centred.

const CLASSES := ["WaterBody3D", "WaterBuoyancy3D", "WaterBoat3D", "WaterWake3D", "WaterSpray3D"]
const SAMPLE_POINTS: Array[Vector3] = [
	Vector3(0.0, 0.0, 0.0),
	Vector3(12.0, 0.0, 8.0),
	Vector3(-9.0, 0.0, 21.0),
	Vector3(30.0, 0.0, -14.0),
]

var _water: WaterBody3D
var _failed := 0
var _skipped := 0

func _ready() -> void:
	_water = get_node_or_null("../WaterBody3D") as WaterBody3D
	print("GPU: %s" % RenderingServer.get_video_adapter_name())
	_check_classes()
	if _water == null:
		_fail("no WaterBody3D sibling node")
	else:
		await _check_waves()
		await _check_recenter()
		await _check_frame_time()
	print("---- %d failed, %d skipped" % [_failed, _skipped])
	get_tree().quit(1 if _failed > 0 else 0)

func _check_classes() -> void:
	var missing: Array[String] = []
	for cls in CLASSES:
		if not ClassDB.class_exists(cls):
			missing.append(cls)
	if missing.is_empty():
		_pass("all water classes registered")
	else:
		_fail("classes missing: %s" % ", ".join(missing))

func _has_compute() -> bool:
	return RenderingServer.get_rendering_device() != null

func _check_waves() -> void:
	if not _has_compute():
		_skip("water is flat: this build has no RenderingDevice compute (headless or Compatibility)")
		return
	var lo := INF
	var hi := -INF
	for i in 90:
		await get_tree().physics_frame
		for point in SAMPLE_POINTS:
			var h := _water.sample_height(point)
			if h == -INF:
				continue
			lo = minf(lo, h)
			hi = maxf(hi, h)
	if lo == INF:
		_skip("no wet sample points")
	elif hi - lo < 0.001:
		_fail("water heights never change over 90 physics frames (range %.4f m)" % (hi - lo))
	else:
		_pass("waves animate: height range %.3f m" % (hi - lo))

func _check_recenter() -> void:
	if not _has_compute():
		_skip("recenter check needs the compute solver")
		return
	var probe := Vector3(40.0, 0.0, 40.0)
	var before := _water.sample_height(probe)
	_water.recenter(probe)
	for i in 10:
		await get_tree().physics_frame
	var after := _water.sample_height(probe)
	if before == -INF or after == -INF:
		_skip("recenter check needs open water")
	elif absf(before - after) <= 0.1:
		_pass("recenter keeps world-anchored waves (delta %.3f m)" % absf(before - after))
	else:
		_fail("waves slid with the square on recenter (delta %.3f m)" % absf(before - after))

func _check_frame_time() -> void:
	if DisplayServer.get_name() == "headless":
		_skip("headless: no frame timing")
		return
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	for i in 30:
		await RenderingServer.frame_post_draw
	var times: Array[float] = []
	var last := Time.get_ticks_usec()
	for i in 180:
		await RenderingServer.frame_post_draw
		var now := Time.get_ticks_usec()
		times.append((now - last) / 1000.0)
		last = now
	times.sort()
	var total := 0.0
	for t in times:
		total += t
	var avg := total / times.size()
	var p95 := times[int(times.size() * 0.95)]
	print("[INFO] frame time: avg %.2f ms, p95 %.2f ms (%.0f fps)" % [avg, p95, 1000.0 / avg])
	if p95 > 16.6:
		print("[WARN] p95 over one 60 Hz frame on this machine")
	_skip("frame-time budget is machine-specific: report, do not gate")

func _pass(what: String) -> void:
	print("[PASS] %s" % what)

func _fail(what: String) -> void:
	_failed += 1
	print("[FAIL] %s" % what)

func _skip(what: String) -> void:
	_skipped += 1
	print("[SKIP] %s" % what)
