# Using the water system

A practical guide to `WaterBody3D` and its neighbours. The class reference
(`doc_classes/`) documents every property; this page is the "what do I add to
my scene" version. Paths below (`demo/`, `doc_classes/`) are relative to the
module directory, `modules/water/`.

## Minimal setup

Two nodes are enough: an environment to light the scene, and the water.

1. `WorldEnvironment` with an `Environment` (a `Sky` gives the water its
   reflection colour; a `ProceduralSkyMaterial` is picked up automatically).
2. `DirectionalLight3D` for the sun. The water finds the first one in the
   scene, or point `sun_path` at a specific one.
3. `WaterBody3D`. It creates its own `MeshInstance3D` child and material — no
   mesh or material to author.
4. A `Camera3D`.

The water itself needs neither a light nor a sky at runtime — the material is
unshaded and falls back to its own sun/sky colours — so a bare `WaterBody3D`
plus a camera already renders an ocean; the environment is for the rest of the
scene. The defaults are an open sea: 128² solver grids,
a 64 m ocean tile, `render_extent` 1000, the interactive ripple square follows
the camera (`auto_recenter`), and diving grades the frame (`underwater_effect`,
provided the scene has a `WorldEnvironment`). Add the node: it runs the waves
right there in the editor, and in Play.

```gdscript
var water := WaterBody3D.new()
water.wind_speed = 10.0
water.wave_amplitude = 1.2
water.render_extent = 1500.0   # draw the FFT ocean out to the horizon
add_child(water)
```

Limits worth knowing up front:

- Keep the node **unrotated and unscaled**: the simulation ignores rotation
  and scale.
- It needs `RenderingDevice` compute (Forward+ or Mobile renderer). In
  Compatibility or headless it stays flat and `sample_height()` returns the
  still-water level — a no-op, not an error. Forward+/Mobile editors preview
  the same waves live, with no setup.
- The material is unshaded and alpha-blended by design: the transparent pass
  has no scene lights, and it must not add a second PBR response on top of the
  scene.

## Open sea

Defaults are a 20 m ripple patch and a 40 m ocean tile. For a sea:

| Property | Value | Why |
| --- | --- | --- |
| `render_extent` | `1000`–`2000` (default `1000`) | Renders the tiled FFT ocean past the simulated square, out to this width. The mesh's dense core follows the camera. |
| `auto_recenter` | `true` (default) | The ripple square and node follow the camera, so the endless sea needs no script. Turn off for water the game moves. |
| `ocean_domain_size` | `60`–`100` | The FFT's periodic tile: the longest waves and the caustics repeat at this size. |
| `ocean_grid_resolution` | `64`–`128` | Wave modes; power of two. |
| `wind_speed` / `wind_direction` | `8`–`14` m/s | JONSWAP spectrum: longer, taller waves. |
| `wave_amplitude` | `1.0`–`1.5` | Physical wave height in meters at `1.0`. |
| `choppiness` | `0.8`–`1.4` | Sharpens crests and feeds the whitecap foam. |

The sea state can change live — `set_sea_state(wind_speed, wind_direction,
wave_amplitude, fetch)` updates the spectrum in place (phases preserved), so a
storm can build frame by frame.

For an endless sea that follows the player, call `recenter(player_position)` as
the player moves away (every ~10 m). The ocean is world-anchored and tiles
forever; only the interactive ripple square moves, and its ripples are cleared.

## Lakes, ponds and pools

Set `surface_mesh` and the water takes its X/Z footprint as the outline. The
mesh is *only* read for that outline: it is not rendered, its material, UVs,
normals and vertex density are ignored, and the node resamples the footprint
into its own even grid. `sample_height()`/`is_wet()` report no water outside
it, waves reflect off the edge, and with `seabed_from_floor` they shoal over
the shallows.

### Which mesh?

| Water shape | Use |
| --- | --- |
| Square or rectangular pool | **no mesh**: the default `domain_size` plane is exact and cheaper |
| Round pond, kidney bean, island ring, irregular shoreline | any flat, low-poly mesh with an X/Z footprint: `CylinderMesh` disc, `PrismMesh`, a baked CSG shape, or a hand-built `ArrayMesh` polygon |
| Lake or pond that follows the terrain | the same, sized to where the ground sits below `water_level` |

Rules that actually matter:

- Only the triangles' X/Z positions are read. **Y is ignored**, as are UVs,
  normals and density — coarse geometry is fine.
- Only **up-facing** triangles are used, so a closed shape's bottom and walls
  are dropped. A mesh with no up-facing triangles falls back to all of them,
  so flipped winding still works.
- Author the mesh at final size and keep the node's scale at 1.
- `render_extent` is ignored with a `surface_mesh`; the footprint is what is
  drawn.

### Sizing

The simulation fits a **square** domain around the footprint's bounds, so the
longest side sets the ripple cell size:

```
cell size (m) ≈ longest side of the footprint / (grid_resolution - 2)
```

Aim for ~0.2–0.4 m cells (20 m pond: 96; 60 m: 192; 150 m: 256). A long, thin
shape wastes most of that square and gets coarse cells — split a river into
sections or keep the aspect under ~3:1.

```gdscript
var pond := WaterBody3D.new()
pond.render_extent = 0.0    # a pool, not a sea
var disc := CylinderMesh.new()
disc.top_radius = 20.0      # 40 m across
disc.bottom_radius = 20.0
disc.height = 0.1           # Y is ignored; any shallow height works
pond.surface_mesh = disc
pond.grid_resolution = 96   # 40 m / 94 -> ~0.42 m ripple cells
pond.wind_speed = 4.0       # fetch comes from the footprint: a pond gets small ripples
pond.wave_amplitude = 0.3
```

The footprint also sets the auto `fetch` (its longest side), so enclosed water
gets short wind ripples while a plain open square is a developed sea.

## Beaches and shorelines

Enable `seabed_from_floor`; the first physics tick casts rays down onto static
bodies in `seabed_collision_mask` (layer 1 by default). Then:

- depth is measured per ripple cell, so waves slow and bunch over shallows;
- the waterline follows the ground, with swash run-up, shore foam, wet sand;
- the ocean chop fades out over `shallow_fade_depth` meters.

The beach or lake bed needs a static collider: a `StaticBody3D` with a
`CollisionShape3D` (a tilted box makes a serviceable beach — see
`demo/scenes/shore.tscn`).

## Boats and buoyancy

`WaterBuoyancy3D`, `WaterBoat3D`, `WaterWake3D` and `WaterSpray3D` are children
of the `RigidBody3D` they drive. Buoyancy samples the hull's collision shapes
(6 points by default) and applies Archimedes forces — it works with any physics
engine, Jolt included.

```
Boat (RigidBody3D, mass 500)
├── Hull (MeshInstance3D + CollisionShape3D)
├── Buoyancy (WaterBuoyancy3D)   water_surface_path = ../WaterBody3D
├── Drive    (WaterBoat3D)       propeller_position below the waterline
└── Wake     (WaterWake3D)       water_surface_path = ../WaterBody3D
```

```gdscript
# WaterBoat3D: throttle -1..1, steering -1..1 (+1 turns left).
func _physics_process(_delta: float) -> void:
	throttle = Input.get_action_strength("ui_up") - Input.get_action_strength("ui_down")
	steering = Input.get_action_strength("ui_left") - Input.get_action_strength("ui_right")
```

- `hull_area` / `hull_height` default to three quarters of the collision
  bounds' footprint and the bounds' height. `water_density` is 1000 kg/m³.
- The propeller must be under the water at rest, or there is no thrust.
- `WaterWake3D` keeps its foam and displacement where the boat passed; set
  `hull_beam` / `hull_length` (or `hull_profile`) so the foam hugs the hull.
- `WaterSpray3D` drives `GPUParticles3D` emitters you author: assign the bow,
  stern and slam emitter node paths; save them with `emitting = false`.

## Querying the water from scripts

```gdscript
# Height of the surface at a world XZ (bilinear, refreshed every physics tick).
var height := water.sample_height(global_position)
if height != -INF and global_position.y < height:
	print("wet; %.2f m under the surface" % (height - global_position.y))

# Is there water at all here (outside a lake's footprint: false)?
if water.is_wet(global_position):
	pass
```

`sample_height()` is a CPU readback of the GPU simulation, a few frames behind:
fine for buoyancy, not for sub-frame precision. For buoyancy scripts, pass your
own wake to `sample_height_excluding_wake(pos, wake)` so a boat doesn't ride
its own dip.

Make floating bodies push the water, not just read it:

```gdscript
water.submit_sphere(body.get_instance_id(), body.global_position, 1.2, 1.0)  # every physics tick
water.clear_sphere(body.get_instance_id())                                    # when it leaves
water.submit_impulse(pos, 2.0, 1.0)                                           # one-off splash
```

## Look and feel

The built-in material exposes its look as shader uniforms (select the
`WaterBody3D`, open its `water_material` in the inspector, or set them from
code). The ones worth touching first:

| Uniform | Default | Effect |
| --- | --- | --- |
| `water_color` / `shallow_color` | teal / milky | Body tint and the churned colour over shallows (`shallow_tint` blends them) |
| `deep_color` / `absorption` | dark blue / 0.22 | How quickly deep water hides what's behind it |
| `reflection_strength` | 1.0 | Sky reflection weight (0 = flat, matte water) |
| `specular_strength` / `specular_shininess` | 0.45 / 480 | Sun glitter |
| `crest_sss` | 0.6 | Subsurface glow on backlit crests |
| `detail_normal_strength` / `_scale` | 0.35 / 0.35 | Fine surface texture (generated at startup) |
| `distance_fade_start` / `_end` | 100 / 650 | Distance band-limiting for distant shimmer |
| `foam_threshold` / `foam_persistence` | node props | Whitecaps and how long they linger |

`quality` on the node picks presets: `Low` (64 grids, no detail normals, no
crest glow), `Medium` (96, one layer), `High` (128, two layers + glow),
`Custom` (leave properties as authored).

`screen_space` (off by default) rebuilds the material with a screen-sampling
variant and adds refraction. Before relying on it, run
`demo/scenes/screen_probe.tscn`: the 1:1 copy quad must be invisible; if it is
washed out, the capture path is broken and refraction stays off.

## Underwater

`WaterBody3D` adds this for you (`underwater_effect`, on by default): while the
camera is below the surface the frame is graded, and a hidden
`WaterUnderwaterEffect` child does the work. Add the node yourself only to
override its settings or to drive several water bodies from one place — and
turn `underwater_effect` off on the body in that case. While the camera is below the surface the
environment's fog, brightness, contrast and saturation ease to the
underwater values; `submerged_changed(bool)` fires on the crossing for audio
bus filters or gameplay. The surface's own Snell's-window / total-internal-
reflection look comes from the material; this node only grades the frame.

## Caustics

```gdscript
water.caustics_enabled = true
water.caustics_sun_direction = Vector3(-0.4, -1.0, -0.2)  # where the sun light travels
```

The node renders a light-space caustics map; any receiving material samples it
with the projection helpers:

```glsl
vec3 wp = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
vec2 p = vec2(dot(wp - caustics_origin, caustics_right), dot(wp - caustics_origin, caustics_up));
vec4 c = texture(caustics_tex, p / (2.0 * caustics_extent) + 0.5);
EMISSION = caustics_color.rgb * (c.r * c.g * caustics_strength);  // green = over water
```

`get_caustics_texture()`, `get_caustics_origin()`, `get_caustics_light_right()`,
`get_caustics_light_up()`, `get_caustics_half_extent()` read the current values;
`get_caustics_tile_size()` is the repeat period. A working receiver is in
`demo/shaders/seabed_caustics.gdshader`.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Water is a flat sheet, no waves | No RenderingDevice compute: Compatibility renderer or headless. Use Forward+/Mobile. |
| Nothing renders at all | The node is not inside the tree/world, or too far from the camera's far plane — check the camera, then `render_extent`. |
| Surface looks unlit / no sun glitter | No `DirectionalLight3D` in the scene and `sun_path` empty, or the sun faces away. |
| Shoreline never gets wet/dry | `seabed_from_floor` off, or the seabed's static body is not in `seabed_collision_mask`. |
| Boat does not move | Propeller above the waterline (`propeller_position`), or no water path on the buoyancy node. |
| Water slides when the player moves | Use `recenter()`, not the node's transform. |
| Speckle/shimmmer at distance | Lower `distance_fade_start`/`_end` on the material, or enable FXAA/SMAA — see "Aliasing" in the module README. |
| Overbright/white water | A custom `water_material` that samples the screen texture: confirm with `demo/scenes/screen_probe.tscn`. |
| The water node moves itself, or ignores where you put it | `auto_recenter` (on by default for open water) is following the camera; disable it, or give the water a `surface_mesh`. |

## Performance

- Solver cost is dominated by `grid_resolution`² + `ocean_grid_resolution`²
  compute dispatches per tick; use `quality` or set them directly.
- `render_extent` grows the vertex count (graded outward, not uniform), and
  the per-pixel normal path costs a few extra texture fetches; `normal_mode =
  Per Vertex` is cheaper and faceted.
- `caustics_enabled` adds a draw pass every frame, and `screen_space` adds a
  screen copy; both are off by default.
- At hundreds of kilometers from the origin, float precision jitters like any
  Godot scene: shift the world or use a double-precision build.

## Where to go next

- Class reference: `doc_classes/` (in-editor F1 on any water class).
- Demo scenes: `demo/scenes/` — open sea, shore, boat, underwater, refraction,
  screen probe, and `checklist.tscn` for automated checks.
- Module overview and design notes: [`../README.md`](../README.md).
