# Water — ocean, lakes and rivers

New to the module? [`docs/usage.md`](docs/usage.md) is the step-by-step guide: minimal
setup, open sea, lakes, beaches, boats, tuning and troubleshooting.

A bare `WaterBody3D` is a usable open sea: 1 km render extent, 128² solver
grids, the ripple patch follows the camera (`auto_recenter`) and underwater
grading is automatic (`underwater_effect`). Pools and lakes set
`render_extent = 0` or a `surface_mesh`; the rest of the setup is optional.

A GPU water system: a Tessendorf FFT ocean with a wind-driven JONSWAP spectrum
and a local shallow-water ripple layer that bodies disturb, rendered as one
displaced surface. `WaterBody3D` is the body; buoyancy, boats, wakes and spray
are separate nodes that read or disturb it.

Runs on any GPU with `RenderingDevice` compute (Forward+ or Mobile renderer).
No PhysX SDK and no particular physics backend is required: everything here
applies ordinary `RigidBody3D` forces, so Jolt and Godot Physics work too. On a
build without compute (headless, Compatibility renderer) the surface stays flat
and `sample_height()` returns the still-water level — a no-op, not an error.

| Node | Role |
| --- | --- |
| `WaterBody3D` | The water surface: FFT ocean + ripple layer, material, caustics, `sample_height()`/`is_wet()` queries |
| `WaterBuoyancy3D` | Floats its parent `RigidBody3D` from multi-point hull sampling |
| `WaterBoat3D` | Propeller thrust, steering and hull drag for a `RigidBody3D` |
| `WaterWake3D` | A moving GPU wake grid and foam, drawn by the water and sampled by buoyancy |
| `WaterSpray3D` | Drives authored `GPUParticles3D` bow/stern/slam emitters |
| `WaterUnderwaterEffect` | Underwater fog/grade, driven by the camera's submersion |
| `WaterRippleProbe` | Test/automation probe for the shared `RenderingDevice` |

## The water body — `WaterBody3D`

An animated water surface with two GPU compute layers summed into one displaced
mesh (created automatically as a child): a Tessendorf FFT ocean with a
fetch-limited JONSWAP wind spectrum (`wind_speed`, `wind_direction`, `fetch`,
over `ocean_domain_size`; heights in meters at `wave_amplitude` 1) and a local
ripple layer — a damped shallow-water wave equation over `domain_size` whose
wave speed follows `depth` — that bodies disturb.

The water is a square by default. Set `surface_mesh` to any flat mesh (a disc
baked from `CSGCylinder3D`, a kidney bean, a lake with an island) and its X/Z
footprint becomes the water: the ripple grid fits itself to the outline, waves
reflect off the shoreline, and the rendered surface is the footprint resampled
into an even grid, so a coarse or fan-triangulated mesh still animates
everywhere. With `fetch` left at 0 the footprint also sets the fetch, so an
enclosed pool only gets small, short wind ripples while an open square is a
developed sea. The simulation is centered on the node's position; keep the node
unrotated and unscaled.

`choppiness` adds the ocean waves' horizontal displacement, so crests come
to a point and troughs flatten. Where it squeezes the surface past
`foam_threshold` whitecap foam forms; it's kept with the water, not the wave,
and fades over `foam_persistence` seconds, so crests leave streaks that break
into lace. A strong wind gets whitecaps and a light one doesn't. `normal_mode` picks how normals and foam are shaded: Per Pixel
(default) uses exact slopes from the FFT, so crests stay smooth at any mesh
density; Per Vertex follows the mesh, for a softer or faceted, stylized look.

Set `seabed_from_floor` for water over uneven ground, like a beach or a lake
bed: each ripple cell's depth is measured once from the floor (static bodies in
`seabed_collision_mask`). Waves then travel at the local shallow-water speed,
slowing over shallows, the shoreline falls wherever the ground rises above the
water, and the ocean chop fades out over the last `shallow_fade_depth` meters.
Without it, the constant `depth` applies everywhere. On a beach, each arriving
crest also leaves a thin foam sheet at the waterline (`shore_foam_band`) that
surges in with the wave and slides back out with the backwash and undertow
(`shore_undertow`), the shallows turn milky, and the surface feathers out onto
the sand along a waterline that scallops with the arriving waves rather than
following the depth contour. That edge is measured in meters of water
(`swash_reach`, `shore_edge_softness` in the material), so it behaves the same
whatever `shallow_fade_depth` is. The bigger waves also run up past the
still-water line as a thin film on the sand (`swash_run_up`,
`swash_drain_speed`), and the sand they cover stays dark and glossy while it
dries (`wet_sand_dry_time`). The material's `shore_detail_scale` sets how fine
the swash edge, the rim lace and the wet-sand line are. The shore patterns
mirror-repeat past the simulated square, so the beach keeps its variation all
along the shore.

For open water, `render_extent` renders the surface out to the horizon: the FFT
ocean tiles seamlessly past the simulated square and ripples settle to still
water there. The surface draws only the side facing the camera (the top from
above, the underside from below), and its shaders take their clock from the
node's `water_time` parameter rather than `TIME`, so an open scene doesn't
keep the editor redrawing (and competing with Play Scene for the GPU).

**An endless sea that follows the player.** The FFT ocean is periodic and
fixed to the world (it's sampled at world X/Z, by the shader and by
`sample_height()` alike), so it has no edge: it covers the whole world by
itself. Only the ripple square is bounded — the interactive patch where
floaters splash — so keep it with the player: call `recenter(position)` when
the player gets some distance from the node (the ocean storm demo uses 10 m).
It moves the square without rebuilding the surface: the mesh, the textures and
the ocean carry on, the waves don't move, and only the square's ripples are
cleared (a surface with a `surface_mesh` footprint or a measured seabed still
rebuilds, since its depths depend on where the square is). Together with
`render_extent` that's an endless ocean out to the horizon. The one limit is
ordinary floating-point precision far from the world origin — at hundreds of
kilometers rendering and physics start to jitter, as for any Godot scene; large
worlds shift their origin or use a double-precision build. Over a coastline,
turn on `open_sea_beyond` so the sea past a moving square stays open deep water
instead of carrying the square's edge — its seabed, shore foam and wet sand —
outward.

A custom water material must sample the ocean textures at world X/Z too
(`world_xz / ocean_domain_size`, not relative to `grid_center`), or its drawn
waves slide by the re-centre distance on every `recenter()` while the physics
stays put.

The sea state can change during play: setting `wind_speed`, `wind_direction`,
`wave_amplitude` or `fetch` on a running surface (or all four at once with
`set_sea_state()`) regenerates the wave spectrum in place — every wave keeps its
phase and grows, calms or turns, and ripples, wakes and foam carry on — so a
storm can build smoothly, every frame if needed. Changing `ocean_domain_size`
still rebuilds the surface.

`sample_height(position)` returns the same combined surface height on the CPU,
for buoyancy scripts. Over dry cells (outside a `surface_mesh` footprint, or
where the seabed is above the water) it returns `-INF` and `is_wet()` is false,
so floaters on dry land don't float. Call `submit_sphere()` once per physics tick per floating
body so it actually pushes the water (`ripple_amplitude` scales the wake), or
`submit_impulse()` for one-off splashes.

`caustics_enabled` (off by default -- the pass draws every frame, so turn it on
where a floor shows through shallow water) renders a light-space caustic map
with a direct `RenderingDevice` draw pass along `caustics_sun_direction`. Any material — the
floor, walls, submerged props — can sample it by projecting its world position
with `get_caustics_texture()`, `get_caustics_origin()`,
`get_caustics_light_right()` / `_up()` and `get_caustics_half_extent()`, so the
light pattern lands on anything under the water rather than only a flat floor.
The map's green channel marks where sunlight actually came through the water
(multiply by it, or a pool wall's outside picks up caustics), and the pattern
repeats every `ocean_domain_size` (`get_caustics_tile_size()`), so a receiver
can fold its position into one tile and light a whole seabed.

Runs on any GPU with compute support, no CUDA needed. Without compute (e.g.
headless) the surface stays flat and `sample_height()` returns `water_level`.

## Buoyancy, boats, wakes and spray

Four nodes, each a child of a `RigidBody3D`, that float and drive it on a
`WaterBody3D`. Buoyancy and the boat only apply `RigidBody3D` forces,
so they work with whichever physics engine is selected, Jolt included.

- **`WaterBuoyancy3D`** floats the body: sample points on the hull (six from
  its collision shapes' bounds by default) each carry a share of `hull_area`
  and are pushed up by the water they displace, so the body settles at its
  real draft, pitches and rolls with the waves and rights itself. It reads the
  water with `sample_height()`, refreshed every physics tick, waves and other
  bodies' wakes included.
- **`WaterBoat3D`** drives it: thrust from a propeller at `propeller_position`
  (only while it's in the water) that `steering` turns like an outboard, the
  water across the propeller's leg (it steers with flow and holds the stern on
  course), and hull drag in the boat's frame — low along the keel, high
  sideways and vertically, at the bow and stern so the hull's length resists
  turning and pitching. `hull_drag.z` with `max_thrust` sets the top speed.
- **`WaterWake3D`** leaves a wake: a small GPU ripple grid that slides
  with the boat (fixed to the water, so the wake stays where it was made) and
  that the hull pushes down, plus foam — two bands peeling off the hull's
  sides and a propeller band down the middle, fading over `foam_persistence`.
  The water material draws it and `sample_height()` includes it, so other
  floaters ride it. A grid big enough to hold a turning circle (`size`, with
  `cells` for about 0.25 m cells) and a `foam_persistence` of about a lap let
  the wake close into a ring. `hull_profile` gives the hull's shape seen from
  above (8 half-beam widths, bow to stern), so a pointed bow's foam hugs its
  sides; left empty, the hull is treated as a box.
- **`WaterSpray3D`** drives ordinary `GPUParticles3D` nodes authored in
  the editor: bow emitters with speed, stern emitters with thrust, one-shot
  slam emitters when the hull drops back onto the water. Emitters are held at
  the water under their place on the hull. The spray leaves foam where it
  lands (worked out from the emitters' launch geometry, no particle collision
  needed); `water_collision` adds a collider on the water for spray that
  should land and skid. Save the emitters with `emitting` off — the node
  starts them — or the editor redraws every frame for them.

## Shading

The built-in material is unshaded and alpha blended on purpose: Godot's
transparent pass has no scene lights, and blending two independently lit PBR
surfaces exceeds reference white before the single tonemap pass. So the water
contributes its own bounded terms instead:

- Fresnel sky reflection from a zenith/horizon/ground gradient, kept in step
  with the scene's `ProceduralSkyMaterial`, plus a sun glow in the reflection.
- Sun glitter from the first `DirectionalLight3D` (or `sun_path`), Blinn
  specular modulated by the wave normals.
- Crest subsurface glow, strongest when the sun is behind a backlit crest.
- A tiling detail normal generated at startup (overridable) and scrolled over
  the surface, layered on the FFT's exact per-pixel normals.
- Depth-buffer absorption: water thick along the view ray keeps the deep
  color and turns opaque, thin water stays clear, and the surface fades out
  where it grazes opaque geometry.

`quality` (Low/Medium/High) sets 64/96/128 solver grids and the shading tier
(detail layers, crest glow); `Custom` leaves every property as authored.

`screen_space` adds true refraction: the material is rebuilt with a
`#define WATER_SCREEN_SPACE` variant that samples the color buffer and
composites at alpha 1. It is **off by default** because an earlier attempt at
hand-composited screen-space water on this engine build read overbright; the
`scenes/screen_probe.tscn` demo scene is a 1:1 copy test that says whether
the capture is usable before you rely on it.

Seen from below the surface, the material shows Snell's window (the sky
through the surface, inside the 48.6 degree cone) and total internal
reflection outside it, plus the underwater grade from
`WaterUnderwaterEffect`.

Not implemented yet: screen-space reflections (built-in SSR is opaque-only, a
transparent water surface cannot receive it; the in-shader trace needs a real
GPU to iterate on), planar reflections, volumetric god rays, and a half-res
refraction path. Sky reflection + sun glitter cover the calm-water case
meanwhile.

## Aliasing

Distant water aliases from four different sources, so the material band-limits
each one by camera distance (`distance_fade_start` / `distance_fade_end` on
the material):

- Procedural foam/shore noise has no mip chain: its lace melts to its average
  value, and the waterline jitter is dropped, so the edge stops crawling.
- The tiling detail normal now has a mip chain and anisotropic sampling
  (`filter_linear_mipmap_anisotropic`).
- Sub-pixel waves average out: the per-pixel normal is blended toward up, and
  the sun glint widens (shininess toward 90) and dims.
- Foam sprite density fades with distance, as whitecaps do through the air.

Engine-side options, cheapest first: `msaa_3d` (polygon edges, e.g. the
horizon silhouette), `screen_space_aa` FXAA/SMAA (post-process; does smooth the
blended water), `use_taa` (temporal — the water writes no motion vectors, so
check it for ghosting on the water before shipping), and
`rendering/scaling_3d/scale > 1` supersampling (best quality, most expensive).
The demo's `project.godot` lists them as commented settings.

## Demo

`demo/` is a self-contained Godot project with one scene per case:

| Scene | Shows |
| --- | --- |
| `scenes/open_sea.tscn` | open ocean out to the horizon, with a floating crate |
| `scenes/shore.tscn` | a sloped beach: shoaling waves, swash, shore foam, wet sand |
| `scenes/boat.tscn` | a `RigidBody3D` boat with buoyancy and a wake; arrows drive (Esc quits) |
| `scenes/underwater.tscn` | the surface seen from below, with underwater grading; the HUD shows SURFACE/UNDERWATER, camera and water height |
| `scenes/refraction.tscn` | screen-space refraction enabled, over a shallow floor |
| `scenes/screen_probe.tscn` | 1:1 screen-copy probe: the quad must be invisible |

The module is enabled in every standard build — no SDK or build flag is
needed:

```
scons platform=linuxbsd target=editor -j$(nproc)
```

Run a scene with the engine built from this checkout, from the repository root:

```
./bin/godot.linuxbsd.editor.x86_64 --path modules/water/demo res://scenes/open_sea.tscn
```

`scenes/checklist.tscn` runs the automated checks and prints PASS/FAIL/SKIP
per line (it exits non-zero when something fails):

```
./bin/godot.linuxbsd.editor.x86_64 --path modules/water/demo res://scenes/checklist.tscn
```

## Layout

| Path | Contents |
| --- | --- |
| `water_body_3d.*` | `WaterBody3D`: the surface node, its mesh/material and CPU queries |
| `water_solver.*` | the GPU solver: ripple layer, FFT spectrum, foam, shore, caustics, readback |
| `water_wake_solver.*` | the moving wake grid solver |
| `water_buoyancy_3d.*`, `water_boat_3d.*` | buoyancy and boat forces |
| `water_wake_3d.*`, `water_spray_3d.*` | wake and spray nodes |
| `*.glsl` | compute and draw shaders, embedded by `SCsub` through `GLSL_HEADER` |
| `water_underwater_effect.*` | the underwater environment grade |
| `demo/` | a Godot project with open sea, shoreline, boat, underwater, refraction and probe scenes |
| `docs/` | the usage guide (`usage.md`) |
| `doc_classes/` | class reference |

## License

MIT — see the Godot Engine `LICENSE.txt`.
