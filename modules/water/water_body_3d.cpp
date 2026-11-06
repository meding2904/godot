/**************************************************************************/
/*  water_body_3d.cpp                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "water_body_3d.h"

#include "water_temporal_effect.h"
#include "water_underwater_effect.h"
#include "water_wake_3d.h"

#include "core/config/engine.h"
#include "core/io/image.h"
#include "core/math/face3.h"
#include "core/object/class_db.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "scene/main/window.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/3d/sky_material.h"
#include "scene/resources/3d/world_3d.h"
#include "scene/resources/environment.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/shader.h"
#include "scene/resources/sky.h"
#include "scene/resources/texture_rd.h"
#include "servers/physics_3d/physics_server_3d.h"
#include "servers/rendering/rendering_server.h"

#ifdef TOOLS_ENABLED
#include "editor/scene/3d/node_3d_editor_plugin.h"
#endif

namespace {
// A plain spatial Shader (Godot's own shading language, NOT the RD compute
// shaders the rest of this module uses). Displaces the mesh's own VERTEX.y
// by two independently-sampled height fields (ripple + FFT ocean, summed)
// sampled directly from the solver's shared-RenderingDevice textures via
// Texture2DRD; the separate CPU readback cache is only for sample_height().
//
// Two transparency paths, one material source:
// - Default: Godot's own GPU alpha blending (ALPHA, blend_mix) over the
//   scene, plus the depth buffer for soft contact edges and view-ray
//   absorption. No screen-color sampling, so it costs no extra pass.
// - WaterBody3D.screen_space: the material is rebuilt with
//   `#define WATER_SCREEN_SPACE` and samples the color buffer for refraction.
//   Off by default: an earlier manual screen-space composite on this build
//   read severely overbright (even at 15% strength it blew the surface to
//   solid white, and the engine has no REFRACTION built-in -- checked against
//   servers/rendering/shader_types.cpp). The toggle exists so that can be
//   re-tested on a real GPU; until then the alpha-blended path is what ships.
//
// Real per-vertex normals from a central-difference of the SAME combined
// height field the vertex is displaced by -- the surface used to move but
// never actually shade like it moved, which was a real, separate
// contributor to an earlier "looks flat" complaint. Feeds a small, bounded,
// self-contained shading term in fragment() (unshaded, not Godot's real
// PBR NORMAL) -- see fragment()'s own comment for why real PBR lighting on
// this material caused a genuine overexposure bug.
const char *WATER_SHADER_SRC = R"(
shader_type spatial;
render_mode blend_mix, cull_disabled, unshaded;

uniform sampler2D ripple_height_tex : hint_default_black, filter_linear, repeat_disable;
uniform sampler2D ocean_height_tex : hint_default_black, filter_linear, repeat_enable;
// Ocean chop multiplier over the ripple domain: 1 in deep water, fading to 0
// at the waterline (set by WaterBody3D when it has a seabed).
uniform sampler2D ocean_fade_tex : hint_default_white, filter_linear, repeat_disable;
// Choppy horizontal displacement (x, z) of the ocean layer; scaled by
// choppiness (both set by WaterBody3D).
uniform sampler2D ocean_disp_tex : hint_default_black, filter_linear, repeat_enable;
uniform float choppiness = 1.0;
// Ocean slopes and displacement derivatives (dh/dx, dh/dz, dDx/dx, dDz/dz),
// straight from the FFT, for exact per-pixel normals and foam.
uniform sampler2D ocean_deriv_tex : hint_default_black, filter_linear, repeat_enable;
// true: normals and foam per pixel from ocean_deriv_tex (smooth crests at any
// mesh density). false: per vertex, from finite differences of the displaced
// mesh (set by WaterBody3D::normal_mode).
uniform bool per_pixel_normals = true;
uniform vec2 ripple_domain_size = vec2(20.0, 20.0);
uniform vec2 ocean_domain_size = vec2(40.0, 40.0);
// World XZ both height fields are centered on (set by WaterBody3D).
uniform vec2 grid_center = vec2(0.0);
// Open, deep sea past the simulated square (WaterBody3D::open_sea_beyond) instead of the edge's seabed.
uniform bool open_sea_beyond = false;
// World xz the whole open-water mesh is moved by, so its dense core stays
// under the camera; set by the node every frame.
uniform vec2 mesh_offset = vec2(0.0);
// Sub-pixel offset (NDC) for the water-only temporal resolve.
uniform vec2 temporal_jitter = vec2(0.0);
// Moving wake grids (WaterWake3D), up to four: height textures and
// their rects (xy world xz of the corner, z side length -- 0 = unused, w how
// much the wake fades out over the shore ripple grid, 0..1).
uniform sampler2D wake_height_0 : hint_default_black, filter_linear, repeat_disable;
uniform sampler2D wake_height_1 : hint_default_black, filter_linear, repeat_disable;
uniform sampler2D wake_height_2 : hint_default_black, filter_linear, repeat_disable;
uniform sampler2D wake_height_3 : hint_default_black, filter_linear, repeat_disable;
uniform vec4 wake_rect_0 = vec4(0.0);
uniform vec4 wake_rect_1 = vec4(0.0);
uniform vec4 wake_rect_2 = vec4(0.0);
uniform vec4 wake_rect_3 = vec4(0.0);
// Mean surface height (node-relative), what the ripple layer settles back to
// past the edge of its domain when the surface is rendered further out.
uniform float water_level = 0.0;
// Animation clock for the shore noise, set by WaterBody3D -- not the shader's
// built-in clock, so the editor preview and the running game share one clock
// and it stops when the node stops stepping.
uniform float water_time = 0.0;
uniform vec4 water_color : source_color = vec4(0.09, 0.32, 0.42, 0.65);
// Persistent whitecap foam (0..1) on the ocean grid, injected where the
// waves fold and fading over the node's foam_persistence (set by
// WaterBody3D).
uniform sampler2D ocean_foam_tex : hint_default_black, filter_linear, repeat_enable;
uniform vec4 foam_color : source_color = vec4(0.92, 0.95, 0.97, 0.95);
// How much fading foam breaks up into lace instead of thinning evenly
// (0 = solid sheet until it's gone).
uniform float foam_breakup : hint_range(0.0, 1.0) = 0.6;
// Size of that lace: noise cells per meter, fixed to the water.
uniform float foam_detail_scale = 1.5;
// Shore foam (0..1) on the ripple grid: the foamy sheet where waves wash in
// over the shallows (set by WaterBody3D; needs a seabed).
uniform sampler2D shore_foam_tex : hint_default_black, filter_linear, repeat_disable;
// Shore foam is a thin sheet, not a whitecap: fainter, finer and lacier.
uniform float shore_foam_strength : hint_range(0.0, 1.0) = 0.65;
uniform float shore_foam_breakup : hint_range(0.0, 1.0) = 0.85;
// Churned, milky water over the shallows: tints toward shallow_color as the
// ocean fade drops off near the shore (shallow_tint 0 = off). Its alpha also
// replaces water_color's there.
uniform vec4 shallow_color : source_color = vec4(0.42, 0.66, 0.64, 0.55);
uniform float shallow_tint : hint_range(0.0, 1.0) = 0.6;
// Still-water depth in meters over the ripple domain (dry shore negative,
// pool walls deep), set by WaterBody3D: the waterline is measured in
// real depth, independent of shallow_fade_depth.
uniform sampler2D shore_depth_tex : hint_default_white, filter_linear, repeat_disable;
// Soft contact with the shore: the surface fades out over this much water
// depth (m) at the waterline instead of meeting the sand at a hard line.
uniform float shore_edge_softness : hint_range(0.0, 0.5, 0.005) = 0.08;
// Moving waterline: how much water depth (m) the visible edge can pull back
// over. Arriving crests push it toward the sand, troughs pull it back, and
// slow ground-fixed noise keeps it irregular -- so the edge scallops with the
// waves instead of following a depth contour, but only ever goes clear where
// the water really is that thin.
uniform float swash_reach : hint_range(0.0, 1.0, 0.01) = 0.25;
// Swash and wet sand from WaterBody3D (swash_run_up, ...): r = how
// high (m) above still water the last waves ran up, g = sand wetness.
uniform sampler2D swash_tex : hint_default_black, filter_linear, repeat_disable;
// Thickness (m) of the swash film over the sand -- just enough to stay above
// the ground mesh.
uniform float swash_film_thickness : hint_range(0.0, 0.2, 0.005) = 0.03;
// Fine ragged fingering of the swash edge: how much water depth (m) drifting
// small-scale noise adds or takes away there (on a 1:14 beach, 0.035 m is
// about half a meter of ragged edge).
uniform float swash_edge_breakup : hint_range(0.0, 0.2, 0.005) = 0.035;
// Size of the shore's small-scale detail -- the lace on the waterline rim,
// the fingering of the swash edge and the ragged wet-sand line, which
// foam_detail_scale doesn't reach. 1 = default; higher = finer.
uniform float shore_detail_scale : hint_range(0.1, 5.0, 0.05) = 1.0;
// Wet sand: rgb blended over the ground where the water has been, alpha the
// strength -- darkens it, fading as it dries.
uniform vec4 wet_sand_color : source_color = vec4(0.16, 0.12, 0.07, 0.45);
// Wave height (m) that pushes the waterline all the way in.
uniform float swash_wave_height = 0.3;
// Thin bright foam rim just behind the moving edge.
uniform float shore_rim_strength : hint_range(0.0, 1.0) = 0.8;

// --- Shading (Phase 1) ---------------------------------------------------
// This material is unshaded and alpha blended (see the comment above), so it
// has no scene lights of its own: the sun and sky below are set by
// WaterBody3D from the scene's DirectionalLight3D and ProceduralSkyMaterial.
// Every term stays bounded on purpose -- this is a tint on top of an already
// lit scene, not a second PBR response.
uniform vec3 sun_direction = vec3(0.3, 0.85, 0.4); // from the surface toward the sun, world space
uniform vec3 sun_color = vec3(1.0, 0.96, 0.9);
uniform float sun_energy = 1.0;
uniform vec3 sky_zenith_color = vec3(0.13, 0.30, 0.62);
uniform vec3 sky_horizon_color = vec3(0.62, 0.73, 0.86);
uniform vec3 sky_ground_color = vec3(0.08, 0.14, 0.20);
uniform float reflection_strength = 1.0;
uniform float specular_strength = 0.45;
uniform float specular_shininess = 480.0;
// Geometric specular AA: how much the normal's change across one pixel widens
// the sun glint (0 = off). A pixel whose normals vary stands for a range of
// normals, so the highlight is the lobe convolved with that range instead of
// a point sample that flickers between frames.
uniform float specular_aa : hint_range(0.0, 2.0, 0.05) = 1.0;
uniform float crest_sss = 0.6;
uniform vec3 crest_sss_color = vec3(0.06, 0.38, 0.33);
// Surface amplitude in meters (WaterBody3D::wave_amplitude), for normalizing
// the crest height the subsurface glow uses.
uniform float wave_amplitude = 1.0;
// Tiling detail normal, scrolled over the surface: the FFT supplies the
// large-scale normals, this adds the fine surface texture.
uniform sampler2D detail_normal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform float detail_normal_strength = 0.35;
uniform float detail_normal_scale = 0.35;
uniform float detail_normal_speed = 0.35;
// Distance (m) selecting the shading LOD layers and band-limiting the
// material: near is full; by _mid the fine detail is gone (detail normals
// dropped, procedural foam/shore lace melted to its average, the waterline
// jitter removed, the per-pixel normal flattened, the sun glint widened and
// dimmed); by _end the remaining procedural foam/shore/crest terms are gone
// and the geometry stops displacing (sub-pixel waves there only alias the
// horizon). Without this, detail smaller than a distant pixel shimmers.
uniform float distance_fade_start = 100.0;
uniform float distance_fade_mid = 350.0;
uniform float distance_fade_end = 650.0;
// Surface-height LOD response: those three distances are the base values,
// measured with the camera high above the surface. Skimming the water puts
// the whole far field into a few grazing pixels (every one running the
// fragment work, and the sub-pixel waves aliasing), so as the camera drops,
// all three shrink toward lod_surface_scale of themselves; they ramp back to
// full over lod_surface_height meters. Computed in lod_distances(), from the
// drawing camera, so split viewports each get their own.
uniform float lod_surface_scale = 0.40;
uniform float lod_surface_height = 80.0;

// 0 low (no detail normals, no crest SSS), 1 medium, 2 high. Set by
// WaterBody3D::quality.
uniform int shading_quality = 2;

// --- Depth (Phase 1) -----------------------------------------------------
// Opaque depth, for soft contact edges and depth-based absorption: water with
// a long view ray through it keeps the deep color, thin water stays clear.
uniform sampler2D depth_texture : hint_depth_texture, filter_nearest;
uniform float soft_edge_distance = 0.4;
uniform float absorption = 0.22; // per meter of view-ray water
uniform vec3 deep_color = vec3(0.02, 0.09, 0.13);

// --- Screen space (Phase 2) ----------------------------------------------
// Only compiled when the material's shader is built with
// `#define WATER_SCREEN_SPACE` (WaterBody3D.screen_space, off by default).
#ifdef WATER_SCREEN_SPACE
uniform sampler2D screen_texture : hint_screen_texture, filter_linear, repeat_disable;
uniform float refraction_strength = 0.02; // UV offset at full wave slope
uniform float refraction_mix = 0.9;
#endif

varying vec3 v_world_normal;
varying float v_foam;
varying vec2 v_rest_xz;

// Offset of the surface point that rests at world_xz: (dx, height, dz).
// One wake grid's height (x) and foam (y) at world_xz, faded out toward its edges.
vec2 wake_one(sampler2D tex, vec4 rect, vec2 world_xz) {
	if (rect.z <= 0.0) {
		return vec2(0.0);
	}
	vec2 uv = (world_xz - rect.xy) / rect.z;
	float edge = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
	if (edge <= 0.0) {
		return vec2(0.0);
	}
	// Optionally fade out over the shore ripple grid (where a floater's own
	// ripples can take over).
	vec2 from_center = abs((world_xz - grid_center) / ripple_domain_size);
	float outside = smoothstep(0.46, 0.5, max(from_center.x, from_center.y));
	vec4 t = texture(tex, uv); // height, hull foam, propeller foam
	return vec2(t.r, t.g + t.b) * smoothstep(0.0, 0.1, edge) * mix(1.0, outside, rect.w);
}

// 0 inside the simulated square .. 1 past it -- only with open_sea_beyond (else the edge carries on outward).
float beyond_w(vec2 world_xz) {
	vec2 a = abs((world_xz - grid_center) / ripple_domain_size);
	return open_sea_beyond ? smoothstep(0.46, 0.5, max(a.x, a.y)) : 0.0;
}

// The ocean chop multiplier / the still-water depth at world_xz (cuv_r = its clamped ripple UV): open, deep sea past
// the square with open_sea_beyond.
float fade_r(vec2 world_xz, vec2 cuv_r) {
	return mix(texture(ocean_fade_tex, cuv_r).r, 1.0, beyond_w(world_xz));
}

float depth_r(vec2 world_xz, vec2 cuv_r) {
	return mix(texture(shore_depth_tex, cuv_r).r, 1000.0, beyond_w(world_xz));
}

// All wakes at world_xz, everywhere (inside the shore ripple grid too).
vec2 wake_sample(vec2 world_xz) {
	vec2 w = wake_one(wake_height_0, wake_rect_0, world_xz) + wake_one(wake_height_1, wake_rect_1, world_xz)
			+ wake_one(wake_height_2, wake_rect_2, world_xz) + wake_one(wake_height_3, wake_rect_3, world_xz);
	// Not over dry sand: the wake dies out over the last 30 cm of depth.
	vec2 cuv_r = clamp((world_xz - grid_center) / ripple_domain_size + 0.5, vec2(0.0), vec2(1.0));
	return w * smoothstep(0.0, 0.3, depth_r(world_xz, cuv_r));
}

float wake_height(vec2 world_xz) {
	return wake_sample(world_xz).x;
}

// Foam the wakes churned in (bow wave, propeller wash, breaking wake waves).
float wake_foam(vec2 world_xz) {
	return min(wake_sample(world_xz).y, 1.0);
}

vec3 surface_offset(vec2 world_xz, float geom_far) {
	vec2 rel = world_xz - grid_center;
	vec2 cuv_r = clamp(rel / ripple_domain_size + 0.5, vec2(0.0), vec2(1.0));
	vec2 cuv_o = world_xz / ocean_domain_size + 0.5; // wraps: the FFT ocean is periodic -- anchored to the WORLD, so moving grid_center (a re-centre) never slides the waves
	float fade = fade_r(world_xz, cuv_r);
	// Negative: the displacement points away from crests in this FFT's sign
	// convention, so moving against it gathers points into the crests.
	vec2 d = -choppiness * fade * texture(ocean_disp_tex, cuv_o).rg;
	// The ripple layer isn't periodic: past its domain, ease back to still water.
	vec2 from_center = abs(rel / ripple_domain_size);
	float inside = 1.0 - smoothstep(0.46, 0.5, max(from_center.x, from_center.y));
	float ripple = mix(water_level, texture(ripple_height_tex, cuv_r).r, inside);
	float h = ripple + texture(ocean_height_tex, cuv_o).r * fade;
	// The wake grid is laid out where the water is now, not where it rests
	// (its sources sit under the hull, and sample_height() reads it there):
	// read it at this point's displaced position.
	h += wake_height(world_xz + d);
	// LOD: waves smaller than a pixel only alias the horizon silhouette, so
	// the far surface flattens onto the still level (the shading normal is
	// band-limited separately in fragment()). The ground clamp below still
	// lifts the thin film over a distant shore, so its waterline keeps its
	// swash film.
	h = mix(h, water_level, geom_far);
	d *= 1.0 - geom_far;
	// Never below the ground: over dry sand (and in the deepest troughs over
	// shallows) the surface lies on the sand as a thin film, which is where
	// the swash shows.
	float ground = water_level - depth_r(world_xz, cuv_r);
	h = max(h, ground + swash_film_thickness);
	return vec3(d.x, h, d.y);
}

// Exact normal of the displaced surface at the point resting at world_xz:
// tangents of P(x0) = x0 + s D(x0) + h(x0) from the FFT's own derivatives. The ripple layer has no FFT, so its slope
// is a small central difference.
vec3 surface_normal_pixel(vec2 world_xz) {
	vec2 rel = world_xz - grid_center;
	vec2 cuv_r = clamp(rel / ripple_domain_size + 0.5, vec2(0.0), vec2(1.0));
	vec2 cuv_o = world_xz / ocean_domain_size + 0.5; // wraps: the FFT ocean is periodic -- anchored to the WORLD, so moving grid_center (a re-centre) never slides the waves
	float fade = fade_r(world_xz, cuv_r);
	float s = -choppiness * fade;
	vec4 der = texture(ocean_deriv_tex, cuv_o);
	float dxz = texture(ocean_disp_tex, cuv_o).b;
	// LOD: the ripple layer only exists inside its own square -- past it the
	// surface is the still water_level, exactly as surface_offset() draws it,
	// so there is no slope to fetch and four taps to skip.
	vec2 ruv = rel / ripple_domain_size + 0.5;
	bool ripple_here = ruv.x > 0.0 && ruv.x < 1.0 && ruv.y > 0.0 && ruv.y < 1.0;
	float rx = 0.0;
	float rz = 0.0;
	if (ripple_here) {
		const float e = 0.2;
		vec2 du = vec2(e) / ripple_domain_size;
		rx = (texture(ripple_height_tex, cuv_r + vec2(du.x, 0.0)).r - texture(ripple_height_tex, cuv_r - vec2(du.x, 0.0)).r) / (2.0 * e);
		rz = (texture(ripple_height_tex, cuv_r + vec2(0.0, du.y)).r - texture(ripple_height_tex, cuv_r - vec2(0.0, du.y)).r) / (2.0 * e);
	}
	// The ripple layer steps at the waterline (dry cells report their wet
	// neighbours' mean), which the sun picked out as bright streaks along the
	// shore; over the sand the swash film just follows the ground.
	float wet_w = smoothstep(0.0, 0.15, depth_r(world_xz, cuv_r));
	rx *= wet_w;
	rz *= wet_w;
	// Wake slopes (zero over the shore grid), at the displaced position as
	// surface_offset() reads the wake.
	const float ew = 0.12;
	vec2 wake_xz = world_xz + s * texture(ocean_disp_tex, cuv_o).rg;
	rx += (wake_height(wake_xz + vec2(ew, 0.0)) - wake_height(wake_xz - vec2(ew, 0.0))) / (2.0 * ew);
	rz += (wake_height(wake_xz + vec2(0.0, ew)) - wake_height(wake_xz - vec2(0.0, ew))) / (2.0 * ew);
	vec3 tx = vec3(1.0 + s * der.z, der.x * fade + rx, s * dxz);
	vec3 tz = vec3(s * dxz, der.y * fade + rz, 1.0 + s * der.w);
	return normalize(cross(tz, tx));
}

float foam_hash(vec2 p) {
	vec3 p3 = fract(vec3(p.xyx) * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.x + p3.y) * p3.z);
}

float foam_value_noise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(mix(foam_hash(i), foam_hash(i + vec2(1.0, 0.0)), u.x),
			mix(foam_hash(i + vec2(0.0, 1.0)), foam_hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

// Visible foam for the water resting at world_xz: the persistent foam amount,
// eroded by fixed-to-the-water noise so it breaks into lace as it fades.
float foam_at(vec2 world_xz, float mid_filter) {
	// LOD mid tier: past its end one lace cell is smaller than a pixel and
	// the noise only shimmers; the foam has already faded out by then (the
	// (1 - mid_filter) fade below), so skip its fetch and noise.
	if (mid_filter > 0.995) {
		return 0.0;
	}
	vec2 rel = world_xz - grid_center;
	vec2 cuv_r = clamp(rel / ripple_domain_size + 0.5, vec2(0.0), vec2(1.0));
	vec2 cuv_o = world_xz / ocean_domain_size + 0.5; // wraps: the FFT ocean is periodic -- anchored to the WORLD, so moving grid_center (a re-centre) never slides the waves
	float fade = fade_r(world_xz, cuv_r);
	float amount = texture(ocean_foam_tex, cuv_o).r * fade;
	// Wake foam at the displaced position, as surface_offset() reads the wake.
	amount = max(amount, wake_foam(world_xz - choppiness * fade * texture(ocean_disp_tex, cuv_o).rg) * (1.0 - mid_filter));
	vec2 p = world_xz * foam_detail_scale;
	float n = 0.5 * foam_value_noise(p) + 0.3 * foam_value_noise(p * 2.7 + 17.0) + 0.2 * foam_value_noise(p * 7.3 - 5.0);
	// Past a few hundred meters one noise cell is smaller than a pixel: use
	// its average instead of its loop, or the lace shimmers as you move.
	n = mix(n, 0.5, mid_filter);
	// Gentle ramp: fresh foam is solid, thinning foam turns translucent and
	// lacy as the noise eats into it, rather than cutting off at an edge.
	float f = clamp(amount * (1.0 + foam_breakup) - n * foam_breakup, 0.0, 1.0);
	return f * f * (3.0 - 2.0 * f) * (1.0 - mid_filter);
}

// Visible shore foam at world_xz: finer, heavier lace at partial strength.
)"
							   R"(
// Ripple-grid UV for the shore's moving patterns (swash, wet sand, shore
// foam), mirror-repeated past the simulated square: they carry on down the
// beach without a seam instead of stretching the edge row into straight
// lines. (The still-water depth itself stays clamped, so land and sea are
// never mirrored into each other.)
vec2 shore_pattern_uv(vec2 world_xz) {
	vec2 u = (world_xz - grid_center) / ripple_domain_size + 0.5;
	return 1.0 - abs(mod(u, 2.0) - 1.0);
}

float shore_foam_at(vec2 world_xz, float mid_filter) {
	// LOD mid tier: the sheet is sub-pixel there and its lace only shimmers.
	if (mid_filter > 0.995) {
		return 0.0;
	}
	vec2 cuv_r = clamp((world_xz - grid_center) / ripple_domain_size + 0.5, vec2(0.0), vec2(1.0));
	vec2 q = world_xz * foam_detail_scale * 2.2 + 31.0;
	float ns = 0.5 * foam_value_noise(q) + 0.5 * foam_value_noise(q * 2.3 - 11.0);
	ns = mix(ns, 0.5, mid_filter);
	float shore = texture(shore_foam_tex, shore_pattern_uv(world_xz)).r * (1.0 - beyond_w(world_xz));
	float g = clamp(shore * (1.0 + shore_foam_breakup) - ns * shore_foam_breakup, 0.0, 1.0);
	return g * g * (3.0 - 2.0 * g) * shore_foam_strength * (1.0 - mid_filter);
}

// Water depth (m) above the moving waterline: the still-water depth minus an
// offset in [0, swash_reach] -- small under an arriving crest (the edge
// reaches in), large between crests, jittered by slow noise. <= 0 is past
// the visible edge.
float shore_signal(vec2 world_xz, float time, float mid_filter) {
	vec2 rel = world_xz - grid_center;
	vec2 cuv_r = clamp(rel / ripple_domain_size + 0.5, vec2(0.0), vec2(1.0));
	vec2 cuv_o = world_xz / ocean_domain_size + 0.5; // wraps: the FFT ocean is periodic -- anchored to the WORLD, so moving grid_center (a re-centre) never slides the waves
	float depth = depth_r(world_xz, cuv_r);
	float run_up = texture(swash_tex, shore_pattern_uv(world_xz)).r * (1.0 - beyond_w(world_xz));
	// LOD mid tier: both noise fields are sub-pixel past its end distance
	// (their effect was already mixed away at mid_filter 1) -- skip them.
	float offset = 0.0;
	float jitter = 0.0;
	if (mid_filter < 0.995) {
		float eta = texture(ocean_height_tex, cuv_o).r; // unfaded: the waves arriving here
		// Lobes a few meters across (like real swash), drifting, with the waves
		// pushing the whole edge in and out over time.
		vec2 p = world_xz * 0.45;
		float n = 0.65 * foam_value_noise(p + vec2(time * 0.12, time * 0.05)) + 0.35 * foam_value_noise(p * 2.4 - vec2(time * 0.07, 0.0));
		float wave = clamp(eta / max(swash_wave_height, 1e-3), -1.0, 1.0);
		offset = swash_reach * clamp(0.5 + (n - 0.5) * 1.6 - 0.35 * wave, 0.0, 1.0);
		vec2 q = world_xz * shore_detail_scale;
		float eb = 0.5 * foam_value_noise(q * 2.3 + vec2(time * 0.25, 0.0)) + 0.3 * foam_value_noise(q * 6.1 + vec2(3.0, time * 0.4)) + 0.2 * foam_value_noise(q * 15.0 - 7.0);
		// Far away, one cell of either noise is sub-pixel: drop the jitter, or
		// the waterline crawls as the camera moves.
		offset *= 1.0 - mid_filter;
		eb = mix(eb, 0.5, mid_filter);
		jitter = (eb - 0.5) * 2.0 * swash_edge_breakup;
		// On dry sand the jitter may only pull the edge back, never add water
		// past what the swash covers -- or it left a permanent fringe of rim
		// foam just above the still-water line.
		jitter = depth > 0.0 ? jitter : min(jitter, run_up);
	}
	// The pull-back gives way where a swash is running up, so over the sand
	// the film reaches exactly as far as the run-up (and the wet sand); with
	// no swash it applies right up to the still-water line, or a sliver of
	// water was left showing along it.
	return depth - offset * (1.0 - smoothstep(0.0, 0.08, run_up)) + run_up + jitter;
}

// The sky seen along a reflected direction: a horizon-to-zenith gradient, the
// ground color below the horizon, and a sun glow. Bounded by construction --
// this material is alpha blended on top of an already lit scene.
vec3 sky_reflection(vec3 dir, float mid_filter) {
	vec3 L = normalize(sun_direction);
	float up = clamp(dir.y, 0.0, 1.0);
	vec3 c = mix(sky_horizon_color, sky_zenith_color, pow(up, 0.55));
	c = mix(c, sky_ground_color, smoothstep(0.0, 0.25, -dir.y));
	float glow = pow(max(dot(dir, L), 0.0), 220.0);
	c += sun_color * (glow * 1.5 * (1.0 - 0.8 * mid_filter));
	return c;
}

// Distance along this fragment's view ray to the opaque surface behind it.
// The sky (or a cleared depth buffer) reads as "nothing behind": deep water,
// so the surface stays opaque instead of turning into a window.
float view_thickness(vec2 screen_uv, mat4 inv_projection, float fragment_distance) {
	float raw_depth = texture(depth_texture, screen_uv).r;
	// A cleared depth buffer is the far plane: 0 with reverse-Z, 1 without.
	if (raw_depth < 0.00001 || raw_depth > 0.99999) {
		return 1000.0;
	}
	vec4 view_pos = inv_projection * vec4(screen_uv * 2.0 - 1.0, raw_depth, 1.0);
	if (abs(view_pos.w) < 1e-6) {
		return 1000.0;
	}
	vec3 scene_view = view_pos.xyz / view_pos.w;
	return max(length(scene_view) - fragment_distance, 0.0);
}

// The three distance_fade_* thresholds as this view's height above the
// surface scales them: at the waterline detail stops at lod_surface_scale of
// the way out, and is fully restored lod_surface_height meters up. (A grazing
// view packs the far sea into few pixels -- the stretched fragments alias and
// overdraw, so a lower eye cuts them off sooner.) The height is passed in
// because a shader built-in cannot be read from a helper function.
vec3 lod_distances(float p_above) {
	float scale = mix(lod_surface_scale, 1.0, smoothstep(0.0, max(lod_surface_height, 0.001), p_above));
	float start = distance_fade_start * scale;
	float end = max(distance_fade_end * scale, start + 2.0);
	float mid = clamp(distance_fade_mid * scale, start + 1.0, end - 1.0);
	return vec3(start, mid, end);
}

// The three LOD layers for a point p_distance away: x is the mid layer's
// filter (fine detail and per-pixel normals gone at 1, from _start to _mid),
// y is the far layer's (procedural foam/shore/crest and displaced geometry
// gone at 1, from _mid to _end). Both are smooth, so the layers hand over
// without a band.
vec2 lod_filters(float p_distance, float p_above) {
	vec3 lod = lod_distances(p_above);
	return vec2(smoothstep(lod.x, lod.y, p_distance), smoothstep(lod.y, lod.z, p_distance));
}

// Fine detail normals from the scrolling normal map, applied in a world-space
// tangent frame (the FFT's own slopes supply the large-scale normal).
vec3 detail_normal_perturb(vec3 world_normal, vec2 world_xz, float mid_filter) {
	// LOD mid layer: past its end the map is sub-pixel (the wave normal is
	// flattened there anyway); past the halfway mark one layer is enough.
	if (shading_quality < 1 || mid_filter > 0.995) {
		return world_normal;
	}
	vec2 scroll = vec2(0.62, 0.78) * (water_time * detail_normal_speed);
	vec2 uv = world_xz * detail_normal_scale + scroll;
	vec3 nm = texture(detail_normal, uv).xyz * 2.0 - 1.0;
	if (shading_quality >= 2 && mid_filter < 0.5) {
		vec3 nm2 = texture(detail_normal, uv * 3.3 - scroll * 1.7 + vec2(0.31)).xyz * 2.0 - 1.0;
		nm.xy = (nm.xy + nm2.xy) * 0.5;
	}
	// Tangent frame from world X: the surface is mostly horizontal, so this
	// stays stable instead of spinning with the view.
	vec3 T = normalize(vec3(1.0, 0.0, 0.0) - world_normal * world_normal.x);
	vec3 B = normalize(cross(world_normal, T));
	float strength = detail_normal_strength * (shading_quality >= 2 ? 1.0 : 0.6);
	return normalize(world_normal + (T * nm.x + B * nm.y) * strength);
}

void vertex() {
	VERTEX += (inverse(MODEL_MATRIX) * vec4(mesh_offset.x, 0.0, mesh_offset.y, 0.0)).xyz;
	vec3 world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	v_rest_xz = world_pos.xz;
	// Vertex LOD: the displaced geometry flattens with the mid layer (fine
	// detail gone by _mid); keeping sub-pixel waves out there only aliases
	// the silhouette, and the mid distance is already cut right in when the
	// eye skims the water. The far layer finishes the remaining shading.
	// Computed here where the camera built-in is in the vertex stage.
	float surface_above = abs(CAMERA_POSITION_WORLD.y - (MODEL_MATRIX[3].y + water_level));
	vec2 lod = lod_filters(distance(CAMERA_POSITION_WORLD, world_pos), surface_above);
	float geom_fade = lod.x;
	VERTEX += surface_offset(world_pos.xz, geom_fade);
	v_world_normal = vec3(0.0, 1.0, 0.0);
	v_foam = 0.0;
	// Per pixel, fragment() computes the normal and foam instead.
	if (!per_pixel_normals) {
		// Per-vertex: normal from central differences of the displaced
		// surface -- e is in world meters, small against the grids' cell
		// size.
		const float e = 0.2;
		vec3 px1 = vec3(e, 0.0, 0.0) + surface_offset(world_pos.xz + vec2(e, 0.0), geom_fade);
		vec3 px0 = vec3(-e, 0.0, 0.0) + surface_offset(world_pos.xz - vec2(e, 0.0), geom_fade);
		vec3 pz1 = vec3(0.0, 0.0, e) + surface_offset(world_pos.xz + vec2(0.0, e), geom_fade);
		vec3 pz0 = vec3(0.0, 0.0, -e) + surface_offset(world_pos.xz - vec2(0.0, e), geom_fade);
		vec3 tx = px1 - px0;
		vec3 tz = pz1 - pz0;
		v_world_normal = normalize(cross(tz, tx));
		v_foam = foam_at(world_pos.xz, lod.x);
	}

#ifdef WATER_TEMPORAL
	// Water-only temporal AA: the renderer draws this material into its own
	// target and accumulates it, so each frame the surface is sampled a
	// fraction of a pixel away (Halton) and the history resolves the extra
	// samples.
	POSITION = PROJECTION_MATRIX * MODELVIEW_MATRIX * vec4(VERTEX, 1.0);
	POSITION.xy += temporal_jitter * POSITION.w;
#endif
}

void fragment() {
	// Draw only the side of the surface facing the camera's side of the
	// water: the top from above, the underside from below. The surface is
	// transparent and double-sided, and its triangles blend in mesh order,
	// not by distance -- so where it overlaps itself (the back of a crest
	// behind its front, far waves seen through near ones) the far side
	// painted over the near one as jagged translucent teeth, and the
	// pattern changed wherever the mesh order did, drawing a seam.
	bool camera_above = CAMERA_POSITION_WORLD.y > MODEL_MATRIX[3].y + water_level;
	if (FRONT_FACING != camera_above) {
		discard;
	}
	vec3 wp = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
	vec3 to_camera = normalize(CAMERA_POSITION_WORLD - wp);
	float surface_above = abs(CAMERA_POSITION_WORLD.y - (MODEL_MATRIX[3].y + water_level));
	vec2 lod = lod_filters(distance(CAMERA_POSITION_WORLD, wp), surface_above);
	float mid_filter = lod.x;
	float far_filter = lod.y;
	vec3 world_normal = v_world_normal;
	float foam = v_foam;
	// LOD mid layer: once the normal is fully flat and the foam has faded,
	// the per-pixel FFT/ripple/wake normal fetch (and its foam) is skipped
	// outright -- it is the widest part of a grazing view.
	if (per_pixel_normals && mid_filter < 0.995) {
		world_normal = surface_normal_pixel(v_rest_xz);
		foam = foam_at(v_rest_xz, mid_filter);
	}
	// Shading frame, facing the camera (from below the geometric normal
	// points away): every term below assumes the lit side.
	vec3 shade_normal = dot(world_normal, to_camera) < 0.0 ? -world_normal : world_normal;
	shade_normal = detail_normal_perturb(shade_normal, v_rest_xz, mid_filter);
	// Sub-pixel waves average out at distance; flatten instead of aliasing.
	shade_normal = normalize(mix(shade_normal, vec3(0.0, 1.0, 0.0), mid_filter));
	vec3 sun = normalize(sun_direction);

	if (!camera_above) {
		// Seen from below: Snell's window where the view is steep enough to
		// escape the water, total internal reflection beyond the critical
		// angle (water to air, 48.6 degrees). Approximated from the sky
		// gradient -- the surface under water shows the sky through the
		// window and the water body's own darkness outside it.
		float cos_i = clamp(dot(-to_camera, world_normal), 0.0, 1.0);
		float critical_cos = 0.6614;
		float window = smoothstep(critical_cos, critical_cos + 0.07, cos_i);
		vec3 reflected = sky_reflection(reflect(-to_camera, shade_normal), mid_filter);
		vec3 body = deep_color.rgb * (1.0 - 0.35 * window);
		ALBEDO = mix(body, reflected, window);
		ALPHA = 1.0;
	} else {

	// unshaded + small self-contained, bounded shading terms -- NOT Godot's
	// real diffuse/specular PBR response. Real root cause of the "whitewash":
	// this material used render_mode diffuse_burley, specular_schlick_ggx, so
	// it got its own full sun+ambient PBR light response computed as if fully
	// opaque, and THAT result was then alpha-blended on top of the floor's OWN
	// independently-lit PBR response. Alpha blending two independently, fully
	// lit PBR surfaces in linear HDR sums MORE light than either surface alone
	// before the single tonemap pass sees it. Going unshaded removes the water
	// from that PBR light budget: everything below is a bounded tint on top of
	// the scene, never a second light response.
	vec3 n = normalize((VIEW_MATRIX * vec4(shade_normal, 0.0)).xyz);
	vec3 view_dir = normalize(-VERTEX);
	vec3 sun_view = normalize((VIEW_MATRIX * vec4(sun, 0.0)).xyz);
	vec3 half_dir = normalize(sun_view + view_dir);
	float ndotl = clamp(dot(n, vec3(0.3, 0.85, 0.4)), 0.0, 1.0);
	float sun_up = clamp(dot(shade_normal, sun), 0.0, 1.0);
	float shore = shore_signal(v_rest_xz, water_time, mid_filter);
	// Shore foam belongs to the moving edge: full at it, thinning over the
	// next ~0.6 m of depth, so the white sheet scallops with the edge instead
	// of filling the band. (Whitecaps are left alone.)
	foam = max(foam, shore_foam_at(v_rest_xz, mid_filter) * (1.0 - smoothstep(0.02, 0.6, shore)));
	// Bright rim right behind the moving edge, broken into lace.
	// Over dry sand the swash film is only as thick as the run-up's excess
	// over the sand -- a few cm -- so it fades in over a fifth of the
	// softness there, or the uprush would barely show.
	vec2 cuv_here = clamp((v_rest_xz - grid_center) / ripple_domain_size + 0.5, vec2(0.0), vec2(1.0));
	float soft = max(shore_edge_softness, 1e-3) * mix(0.2, 1.0, smoothstep(-0.05, 0.1, depth_r(v_rest_xz, cuv_here)));
	soft *= mix(1.0, 2.5, mid_filter); // a distant waterline covers several pixels
	float rim = 0.0;
	if (mid_filter < 0.995) {
		// LOD mid layer: a distant waterline is thinner than a pixel, so the
		// rim and its lace only shimmer. Fade it before the cutoff.
		rim = smoothstep(0.0, soft * 0.5, shore) * (1.0 - smoothstep(soft * 0.6, soft * 2.2, shore));
		rim *= shore_rim_strength * (0.55 + 0.45 * foam_value_noise(v_rest_xz * 6.0 * shore_detail_scale)) * (1.0 - mid_filter);
	}
	foam = max(foam, rim);
	vec2 cuv_fade = clamp((v_rest_xz - grid_center) / ripple_domain_size + 0.5, vec2(0.0), vec2(1.0));
	float shallow = (1.0 - fade_r(v_rest_xz, cuv_fade)) * shallow_tint;
	vec3 base_rgb = mix(water_color.rgb, shallow_color.rgb, shallow) * mix(0.75, 1.15, ndotl);
	float base_a = mix(water_color.a, shallow_color.a, shallow);
	// Thin water is clear: the swash film and the last few cm at the edge let
	// the sand show through, turning opaque by ~25 cm of depth.
	base_a *= mix(0.4, 1.0, smoothstep(0.0, 0.25, shore));

	// How much water sits along this fragment's view ray: thick water keeps
	// the deep color and turns opaque, thin water stays clear.
	float thickness = view_thickness(SCREEN_UV, INV_PROJECTION_MATRIX, length(VERTEX));
	float absorb = 1.0 - exp(-absorption * thickness);
	base_rgb = mix(base_rgb, deep_color.rgb, absorb * 0.55);
	base_a = clamp(base_a + absorb * 0.30, 0.0, 1.0);

	// Sky reflection at grazing angles (Fresnel) and the sun's glitter.
	vec3 reflected = sky_reflection(reflect(-to_camera, shade_normal), mid_filter);
	float fresnel = 0.02 + 0.98 * pow(1.0 - clamp(dot(shade_normal, to_camera), 0.0, 1.0), 5.0);
	float reflection_amt = fresnel * reflection_strength;
	vec3 specular = vec3(0.0);
	if (shading_quality >= 1) {
		// The mid LOD layer widens the glint to the far layer's width; the
		// far layer then dims what is left. Combined with the geometric AA
		// below, a sub-pixel highlight ends as a soft, faint band.
		float dist_shininess = mix(specular_shininess, 90.0, mid_filter);
		// Geometric specular AA: the pixel's normal varies, so it sees a
		// spread of normals, not one. Variances add on the Blinn-Phong lobe
		// (alpha^2 = 2 / (shininess + 2)), so widen the exponent by the
		// pixel's normal variance instead of sampling a point that sparkles
		// as the waves move under it. Derivatives are taken from the
		// view-space normal actually used for the glint.
		float normal_var = 0.25 * (dot(dFdx(n), dFdx(n)) + dot(dFdy(n), dFdy(n)));
		float filtered_shininess = 2.0 / (2.0 / (dist_shininess + 2.0) + normal_var * specular_aa) - 2.0;
		float shininess = max(min(dist_shininess, filtered_shininess), 1.0);
		// A wider lobe peaks lower: scale by the normalized lobe's peak so the
		// widening spreads the highlight rather than brightening more pixels.
		float glint_fade = (1.0 - 0.7 * mid_filter) * (1.0 - 0.5 * far_filter) * min(1.0, (shininess + 2.0) / (dist_shininess + 2.0));
		float spec = pow(clamp(dot(n, half_dir), 0.0, 1.0), shininess);
		spec = clamp(spec * specular_strength * (0.3 + 0.7 * sun_up) * min(sun_energy, 2.0) * 2.0 * glint_fade, 0.0, 2.0);
		specular = sun_color * spec;
	}
	base_rgb = mix(base_rgb, reflected, reflection_amt) + specular;
	base_a = clamp(base_a + reflection_amt * 0.6, 0.0, 1.0);
	if (shading_quality >= 2 && mid_filter < 0.995) {
		// Sun through a backlit crest: a dark green glow, bounded to a tint.
		// LOD: the crest is sub-pixel far away; fade it out before the cutoff
		// so the switch is invisible.
		float crest = texture(ocean_height_tex, v_rest_xz / ocean_domain_size + 0.5).r;
		float crest_up = clamp(crest / max(wave_amplitude, 0.05), 0.0, 1.0);
		float back = pow(clamp(dot(-to_camera, sun) * 0.5 + 0.5, 0.0, 1.0), 2.0);
		base_rgb += crest_sss_color * (crest_up * crest_up * back * crest_sss) * (1.0 - mid_filter);
	}

#ifdef WATER_SCREEN_SPACE
	// Refraction: where the water is fully covered (not the fading shore
	// film), sample the opaque scene with a distortion offset and composite
	// at alpha 1, so the background is replaced rather than counted twice.
	float covered = smoothstep(0.0, max(soft_edge_distance, 1e-3), thickness);
	float shore_cover = smoothstep(0.25, 0.75, depth_r(v_rest_xz, cuv_fade));
	float refr = covered * shore_cover * refraction_mix;
	if (refr > 0.0) {
		vec2 offset = shade_normal.xz * refraction_strength * (0.35 + 0.65 * absorb);
		vec2 ruv = clamp(SCREEN_UV + offset, vec2(0.001), vec2(0.999));
		vec3 behind = texture(screen_texture, ruv).rgb;
		vec3 tint = exp(-thickness * vec3(absorption * 1.6, absorption, absorption * 0.7));
		vec3 refracted = behind * tint + deep_color.rgb * (1.0 - exp(-absorption * thickness));
		base_rgb = mix(base_rgb, refracted, refr);
		base_a = mix(base_a, 1.0, refr);
	}
#endif

	ALBEDO = mix(base_rgb, foam_color.rgb, foam);
	float water_a = mix(base_a, foam_color.a, foam) * smoothstep(0.0, soft, shore);
	// Soft contact with opaque geometry: fade out where the water plane
	// grazes it. Skipped over the shallow shore, which already has its own
	// soft edge and swash film (a depth-based fade would erase the film).
	float depth_soft = smoothstep(0.0, max(soft_edge_distance, 1e-3), thickness);
	float soft_mask = smoothstep(0.25, 0.75, depth_r(v_rest_xz, cuv_fade));
	water_a *= mix(1.0, depth_soft, soft_mask);
	// Wet sand under and behind the swash: blended in where the water is
	// thin or gone, so it shows through the film and lingers after it.
	// Noise eats into the wet edge, so it dries back ragged rather than
	// along a clean line.
	vec2 wp = v_rest_xz * shore_detail_scale;
	float wn = 0.6 * foam_value_noise(wp * 1.7) + 0.4 * foam_value_noise(wp * 5.3 + 7.0);
	float wet = clamp((texture(swash_tex, shore_pattern_uv(v_rest_xz)).g - 0.35 * wn) / 0.65, 0.0, 1.0);
	// Only where sand is near the surface: under deeper water the "wet" flag
	// is meaningless, and blending it there tinted and roughened all deep
	// water -- which showed as a seam past the simulated square, where the
	// mirrored patterns bring dry beach (not wet) out over the open sea.
	wet *= 1.0 - smoothstep(0.05, 0.35, depth_r(v_rest_xz, cuv_fade));
	float wet_a = wet * wet * (3.0 - 2.0 * wet) * wet_sand_color.a * (1.0 - water_a);
	ALPHA = water_a + wet_a;
	ALBEDO = (ALBEDO * water_a + wet_sand_color.rgb * wet_a) / max(ALPHA, 1e-4);
	}
}
)";

// Sign of the X/Z cross product of an up-facing PlaneMesh triangle -- measured
// from PlaneMesh itself rather than assumed from a winding convention.
real_t plane_mesh_xz_winding() {
	Ref<PlaneMesh> ref_plane;
	ref_plane.instantiate();
	const Vector<Face3> faces = ref_plane->get_faces();
	if (faces.is_empty()) {
		return 1.0;
	}
	const Face3 &f = faces[0];
	const Vector2 a(f.vertex[0].x, f.vertex[0].z), b(f.vertex[1].x, f.vertex[1].z), c(f.vertex[2].x, f.vertex[2].z);
	return (b - a).cross(c - a) < 0 ? -1.0 : 1.0;
}

// Inside or on the edge of triangle abc, either winding.
bool point_in_triangle(const Vector2 &p_p, const Vector2 &p_a, const Vector2 &p_b, const Vector2 &p_c) {
	const real_t d1 = (p_b - p_a).cross(p_p - p_a);
	const real_t d2 = (p_c - p_b).cross(p_p - p_b);
	const real_t d3 = (p_a - p_c).cross(p_p - p_c);
	const bool has_neg = d1 < 0 || d2 < 0 || d3 < 0;
	const bool has_pos = d1 > 0 || d2 > 0 || d3 > 0;
	return !(has_neg && has_pos);
}

// Liang-Barsky: does segment p0-p1 touch the box [p_min, p_max]?
bool segment_touches_box(const Vector2 &p_p0, const Vector2 &p_p1, const Vector2 &p_min, const Vector2 &p_max) {
	real_t t0 = 0.0, t1 = 1.0;
	const Vector2 d = p_p1 - p_p0;
	for (int axis = 0; axis < 2; axis++) {
		const real_t q0 = p_min[axis] - p_p0[axis];
		const real_t q1 = p_max[axis] - p_p0[axis];
		if (Math::abs(d[axis]) < (real_t)1e-12) {
			if (q0 > 0 || q1 < 0) {
				return false;
			}
			continue;
		}
		real_t ta = q0 / d[axis];
		real_t tb = q1 / d[axis];
		if (ta > tb) {
			SWAP(ta, tb);
		}
		t0 = MAX(t0, ta);
		t1 = MIN(t1, tb);
		if (t0 > t1) {
			return false;
		}
	}
	return true;
}

// Sutherland-Hodgman: clip a convex polygon to a counter-clockwise triangle.
void clip_to_triangle(LocalVector<Vector2> &r_poly, const Vector2 p_tri[3]) {
	LocalVector<Vector2> out;
	for (int e = 0; e < 3 && r_poly.size() >= 3; e++) {
		const Vector2 a = p_tri[e];
		const Vector2 edge = p_tri[(e + 1) % 3] - a;
		out.clear();
		for (uint32_t i = 0; i < r_poly.size(); i++) {
			const Vector2 cur = r_poly[i];
			const Vector2 prev = r_poly[(i + r_poly.size() - 1) % r_poly.size()];
			const real_t side_cur = edge.cross(cur - a);
			const real_t side_prev = edge.cross(prev - a);
			if (side_cur >= 0) {
				if (side_prev < 0) {
					out.push_back(prev + (cur - prev) * (side_prev / (side_prev - side_cur)));
				}
				out.push_back(cur);
			} else if (side_prev >= 0) {
				out.push_back(prev + (cur - prev) * (side_prev / (side_prev - side_cur)));
			}
		}
		r_poly = out;
	}
}

} // namespace

void WaterBody3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_domain_size", "size"), &WaterBody3D::set_domain_size);
	ClassDB::bind_method(D_METHOD("get_domain_size"), &WaterBody3D::get_domain_size);
	ClassDB::bind_method(D_METHOD("set_grid_resolution", "n"), &WaterBody3D::set_grid_resolution);
	ClassDB::bind_method(D_METHOD("get_grid_resolution"), &WaterBody3D::get_grid_resolution);
	ClassDB::bind_method(D_METHOD("set_depth", "depth"), &WaterBody3D::set_depth);
	ClassDB::bind_method(D_METHOD("get_depth"), &WaterBody3D::get_depth);
	ClassDB::bind_method(D_METHOD("set_water_level", "level"), &WaterBody3D::set_water_level);
	ClassDB::bind_method(D_METHOD("get_water_level"), &WaterBody3D::get_water_level);
	ClassDB::bind_method(D_METHOD("set_damping", "damping"), &WaterBody3D::set_damping);
	ClassDB::bind_method(D_METHOD("get_damping"), &WaterBody3D::get_damping);
	ClassDB::bind_method(D_METHOD("set_ripple_amplitude", "amplitude"), &WaterBody3D::set_ripple_amplitude);
	ClassDB::bind_method(D_METHOD("get_ripple_amplitude"), &WaterBody3D::get_ripple_amplitude);

	ClassDB::bind_method(D_METHOD("set_ocean_grid_resolution", "n"), &WaterBody3D::set_ocean_grid_resolution);
	ClassDB::bind_method(D_METHOD("get_ocean_grid_resolution"), &WaterBody3D::get_ocean_grid_resolution);
	ClassDB::bind_method(D_METHOD("set_ocean_domain_size", "size"), &WaterBody3D::set_ocean_domain_size);
	ClassDB::bind_method(D_METHOD("get_ocean_domain_size"), &WaterBody3D::get_ocean_domain_size);
	ClassDB::bind_method(D_METHOD("set_wind_speed", "speed"), &WaterBody3D::set_wind_speed);
	ClassDB::bind_method(D_METHOD("get_wind_speed"), &WaterBody3D::get_wind_speed);
	ClassDB::bind_method(D_METHOD("set_wind_direction", "dir"), &WaterBody3D::set_wind_direction);
	ClassDB::bind_method(D_METHOD("get_wind_direction"), &WaterBody3D::get_wind_direction);
	ClassDB::bind_method(D_METHOD("set_wave_amplitude", "amplitude"), &WaterBody3D::set_wave_amplitude);
	ClassDB::bind_method(D_METHOD("get_wave_amplitude"), &WaterBody3D::get_wave_amplitude);
	ClassDB::bind_method(D_METHOD("set_fetch", "fetch"), &WaterBody3D::set_fetch);
	ClassDB::bind_method(D_METHOD("get_fetch"), &WaterBody3D::get_fetch);
	ClassDB::bind_method(D_METHOD("get_effective_fetch"), &WaterBody3D::get_effective_fetch);
	ClassDB::bind_method(D_METHOD("set_choppiness", "choppiness"), &WaterBody3D::set_choppiness);
	ClassDB::bind_method(D_METHOD("get_choppiness"), &WaterBody3D::get_choppiness);
	ClassDB::bind_method(D_METHOD("set_normal_mode", "mode"), &WaterBody3D::set_normal_mode);
	ClassDB::bind_method(D_METHOD("get_normal_mode"), &WaterBody3D::get_normal_mode);
	ClassDB::bind_method(D_METHOD("set_foam_enabled", "enabled"), &WaterBody3D::set_foam_enabled);
	ClassDB::bind_method(D_METHOD("get_foam_enabled"), &WaterBody3D::get_foam_enabled);
	ClassDB::bind_method(D_METHOD("set_foam_threshold", "threshold"), &WaterBody3D::set_foam_threshold);
	ClassDB::bind_method(D_METHOD("get_foam_threshold"), &WaterBody3D::get_foam_threshold);
	ClassDB::bind_method(D_METHOD("set_foam_persistence", "seconds"), &WaterBody3D::set_foam_persistence);
	ClassDB::bind_method(D_METHOD("get_foam_persistence"), &WaterBody3D::get_foam_persistence);
	ClassDB::bind_method(D_METHOD("set_shore_foam_band", "depth"), &WaterBody3D::set_shore_foam_band);
	ClassDB::bind_method(D_METHOD("get_shore_foam_band"), &WaterBody3D::get_shore_foam_band);
	ClassDB::bind_method(D_METHOD("set_open_sea_beyond", "open"), &WaterBody3D::set_open_sea_beyond);
	ClassDB::bind_method(D_METHOD("set_sea_state", "wind_speed", "wind_direction", "wave_amplitude", "fetch"), &WaterBody3D::set_sea_state);
	ClassDB::bind_method(D_METHOD("recenter", "world_position"), &WaterBody3D::recenter);
	ClassDB::bind_method(D_METHOD("get_open_sea_beyond"), &WaterBody3D::get_open_sea_beyond);
	ClassDB::bind_method(D_METHOD("set_shore_undertow", "speed"), &WaterBody3D::set_shore_undertow);
	ClassDB::bind_method(D_METHOD("get_shore_undertow"), &WaterBody3D::get_shore_undertow);
	ClassDB::bind_method(D_METHOD("set_swash_run_up", "height"), &WaterBody3D::set_swash_run_up);
	ClassDB::bind_method(D_METHOD("get_swash_run_up"), &WaterBody3D::get_swash_run_up);
	ClassDB::bind_method(D_METHOD("set_swash_drain_speed", "speed"), &WaterBody3D::set_swash_drain_speed);
	ClassDB::bind_method(D_METHOD("get_swash_drain_speed"), &WaterBody3D::get_swash_drain_speed);
	ClassDB::bind_method(D_METHOD("set_wet_sand_dry_time", "seconds"), &WaterBody3D::set_wet_sand_dry_time);
	ClassDB::bind_method(D_METHOD("get_wet_sand_dry_time"), &WaterBody3D::get_wet_sand_dry_time);

	ClassDB::bind_method(D_METHOD("set_surface_mesh", "mesh"), &WaterBody3D::set_surface_mesh);
	ClassDB::bind_method(D_METHOD("get_surface_mesh"), &WaterBody3D::get_surface_mesh);
	ClassDB::bind_method(D_METHOD("set_seabed_from_floor", "enabled"), &WaterBody3D::set_seabed_from_floor);
	ClassDB::bind_method(D_METHOD("get_seabed_from_floor"), &WaterBody3D::get_seabed_from_floor);
	ClassDB::bind_method(D_METHOD("set_seabed_collision_mask", "mask"), &WaterBody3D::set_seabed_collision_mask);
	ClassDB::bind_method(D_METHOD("get_seabed_collision_mask"), &WaterBody3D::get_seabed_collision_mask);
	ClassDB::bind_method(D_METHOD("set_shallow_fade_depth", "depth"), &WaterBody3D::set_shallow_fade_depth);
	ClassDB::bind_method(D_METHOD("get_shallow_fade_depth"), &WaterBody3D::get_shallow_fade_depth);
	ClassDB::bind_method(D_METHOD("set_render_extent", "extent"), &WaterBody3D::set_render_extent);
	ClassDB::bind_method(D_METHOD("get_render_extent"), &WaterBody3D::get_render_extent);
	ClassDB::bind_method(D_METHOD("set_water_material", "material"), &WaterBody3D::set_water_material);
	ClassDB::bind_method(D_METHOD("get_water_material"), &WaterBody3D::get_water_material);
	ClassDB::bind_method(D_METHOD("set_quality", "quality"), &WaterBody3D::set_quality);
	ClassDB::bind_method(D_METHOD("get_quality"), &WaterBody3D::get_quality);
	ClassDB::bind_method(D_METHOD("set_lod_quality", "quality"), &WaterBody3D::set_lod_quality);
	ClassDB::bind_method(D_METHOD("get_lod_quality"), &WaterBody3D::get_lod_quality);
	ClassDB::bind_method(D_METHOD("set_lod_distance_start", "distance"), &WaterBody3D::set_lod_distance_start);
	ClassDB::bind_method(D_METHOD("get_lod_distance_start"), &WaterBody3D::get_lod_distance_start);
	ClassDB::bind_method(D_METHOD("set_lod_distance_mid", "distance"), &WaterBody3D::set_lod_distance_mid);
	ClassDB::bind_method(D_METHOD("get_lod_distance_mid"), &WaterBody3D::get_lod_distance_mid);
	ClassDB::bind_method(D_METHOD("set_lod_distance_end", "distance"), &WaterBody3D::set_lod_distance_end);
	ClassDB::bind_method(D_METHOD("get_lod_distance_end"), &WaterBody3D::get_lod_distance_end);
	ClassDB::bind_method(D_METHOD("set_lod_surface_scale", "scale"), &WaterBody3D::set_lod_surface_scale);
	ClassDB::bind_method(D_METHOD("get_lod_surface_scale"), &WaterBody3D::get_lod_surface_scale);
	ClassDB::bind_method(D_METHOD("set_lod_surface_height", "height"), &WaterBody3D::set_lod_surface_height);
	ClassDB::bind_method(D_METHOD("get_lod_surface_height"), &WaterBody3D::get_lod_surface_height);
	ClassDB::bind_method(D_METHOD("set_lod_mesh_growth", "growth"), &WaterBody3D::set_lod_mesh_growth);
	ClassDB::bind_method(D_METHOD("get_lod_mesh_growth"), &WaterBody3D::get_lod_mesh_growth);
	ClassDB::bind_method(D_METHOD("set_sun_path", "path"), &WaterBody3D::set_sun_path);
	ClassDB::bind_method(D_METHOD("get_sun_path"), &WaterBody3D::get_sun_path);
	ClassDB::bind_method(D_METHOD("set_screen_space", "enabled"), &WaterBody3D::set_screen_space);
	ClassDB::bind_method(D_METHOD("get_screen_space"), &WaterBody3D::get_screen_space);
	ClassDB::bind_method(D_METHOD("set_temporal_aa", "enabled"), &WaterBody3D::set_temporal_aa);
	ClassDB::bind_method(D_METHOD("get_temporal_aa"), &WaterBody3D::get_temporal_aa);
	ClassDB::bind_method(D_METHOD("set_auto_recenter", "enabled"), &WaterBody3D::set_auto_recenter);
	ClassDB::bind_method(D_METHOD("get_auto_recenter"), &WaterBody3D::get_auto_recenter);
	ClassDB::bind_method(D_METHOD("set_underwater_effect", "enabled"), &WaterBody3D::set_underwater_effect);
	ClassDB::bind_method(D_METHOD("get_underwater_effect"), &WaterBody3D::get_underwater_effect);

	ClassDB::bind_method(D_METHOD("set_caustics_enabled", "enabled"), &WaterBody3D::set_caustics_enabled);
	ClassDB::bind_method(D_METHOD("get_caustics_enabled"), &WaterBody3D::get_caustics_enabled);
	ClassDB::bind_method(D_METHOD("set_caustics_sun_direction", "direction"), &WaterBody3D::set_caustics_sun_direction);
	ClassDB::bind_method(D_METHOD("get_caustics_sun_direction"), &WaterBody3D::get_caustics_sun_direction);
	ClassDB::bind_method(D_METHOD("set_caustics_reference_depth", "depth"), &WaterBody3D::set_caustics_reference_depth);
	ClassDB::bind_method(D_METHOD("get_caustics_reference_depth"), &WaterBody3D::get_caustics_reference_depth);
	ClassDB::bind_method(D_METHOD("get_caustics_texture"), &WaterBody3D::get_caustics_texture);
	ClassDB::bind_method(D_METHOD("get_caustics_light_right"), &WaterBody3D::get_caustics_light_right);
	ClassDB::bind_method(D_METHOD("get_caustics_light_up"), &WaterBody3D::get_caustics_light_up);
	ClassDB::bind_method(D_METHOD("get_caustics_origin"), &WaterBody3D::get_caustics_origin);
	ClassDB::bind_method(D_METHOD("get_caustics_half_extent"), &WaterBody3D::get_caustics_half_extent);
	ClassDB::bind_method(D_METHOD("get_caustics_tile_size"), &WaterBody3D::get_caustics_tile_size);

	ClassDB::bind_method(D_METHOD("sample_height", "world_pos"), &WaterBody3D::sample_height);
	ClassDB::bind_method(D_METHOD("sample_height_excluding_wake", "world_pos", "wake"), &WaterBody3D::_sample_height_excluding_wake_bind);
	ClassDB::bind_method(D_METHOD("is_wet", "world_pos"), &WaterBody3D::is_wet);
	ClassDB::bind_method(D_METHOD("submit_sphere", "owner", "world_pos", "radius", "strength"), &WaterBody3D::submit_sphere, DEFVAL(1.0f));
	ClassDB::bind_method(D_METHOD("clear_sphere", "owner"), &WaterBody3D::clear_sphere);
	ClassDB::bind_method(D_METHOD("submit_impulse", "world_pos", "radius", "strength"), &WaterBody3D::submit_impulse);

	BIND_ENUM_CONSTANT(NORMAL_MODE_PER_PIXEL);
	BIND_ENUM_CONSTANT(NORMAL_MODE_PER_VERTEX);
	BIND_ENUM_CONSTANT(QUALITY_LOW);
	BIND_ENUM_CONSTANT(QUALITY_MEDIUM);
	BIND_ENUM_CONSTANT(QUALITY_HIGH);
	BIND_ENUM_CONSTANT(QUALITY_CUSTOM);
	BIND_ENUM_CONSTANT(LOD_QUALITY_ULTRA);
	BIND_ENUM_CONSTANT(LOD_QUALITY_HIGH);
	BIND_ENUM_CONSTANT(LOD_QUALITY_MEDIUM);
	BIND_ENUM_CONSTANT(LOD_QUALITY_LOW);
	BIND_ENUM_CONSTANT(LOD_QUALITY_CUSTOM);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "surface_mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_surface_mesh", "get_surface_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "domain_size"), "set_domain_size", "get_domain_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "grid_resolution"), "set_grid_resolution", "get_grid_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "depth"), "set_depth", "get_depth");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "water_level"), "set_water_level", "get_water_level");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping"), "set_damping", "get_damping");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ripple_amplitude", PROPERTY_HINT_RANGE, "0,2,0.01,or_greater"), "set_ripple_amplitude", "get_ripple_amplitude");

	ADD_GROUP("Seabed", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "seabed_from_floor"), "set_seabed_from_floor", "get_seabed_from_floor");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "seabed_collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_seabed_collision_mask", "get_seabed_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "shallow_fade_depth", PROPERTY_HINT_RANGE, "0.01,10,0.01,or_greater,suffix:m"), "set_shallow_fade_depth", "get_shallow_fade_depth");

	ADD_GROUP("Ocean", "ocean_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ocean_grid_resolution"), "set_ocean_grid_resolution", "get_ocean_grid_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "ocean_domain_size"), "set_ocean_domain_size", "get_ocean_domain_size");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "wind_speed"), "set_wind_speed", "get_wind_speed");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "wind_direction"), "set_wind_direction", "get_wind_direction");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "wave_amplitude"), "set_wave_amplitude", "get_wave_amplitude");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fetch", PROPERTY_HINT_RANGE, "0,100000,1,or_greater,suffix:m"), "set_fetch", "get_fetch");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "choppiness", PROPERTY_HINT_RANGE, "0,3,0.01,or_greater"), "set_choppiness", "get_choppiness");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "foam_enabled"), "set_foam_enabled", "get_foam_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "foam_threshold", PROPERTY_HINT_RANGE, "0,1.5,0.01"), "set_foam_threshold", "get_foam_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "shore_foam_band", PROPERTY_HINT_RANGE, "0,5,0.05,or_greater,suffix:m"), "set_shore_foam_band", "get_shore_foam_band");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "open_sea_beyond"), "set_open_sea_beyond", "get_open_sea_beyond");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "shore_undertow", PROPERTY_HINT_RANGE, "0,1.5,0.01,or_greater,suffix:m/s"), "set_shore_undertow", "get_shore_undertow");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "foam_persistence", PROPERTY_HINT_RANGE, "0.05,20,0.05,or_greater,suffix:s"), "set_foam_persistence", "get_foam_persistence");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "swash_run_up", PROPERTY_HINT_RANGE, "0,1,0.01,or_greater,suffix:m"), "set_swash_run_up", "get_swash_run_up");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "swash_drain_speed", PROPERTY_HINT_RANGE, "0.01,1,0.01,or_greater,suffix:m/s"), "set_swash_drain_speed", "get_swash_drain_speed");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "wet_sand_dry_time", PROPERTY_HINT_RANGE, "0.1,60,0.1,or_greater,suffix:s"), "set_wet_sand_dry_time", "get_wet_sand_dry_time");

	ADD_GROUP("Rendering", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "render_extent", PROPERTY_HINT_RANGE, "0,2000,1,or_greater,suffix:m"), "set_render_extent", "get_render_extent");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "normal_mode", PROPERTY_HINT_ENUM, "Per Pixel,Per Vertex"), "set_normal_mode", "get_normal_mode");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "water_material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial"), "set_water_material", "get_water_material");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "quality", PROPERTY_HINT_ENUM, "Low,Medium,High,Custom"), "set_quality", "get_quality");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "screen_space"), "set_screen_space", "get_screen_space");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "temporal_aa"), "set_temporal_aa", "get_temporal_aa");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_recenter"), "set_auto_recenter", "get_auto_recenter");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "underwater_effect"), "set_underwater_effect", "get_underwater_effect");

	ADD_GROUP("LOD", "lod_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "lod_quality", PROPERTY_HINT_ENUM, "Ultra,High,Medium,Low,Custom"), "set_lod_quality", "get_lod_quality");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_distance_start", PROPERTY_HINT_RANGE, "0,5000,1,or_greater,suffix:m"), "set_lod_distance_start", "get_lod_distance_start");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_distance_mid", PROPERTY_HINT_RANGE, "0,5000,1,or_greater,suffix:m"), "set_lod_distance_mid", "get_lod_distance_mid");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_distance_end", PROPERTY_HINT_RANGE, "0,5000,1,or_greater,suffix:m"), "set_lod_distance_end", "get_lod_distance_end");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_surface_scale", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_lod_surface_scale", "get_lod_surface_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_surface_height", PROPERTY_HINT_RANGE, "0,1000,1,or_greater,suffix:m"), "set_lod_surface_height", "get_lod_surface_height");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_mesh_growth", PROPERTY_HINT_RANGE, "1.01,2,0.01"), "set_lod_mesh_growth", "get_lod_mesh_growth");

	ADD_GROUP("Sun", "");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "sun_path", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "DirectionalLight3D"), "set_sun_path", "get_sun_path");

	ADD_GROUP("Caustics", "caustic");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "caustics_enabled"), "set_caustics_enabled", "get_caustics_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "caustics_sun_direction"), "set_caustics_sun_direction", "get_caustics_sun_direction");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "caustics_reference_depth", PROPERTY_HINT_RANGE, "0.1,50,0.1,or_greater"), "set_caustics_reference_depth", "get_caustics_reference_depth");
}

void WaterBody3D::set_domain_size(Vector2 p_size) {
	domain_size = p_size;
	_rebuild();
}

void WaterBody3D::set_grid_resolution(int p_n) {
	grid_resolution = MAX(p_n, 4);
	_rebuild();
}

void WaterBody3D::set_depth(float p_depth) {
	depth = MAX(p_depth, 0.1f);
	_rebuild();
}

void WaterBody3D::set_water_level(float p_level) {
	water_level = p_level;
	_rebuild();
}

void WaterBody3D::set_damping(float p_damping) {
	damping = p_damping;
	_rebuild();
}

void WaterBody3D::set_ripple_amplitude(float p_amplitude) {
	ripple_amplitude = MAX(p_amplitude, 0.0f);
	_rebuild();
}

void WaterBody3D::set_ocean_grid_resolution(int p_n) {
	// Must stay a power of two -- the Stockham FFT passes require it (see
	// water_solver.h). Round up rather than silently misbehave.
	int n = MAX(p_n, 4);
	int pow2 = 4;
	while (pow2 < n) {
		pow2 <<= 1;
	}
	ocean_grid_resolution = pow2;
	_rebuild();
}

void WaterBody3D::set_ocean_domain_size(Vector2 p_size) {
	ocean_domain_size = p_size;
	_rebuild();
}

void WaterBody3D::set_wind_speed(float p_speed) {
	wind_speed = MAX(p_speed, 0.01f);
	_reinit_spectrum();
}

void WaterBody3D::set_wind_direction(Vector2 p_dir) {
	wind_direction = p_dir;
	_reinit_spectrum();
}

void WaterBody3D::set_wave_amplitude(float p_amp) {
	wave_amplitude = MAX(p_amp, 0.0f);
	if (water_material.is_valid()) {
		water_material->set_shader_parameter("wave_amplitude", wave_amplitude);
	}
	_reinit_spectrum();
}

void WaterBody3D::set_sea_state(float p_wind_speed, Vector2 p_wind_direction, float p_wave_amplitude, float p_fetch) {
	wind_speed = MAX(p_wind_speed, 0.01f);
	wind_direction = p_wind_direction;
	wave_amplitude = MAX(p_wave_amplitude, 0.0f);
	fetch = MAX(p_fetch, 0.0f);
	_reinit_spectrum();
}

// Move the ripple square to `p_world_position` (its XZ; the node's height is kept) WITHOUT a rebuild: the mesh, the FFT
// ocean and every texture stay bound (a rebuild left the sea undrawn for a frame -- a flash each time a camera-following
// square re-centred); the ripples are cleared. A footprint / per-cell seabed depends on where the grid is, so those
// still rebuild.
void WaterBody3D::recenter(Vector3 p_world_position) {
	Vector3 p = get_global_position();
	p.x = p_world_position.x;
	p.z = p_world_position.z;
	if (!solver.is_available() || Engine::get_singleton()->is_editor_hint() || surface_mesh.is_valid() || seabed_from_floor || !cell_depth.is_empty()) {
		set_global_position(p);
		_rebuild();
		return;
	}
	set_global_position(p);
	grid_center = Vector2(p.x, p.z);
	if (water_material.is_valid()) {
		water_material->set_shader_parameter("grid_center", grid_center);
	}
	solver.recenter(grid_center);
	cached_ripple_height.clear(); // (the old square's heights are not this one's)
	frames_since_refresh = REFRESH_EVERY_FRAMES;
}

// The sea state changed: a built solver regenerates its spectrum IN PLACE (no rebuild -- the waves keep their phases,
// ripples / wakes / foam survive: storms can ramp it); not built yet → the usual rebuild.
void WaterBody3D::_reinit_spectrum() {
	if (solver.is_available()) {
		solver.reinit_spectrum(wind_speed, wind_direction, wave_amplitude, get_effective_fetch());
		return;
	}
	_rebuild();
}

void WaterBody3D::set_choppiness(float p_choppiness) {
	choppiness = MAX(p_choppiness, 0.0f);
	if (water_material.is_valid()) {
		water_material->set_shader_parameter("choppiness", choppiness);
	}
	solver.set_foam_settings(foam_enabled, choppiness, foam_threshold, foam_persistence, shore_foam_band, shore_undertow);
}

void WaterBody3D::set_foam_enabled(bool p_enabled) {
	foam_enabled = p_enabled;
	solver.set_foam_settings(foam_enabled, choppiness, foam_threshold, foam_persistence, shore_foam_band, shore_undertow);
}

void WaterBody3D::set_shore_undertow(float p_speed) {
	shore_undertow = MAX(p_speed, 0.0f);
	solver.set_foam_settings(foam_enabled, choppiness, foam_threshold, foam_persistence, shore_foam_band, shore_undertow);
}

void WaterBody3D::set_swash_run_up(float p_height) {
	swash_run_up = MAX(p_height, 0.0f);
	solver.set_swash_settings(swash_run_up, swash_drain_speed, wet_sand_dry_time);
}

void WaterBody3D::set_swash_drain_speed(float p_speed) {
	swash_drain_speed = MAX(p_speed, 0.01f);
	solver.set_swash_settings(swash_run_up, swash_drain_speed, wet_sand_dry_time);
}

void WaterBody3D::set_wet_sand_dry_time(float p_seconds) {
	wet_sand_dry_time = MAX(p_seconds, 0.1f);
	solver.set_swash_settings(swash_run_up, swash_drain_speed, wet_sand_dry_time);
}

void WaterBody3D::set_open_sea_beyond(bool p_open) {
	open_sea_beyond = p_open;
	if (water_material.is_valid()) {
		water_material->set_shader_parameter("open_sea_beyond", open_sea_beyond);
	}
}

void WaterBody3D::set_shore_foam_band(float p_depth) {
	shore_foam_band = MAX(p_depth, 0.0f);
	solver.set_foam_settings(foam_enabled, choppiness, foam_threshold, foam_persistence, shore_foam_band, shore_undertow);
}

void WaterBody3D::set_foam_threshold(float p_threshold) {
	foam_threshold = p_threshold;
	solver.set_foam_settings(foam_enabled, choppiness, foam_threshold, foam_persistence, shore_foam_band, shore_undertow);
}

void WaterBody3D::set_foam_persistence(float p_seconds) {
	foam_persistence = MAX(p_seconds, 0.05f);
	solver.set_foam_settings(foam_enabled, choppiness, foam_threshold, foam_persistence, shore_foam_band, shore_undertow);
}

void WaterBody3D::set_normal_mode(NormalMode p_mode) {
	normal_mode = p_mode;
	if (water_material.is_valid()) {
		water_material->set_shader_parameter("per_pixel_normals", normal_mode == NORMAL_MODE_PER_PIXEL);
	}
}

void WaterBody3D::set_fetch(float p_fetch) {
	fetch = MAX(p_fetch, 0.0f);
	_reinit_spectrum();
}

float WaterBody3D::get_effective_fetch() const {
	if (fetch > 0.0f) {
		return fetch;
	}
	// Enclosed water (a surface_mesh outline) can't have more fetch than its
	// own size; an open square stands for a patch of open sea.
	return footprint_extent > 0.0 ? (float)footprint_extent : 100000.0f;
}

void WaterBody3D::set_seabed_from_floor(bool p_enabled) {
	seabed_from_floor = p_enabled;
	_rebuild();
}

void WaterBody3D::set_seabed_collision_mask(uint32_t p_mask) {
	seabed_collision_mask = p_mask;
	_rebuild();
}

void WaterBody3D::set_shallow_fade_depth(float p_depth) {
	shallow_fade_depth = MAX(p_depth, 0.01f);
	_rebuild();
}

void WaterBody3D::set_render_extent(float p_extent) {
	render_extent = MAX(p_extent, 0.0f);
	_rebuild();
}

void WaterBody3D::set_surface_mesh(const Ref<Mesh> &p_mesh) {
	surface_mesh = p_mesh;
	_rebuild();
}

void WaterBody3D::set_water_material(const Ref<ShaderMaterial> &p_material) {
	water_material = p_material;
	own_material = p_material.is_null();
	if (p_material.is_valid() && p_material->get_shader().is_valid()) {
		// The node's own material can come back from a saved scene as a
		// sub-resource: recognize it so temporal_aa/screen_space can rebuild
		// its shader variant instead of treating it as a custom material.
		const String code = p_material->get_shader()->get_code();
		if (code.contains("WaterBody3D built-in water material")) {
			own_material = true;
		} else {
			// Scenes saved before the marker was added: the code is still an
			// exact copy of one of the generated variants, so compare.
			for (int screen = 0; screen < 2 && !own_material; screen++) {
				for (int temporal = 0; temporal < 2 && !own_material; temporal++) {
					own_material = code == _make_shader_code(screen != 0, temporal != 0, false);
				}
			}
		}
	}
	if (mesh_instance != nullptr && water_material.is_valid()) {
		mesh_instance->set_material_override(water_material);
	}
}

void WaterBody3D::set_quality(Quality p_quality) {
	if (quality == p_quality) {
		return;
	}
	quality = p_quality;
	_apply_quality();
}

void WaterBody3D::_apply_quality() {
	if (quality == QUALITY_CUSTOM) {
		return;
	}
	const int n = quality == QUALITY_LOW ? 64 : (quality == QUALITY_MEDIUM ? 96 : 128);
	set_ocean_grid_resolution(n);
	set_grid_resolution(n);
	if (water_material.is_valid()) {
		water_material->set_shader_parameter("shading_quality", (int)quality);
	}
}

void WaterBody3D::set_lod_quality(LODQuality p_quality) {
	if (lod_quality == p_quality) {
		return;
	}
	lod_quality = p_quality;
	_apply_lod_quality();
	notify_property_list_changed();
}

void WaterBody3D::_apply_lod_quality() {
	const float previous_growth = lod_mesh_growth;
	switch (lod_quality) {
		case LOD_QUALITY_ULTRA:
			lod_distance_start = 150.0f;
			lod_distance_mid = 450.0f;
			lod_distance_end = 900.0f;
			lod_surface_scale = 0.55f;
			lod_surface_height = 120.0f;
			lod_mesh_growth = 1.1f;
			break;
		case LOD_QUALITY_HIGH:
			lod_distance_start = 100.0f;
			lod_distance_mid = 350.0f;
			lod_distance_end = 650.0f;
			lod_surface_scale = 0.40f;
			lod_surface_height = 80.0f;
			lod_mesh_growth = 1.15f;
			break;
		case LOD_QUALITY_MEDIUM:
			lod_distance_start = 70.0f;
			lod_distance_mid = 230.0f;
			lod_distance_end = 420.0f;
			lod_surface_scale = 0.32f;
			lod_surface_height = 60.0f;
			lod_mesh_growth = 1.25f;
			break;
		case LOD_QUALITY_LOW:
			lod_distance_start = 45.0f;
			lod_distance_mid = 140.0f;
			lod_distance_end = 260.0f;
			lod_surface_scale = 0.22f;
			lod_surface_height = 40.0f;
			lod_mesh_growth = 1.4f;
			break;
		case LOD_QUALITY_CUSTOM:
			return;
	}
	_apply_lod_shading();
	if (!Math::is_equal_approx(lod_mesh_growth, previous_growth)) {
		_rebuild(); // the ring spacing is baked into the mesh
	}
}

void WaterBody3D::_apply_lod_shading() {
	if (water_material.is_null()) {
		return;
	}
	water_material->set_shader_parameter("distance_fade_start", lod_distance_start);
	water_material->set_shader_parameter("distance_fade_mid", lod_distance_mid);
	water_material->set_shader_parameter("distance_fade_end", lod_distance_end);
	water_material->set_shader_parameter("lod_surface_scale", lod_surface_scale);
	water_material->set_shader_parameter("lod_surface_height", lod_surface_height);
}

void WaterBody3D::_mark_lod_custom() {
	if (lod_quality == LOD_QUALITY_CUSTOM) {
		return;
	}
	// A hand-tuned dial is no longer any preset's settings.
	lod_quality = LOD_QUALITY_CUSTOM;
	notify_property_list_changed();
}

void WaterBody3D::set_lod_distance_start(float p_distance) {
	p_distance = MAX(p_distance, 0.0f);
	if (lod_distance_start == p_distance) {
		return;
	}
	lod_distance_start = p_distance;
	_mark_lod_custom();
	_apply_lod_shading();
}

void WaterBody3D::set_lod_distance_mid(float p_distance) {
	p_distance = MAX(p_distance, 0.0f);
	if (lod_distance_mid == p_distance) {
		return;
	}
	lod_distance_mid = p_distance;
	_mark_lod_custom();
	_apply_lod_shading();
}

void WaterBody3D::set_lod_distance_end(float p_distance) {
	p_distance = MAX(p_distance, 0.0f);
	if (lod_distance_end == p_distance) {
		return;
	}
	lod_distance_end = p_distance;
	_mark_lod_custom();
	_apply_lod_shading();
}

void WaterBody3D::set_lod_surface_scale(float p_scale) {
	p_scale = CLAMP(p_scale, 0.0f, 1.0f);
	if (lod_surface_scale == p_scale) {
		return;
	}
	lod_surface_scale = p_scale;
	_mark_lod_custom();
	_apply_lod_shading();
}

void WaterBody3D::set_lod_surface_height(float p_height) {
	p_height = MAX(p_height, 0.0f);
	if (lod_surface_height == p_height) {
		return;
	}
	lod_surface_height = p_height;
	_mark_lod_custom();
	_apply_lod_shading();
}

void WaterBody3D::set_lod_mesh_growth(float p_growth) {
	p_growth = CLAMP(p_growth, 1.01f, 2.0f);
	if (lod_mesh_growth == p_growth) {
		return;
	}
	lod_mesh_growth = p_growth;
	_mark_lod_custom();
	_rebuild();
}

void WaterBody3D::set_sun_path(const NodePath &p_path) {
	sun_path = p_path;
	sun_light = nullptr;
	sun_retry = 0;
	_resolve_sun();
	_update_sun();
}

void WaterBody3D::_resolve_sun() {
	if (sun_light != nullptr) {
		return;
	}
	// sun_path may be set from a scene file before the node enters the tree;
	// get_tree() asserts there. _update() retries.
	if (!is_inside_tree()) {
		return;
	}
	if (!sun_path.is_empty()) {
		sun_light = Object::cast_to<DirectionalLight3D>(get_node_or_null(sun_path));
		if (sun_light == nullptr && !warned_no_sun) {
			warned_no_sun = true;
			WARN_PRINT("WaterBody3D: sun_path does not point at a DirectionalLight3D; the sun glitter stays at its default.");
		}
		return;
	}
	SceneTree *tree = get_tree();
	if (tree == nullptr) {
		return;
	}
	// No explicit light: use the first directional light in the scene, the
	// common "one sun" case. Retried periodically by _update() until one
	// exists (nodes may be added after the water), then cached.
	TypedArray<Node> lights = tree->get_root()->find_children("*", "DirectionalLight3D", true, false);
	if (lights.is_empty()) {
		return;
	}
	sun_light = Object::cast_to<DirectionalLight3D>(lights[0]);
}

void WaterBody3D::_update_sun() {
	if (sun_light == nullptr || water_material.is_null()) {
		return;
	}
	// A DirectionalLight3D shines along its -Z, so +Z points at the sun.
	const Vector3 toward_sun = sun_light->get_global_transform().basis.get_column(2).normalized();
	const Color light_color = sun_light->get_color().srgb_to_linear();
	const Vector3 color(MIN(light_color.r, 1.0f), MIN(light_color.g, 1.0f), MIN(light_color.b, 1.0f));
	const float energy = CLAMP(sun_light->get_param(Light3D::PARAM_ENERGY), 0.0f, 4.0f);
	water_material->set_shader_parameter("sun_direction", toward_sun);
	water_material->set_shader_parameter("sun_color", color);
	water_material->set_shader_parameter("sun_energy", energy);
}

void WaterBody3D::_update_sky_colors() {
	if (water_material.is_null()) {
		return;
	}
	Ref<World3D> world = get_world_3d();
	Environment *environment = world.is_valid() ? world->get_environment().ptr() : nullptr;
	Ref<Sky> sky = environment != nullptr ? environment->get_sky() : Ref<Sky>();
	if (sky.is_null()) {
		return;
	}
	ProceduralSkyMaterial *procedural = Object::cast_to<ProceduralSkyMaterial>(sky->get_material().ptr());
	if (procedural == nullptr) {
		return;
	}
	// Keep the reflected sky in step with the scene's own procedural sky. A
	// panorama/HDRI sky leaves the material's defaults alone.
	const Color zenith = procedural->get_sky_top_color().srgb_to_linear();
	const Color horizon = procedural->get_sky_horizon_color().srgb_to_linear();
	const Color ground = procedural->get_ground_horizon_color().srgb_to_linear();
	water_material->set_shader_parameter("sky_zenith_color", Vector3(zenith.r, zenith.g, zenith.b));
	water_material->set_shader_parameter("sky_horizon_color", Vector3(horizon.r, horizon.g, horizon.b));
	water_material->set_shader_parameter("sky_ground_color", Vector3(ground.r, ground.g, ground.b));
}

static float _detail_noise_hash(int p_x, int p_y) {
	uint32_t h = (uint32_t)p_x * 374761393u + (uint32_t)p_y * 668265263u;
	h = (h ^ (h >> 13)) * 1274126177u;
	h ^= h >> 16;
	return (float)(h & 0xffffu) / 65535.0f;
}

// Tileable value noise: the lattice wraps at p_period cells.
static float _detail_noise(float p_x, float p_y, int p_period) {
	const int x0 = (int)Math::floor(p_x);
	const int y0 = (int)Math::floor(p_y);
	const float fx = p_x - (float)x0;
	const float fy = p_y - (float)y0;
	const int xa = ((x0 % p_period) + p_period) % p_period;
	const int ya = ((y0 % p_period) + p_period) % p_period;
	const int xb = (xa + 1) % p_period;
	const int yb = (ya + 1) % p_period;
	const float sx = fx * fx * (3.0f - 2.0f * fx);
	const float sy = fy * fy * (3.0f - 2.0f * fy);
	const float a = _detail_noise_hash(xa, ya);
	const float b = _detail_noise_hash(xb, ya);
	const float c = _detail_noise_hash(xa, yb);
	const float d = _detail_noise_hash(xb, yb);
	return Math::lerp(Math::lerp(a, b, sx), Math::lerp(c, d, sx), sy);
}

static float _detail_height(float p_u, float p_v) {
	return 0.55f * _detail_noise(p_u * 4.0f, p_v * 4.0f, 4) +
			0.30f * _detail_noise(p_u * 8.0f, p_v * 8.0f, 8) +
			0.15f * _detail_noise(p_u * 16.0f, p_v * 16.0f, 16);
}

// The built-in material's tiling detail normal, generated once (no asset
// dependency, deterministic for a given build): three octaves of tileable
// value noise turned into a tangent-space normal.
void WaterBody3D::_build_detail_normal() {
	if (detail_normal_tex.is_valid()) {
		return;
	}
	const int size = 128;
	const real_t step = 1.0f / (real_t)size;
	// Mipmaps matter: the map is sampled at grazing angles far away, where an
	// unfiltered level 0 aliases into moving speckle.
	Ref<Image> image = Image::create_empty(size, size, true, Image::FORMAT_RGB8);
	for (int y = 0; y < size; y++) {
		for (int x = 0; x < size; x++) {
			const real_t u = (real_t)x / (real_t)size;
			const real_t v = (real_t)y / (real_t)size;
			const real_t hl = _detail_height(u - step, v);
			const real_t hr = _detail_height(u + step, v);
			const real_t hd = _detail_height(u, v - step);
			const real_t hu = _detail_height(u, v + step);
			Vector3 n(-(hr - hl) * 0.5f / step * 0.06f, -(hu - hd) * 0.5f / step * 0.06f, 1.0f);
			n.normalize();
			image->set_pixel(x, y, Color(n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, n.z * 0.5f + 0.5f));
		}
	}
	image->generate_mipmaps();
	detail_normal_tex = ImageTexture::create_from_image(image);
}

// The built-in material's sampler slots are filled from the start: the
// solver's zero-copy Texture2DRD bindings only exist once its RenderingDevice
// work has started (a frame or two later). Flat 1x1 stand-ins mean the node
// is self-contained -- no unbound sampler, no missing texture in the
// inspector -- until _bind_textures() replaces them.
void WaterBody3D::_bind_fallback_textures() {
	if (water_material.is_null()) {
		return;
	}
	if (fallback_black_tex.is_null()) {
		Ref<Image> black = Image::create_empty(1, 1, false, Image::FORMAT_RGB8);
		black->fill(Color(0, 0, 0));
		fallback_black_tex = ImageTexture::create_from_image(black);
		Ref<Image> white = Image::create_empty(1, 1, false, Image::FORMAT_RGB8);
		white->fill(Color(1, 1, 1));
		fallback_white_tex = ImageTexture::create_from_image(white);
	}
	water_material->set_shader_parameter("ripple_height_tex", fallback_black_tex);
	water_material->set_shader_parameter("ocean_height_tex", fallback_black_tex);
	water_material->set_shader_parameter("ocean_fade_tex", fallback_white_tex);
	water_material->set_shader_parameter("shore_depth_tex", fallback_white_tex);
	water_material->set_shader_parameter("ocean_disp_tex", fallback_black_tex);
	water_material->set_shader_parameter("ocean_deriv_tex", fallback_black_tex);
	water_material->set_shader_parameter("ocean_foam_tex", fallback_black_tex);
	water_material->set_shader_parameter("shore_foam_tex", fallback_black_tex);
	water_material->set_shader_parameter("swash_tex", fallback_black_tex);
}

void WaterBody3D::_apply_shading_settings() {
	if (water_material.is_null()) {
		return;
	}
	_build_detail_normal();
	water_material->set_shader_parameter("detail_normal", detail_normal_tex);
	_bind_fallback_textures();
	if (quality != QUALITY_CUSTOM) {
		water_material->set_shader_parameter("shading_quality", (int)quality);
	}
	_apply_lod_shading();
	_update_sky_colors();
	_resolve_sun();
	_update_sun();
}

void WaterBody3D::set_auto_recenter(bool p_enabled) {
	auto_recenter = p_enabled;
}

void WaterBody3D::set_underwater_effect(bool p_enabled) {
	if (underwater_effect == p_enabled) {
		return;
	}
	underwater_effect = p_enabled;
	_update_underwater_node();
}

void WaterBody3D::_update_underwater_node() {
	// No underwater grade in the editor: the editor camera is a preview
	// camera, not a swimmer, so grading its view would fight the user.
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}
	if (!underwater_effect) {
		if (underwater_effect_node != nullptr) {
			remove_child(underwater_effect_node);
			memdelete(underwater_effect_node);
			underwater_effect_node = nullptr;
		}
		return;
	}
	if (underwater_effect_node != nullptr || !is_inside_tree()) {
		return;
	}
	underwater_effect_node = memnew(WaterUnderwaterEffect);
	underwater_effect_node->set_quiet(true);
	underwater_effect_node->set_water_body_path(NodePath(".."));
	add_child(underwater_effect_node, false, INTERNAL_MODE_BACK);
}

void WaterBody3D::set_screen_space(bool p_enabled) {
	if (screen_space == p_enabled) {
		return;
	}
	screen_space = p_enabled;
	if (own_material) {
		// The screen-space path is compiled out by default (it needs a screen
		// copy every frame): rebuilding the material also resets its
		// parameters, which _rebuild()/_bind_textures() restore.
		water_material = Ref<ShaderMaterial>();
		textures_bound = false;
		_rebuild();
	}
}

void WaterBody3D::set_temporal_aa(bool p_enabled) {
	if (temporal_aa == p_enabled) {
		return;
	}
	temporal_aa = p_enabled;
	temporal_frame = 0;
	temporal_warned = false;
	_update_temporal_effect();
	_check_temporal_support();
	if (own_material) {
		// The render_mode is compiled into the shader: rebuild the material.
		water_material = Ref<ShaderMaterial>();
		textures_bound = false;
		_rebuild();
	}
}

static float _halton(int p_index, int p_base) {
	real_t f = 1.0;
	real_t r = 0.0;
	while (p_index > 0) {
		f /= (real_t)p_base;
		r += f * (real_t)(p_index % p_base);
		p_index /= p_base;
	}
	return (float)r;
}

void WaterBody3D::_update_temporal_state() {
	temporal_frame = (temporal_frame + 1) % 64;
	const real_t node_y = get_global_position().y;
	WaterTemporalEffect::set_surface(node_y + water_level);
	if (World3D *world = get_world_3d().ptr()) {
		WaterTemporalEffect::refresh(world);
	}
	if (solver.is_available()) {
		WaterTemporalEffect::set_water_data(solver.get_ripple_height_texture_rd_rid(), solver.get_ocean_height_texture_rd_rid(), solver.get_ripple_height_prev_texture_rd_rid(), solver.get_ocean_height_prev_texture_rd_rid(), solver.get_ocean_fade_texture_rd_rid(), grid_center, active_domain_size, ocean_domain_size, (float)node_y, water_level);
	} else {
		WaterTemporalEffect::set_water_data(RID(), RID(), RID(), RID(), RID(), Vector2(), Vector2(), Vector2(), 0.0f, 0.0f);
	}

	const Viewport *viewport = get_viewport();
	if (viewport == nullptr) {
		return;
	}
	const Size2 render_size = viewport->get_visible_rect().size * viewport->get_scaling_3d_scale();
	if (render_size.x < 1.0 || render_size.y < 1.0) {
		return;
	}
	// Halton(2,3), centered on the pixel: one sample per frame, the history
	// resolve averages the eight-frame cycle.
	const Vector2 pixel_jitter(_halton(temporal_frame + 1, 2) - 0.5f, _halton(temporal_frame + 1, 3) - 0.5f);
	if (water_material.is_valid()) {
		water_material->set_shader_parameter("temporal_jitter", pixel_jitter * 2.0f / render_size);
	}
}

void WaterBody3D::_check_temporal_support() {
	if (!temporal_aa || temporal_warned) {
		return;
	}
	if (Engine::get_singleton()->is_editor_hint()) {
		temporal_warned = true;
		WARN_PRINT("WaterBody3D: temporal_aa runs in Play; the editor preview uses the normal path.");
		return;
	}
	if (RenderingServer::get_singleton()->get_current_rendering_method() != "forward_plus") {
		temporal_warned = true;
		WARN_PRINT("WaterBody3D: temporal_aa needs the Forward+ renderer; the current renderer keeps the normal path.");
		return;
	}
	const Viewport *viewport = get_viewport();
	if (viewport != nullptr && viewport->get_msaa_3d() != Viewport::MSAA_DISABLED) {
		temporal_warned = true;
		WARN_PRINT("WaterBody3D: temporal_aa needs 3D MSAA disabled (rendering/anti_aliasing/quality/msaa_3d = Disabled); the water keeps the normal path.");
		return;
	}
	if (!own_material && water_material.is_valid()) {
		temporal_warned = true;
		WARN_PRINT("WaterBody3D: temporal_aa needs the built-in water material; clear the water_material property (set it to empty) and let the node rebuild it. A custom material renders unchanged.");
	}
}

void WaterBody3D::_update_temporal_effect(bool p_exiting) {
	if (Engine::get_singleton()->is_editor_hint()) {
		p_exiting = true;
	}
	World3D *world = is_inside_world() ? get_world_3d().ptr() : nullptr;
	if (temporal_installed && (p_exiting || !temporal_aa || world == nullptr || world->get_instance_id() != temporal_world)) {
		WaterTemporalEffect::uninstall(temporal_world);
		temporal_installed = false;
		temporal_world = ObjectID();
	}
	if (!p_exiting && temporal_aa && world != nullptr && !temporal_installed) {
		WaterTemporalEffect::install(world);
		temporal_installed = true;
		temporal_world = world->get_instance_id();
	}
	if (!p_exiting && temporal_aa) {
		// Explain at startup why a saved temporal_aa may be inactive
		// (editor, MSAA, renderer, custom material).
		_check_temporal_support();
	}
}

void WaterBody3D::set_caustics_enabled(bool p_enabled) {
	caustics_enabled = p_enabled;
	_rebuild();
}

void WaterBody3D::set_caustics_sun_direction(Vector3 p_direction) {
	caustics_sun_direction = p_direction;
}

void WaterBody3D::set_caustics_reference_depth(float p_depth) {
	caustics_reference_depth = MAX(p_depth, 0.1f);
}

Ref<Texture2D> WaterBody3D::get_caustics_texture() const {
	if (!caustics_enabled || !solver.is_available() || caustics_texture.is_null()) {
		return Ref<Texture2D>();
	}
	return caustics_texture;
}

void WaterBody3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_WORLD: {
			sun_light = nullptr;
			sun_retry = 0;
			_rebuild();
			_update_temporal_effect();
			_update_underwater_node();
			set_physics_process_internal(true);
			set_process_internal(true);
		} break;
		case NOTIFICATION_EXIT_WORLD: {
			_update_temporal_effect(true);
			set_physics_process_internal(false);
			set_process_internal(false);
			// The solver's textures go with this node (or are rebuilt on the
			// way back in): let go of them first.
			_detach_textures();
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			_update(get_physics_process_delta_time());
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (temporal_aa && !Engine::get_singleton()->is_editor_hint()) {
				_update_temporal_state();
			}
			_update_mesh_offset();
		} break;
	}
}

String WaterBody3D::_make_shader_code(bool p_screen_space, bool p_temporal, bool p_mark) const {
	String code = WATER_SHADER_SRC;
	const String type_line = "shader_type spatial;";
	const int type_at = code.find(type_line);
	if (type_at < 0) {
		return code;
	}
	if (p_screen_space) {
		code = code.insert(type_at + type_line.length(), "\n#define WATER_SCREEN_SPACE");
	}
	if (p_temporal) {
		const String mode_line = "render_mode blend_mix, cull_disabled, unshaded;";
		const int mode_at = code.find(mode_line);
		if (mode_at >= 0) {
			code = code.replace(mode_line, "render_mode blend_mix, cull_disabled, unshaded, water_temporal;");
		}
		const int at = code.find(type_line);
		if (at >= 0) {
			code = code.insert(at + type_line.length(), "\n#define WATER_TEMPORAL");
		}
	}
	if (p_mark) {
		const int at = code.find(type_line);
		if (at >= 0) {
			code = code.insert(at + type_line.length(), "\n// WaterBody3D built-in water material");
		}
	}
	return code;
}

void WaterBody3D::_rebuild() {
	if (!is_inside_world()) {
		return;
	}
	if (mesh_instance == nullptr) {
		mesh_instance = memnew(MeshInstance3D);
		mesh_instance->set_cast_shadows_setting(GeometryInstance3D::SHADOW_CASTING_SETTING_OFF);
		add_child(mesh_instance, false, INTERNAL_MODE_BACK);
	}
	const bool want_temporal = temporal_aa && !Engine::get_singleton()->is_editor_hint();
	if (own_material && water_material.is_valid() && water_material->get_shader().is_valid()) {
		if (water_material->get_shader()->get_code() != _make_shader_code(screen_space, want_temporal, true)) {
			water_material = Ref<ShaderMaterial>();
		}
	}
	if (water_material.is_null()) {
		Ref<Shader> shader;
		shader.instantiate();
		shader->set_code(_make_shader_code(screen_space, want_temporal, true));
		Ref<ShaderMaterial> m;
		m.instantiate();
		m->set_shader(shader);
		water_material = m;
		own_material = true;
	}
	mesh_instance->set_material_override(water_material);
	_apply_shading_settings();

	// Shape: a surface_mesh footprint fits the ripple grid to its bounds and
	// masks everything outside it dry; without one, the plain square domain.
	Vector2 local_center;
	active_domain_size = domain_size;
	footprint_mask.clear();
	cell_depth.clear();
	footprint_extent = 0.0;
	water_mesh.unref();
	if (surface_mesh.is_valid()) {
		const Vector<Vector2> tris = _collect_footprint();
		if (tris.is_empty()) {
			WARN_PRINT("WaterBody3D: surface_mesh has no triangles with any X/Z extent; using the square domain instead.");
		} else {
			Rect2 bounds(tris[0], Vector2());
			for (const Vector2 &v : tris) {
				bounds.expand_to(v);
			}
			local_center = bounds.get_center();
			// One dry cell of margin on each side, so the shoreline never sits
			// on the grid edge.
			const real_t side = MAX(bounds.size.x, bounds.size.y);
			footprint_extent = side;
			const real_t padded = side * (real_t)grid_resolution / (real_t)MAX(grid_resolution - 2, 1);
			active_domain_size = Vector2(padded, padded);
			footprint_mask = _rasterize_wet_mask(tris, local_center, active_domain_size, grid_resolution);
			water_mesh = _build_footprint_mesh(tris, bounds, MAX(grid_resolution, ocean_grid_resolution));
		}
	}
	if (water_mesh.is_null() && render_extent > MAX(ocean_domain_size.x, ocean_domain_size.y)) {
		water_mesh = _build_extended_plane(ocean_domain_size, ocean_grid_resolution, render_extent, lod_mesh_growth);
	}
	if (water_mesh.is_null()) {
		Ref<PlaneMesh> plane;
		plane.instantiate();
		plane->set_size(ocean_domain_size);
		// Subdivisions, not vertex count -- PlaneMesh's own convention.
		plane->set_subdivide_width(ocean_grid_resolution);
		plane->set_subdivide_depth(ocean_grid_resolution);
		water_mesh = plane;
	}
	mesh_instance->set_mesh(water_mesh);
	mesh_offset = Vector2(Math::INF, Math::INF); // re-applied for the new mesh on the next update

	const Vector3 origin = get_global_position();
	grid_center = Vector2(origin.x, origin.z) + local_center;

	water_material->set_shader_parameter("ripple_domain_size", active_domain_size);
	water_material->set_shader_parameter("ocean_domain_size", ocean_domain_size);
	water_material->set_shader_parameter("grid_center", grid_center);
	water_material->set_shader_parameter("open_sea_beyond", open_sea_beyond);
	water_material->set_shader_parameter("water_level", water_level);
	water_material->set_shader_parameter("wave_amplitude", wave_amplitude);
	water_material->set_shader_parameter("choppiness", choppiness);
	water_material->set_shader_parameter("per_pixel_normals", normal_mode == NORMAL_MODE_PER_PIXEL);

	// The editor previews the ocean too (see _update()): adding the node is
	// the whole setup, in the editor and in Play alike. A Compatibility
	// editor has no main RenderingDevice, so nothing can run and there is no
	// RD texture its renderer could sample -- there the preview is the flat
	// stand-in surface.
	if (Engine::get_singleton()->is_editor_hint() && RenderingServer::get_singleton()->get_rendering_device() == nullptr) {
		return;
	}

	if (seabed_from_floor) {
		// Rays need the floor already in the physics space -- defer to the
		// first physics tick (see _update()).
		seabed_pending = true;
		return;
	}
	if (!footprint_mask.is_empty()) {
		const int cells = grid_resolution * grid_resolution;
		cell_depth.resize(cells);
		float *dw = cell_depth.ptrw();
		for (int i = 0; i < cells; i++) {
			dw[i] = footprint_mask[i] ? depth : WaterSolver::WALL_DEPTH;
		}
	}
	_configure_solver();
}

void WaterBody3D::_sample_seabed() {
	// One ray per ripple cell center, straight down from well above the water
	// onto static bodies only: anything else it hits (a floater, a player) is
	// excluded and the ray cast again. The seabed can rise above the water --
	// that's dry beach, depth <= 0.
	constexpr real_t PROBE_ABOVE = 100.0;
	constexpr real_t PROBE_BELOW = 1000.0;
	constexpr int MAX_SKIPS = 8;
	const int n = grid_resolution;
	cell_depth.resize(n * n);
	float *dw = cell_depth.ptrw();
	const real_t surface_y = get_global_position().y + water_level;
	const Vector2 cell = active_domain_size / (real_t)n;
	const Vector2 grid_min = grid_center - active_domain_size * 0.5f;
	PhysicsDirectSpaceState3D *space = get_world_3d()->get_direct_space_state();
	PhysicsServer3D *ps = PhysicsServer3D::get_singleton();
	int hits = 0;
	for (int z = 0; z < n; z++) {
		for (int x = 0; x < n; x++) {
			const int i = z * n + x;
			if (!footprint_mask.is_empty() && !footprint_mask[i]) {
				dw[i] = WaterSolver::WALL_DEPTH;
				continue;
			}
			const Vector2 p = grid_min + Vector2(x + 0.5f, z + 0.5f) * cell;
			PhysicsDirectSpaceState3D::RayParameters ray;
			ray.from = Vector3(p.x, surface_y + PROBE_ABOVE, p.y);
			ray.to = Vector3(p.x, surface_y - PROBE_BELOW, p.y);
			ray.collision_mask = seabed_collision_mask;
			ray.collide_with_areas = false;
			float d = depth; // nothing below: open water of the constant depth
			for (int skip = 0; skip < MAX_SKIPS; skip++) {
				PhysicsDirectSpaceState3D::RayResult result;
				if (space == nullptr || !space->intersect_ray(ray, result)) {
					break;
				}
				if (ps->body_get_mode(result.rid) == PhysicsServer3D::BODY_MODE_STATIC) {
					d = surface_y - result.position.y;
					hits++;
					break;
				}
				ray.exclude.insert(result.rid);
			}
			dw[i] = d;
		}
	}
	if (hits == 0) {
		WARN_PRINT("WaterBody3D: seabed_from_floor found no static body under the water (check seabed_collision_mask); using the constant depth.");
	}
}

void WaterBody3D::_update_cell_fade() {
	cell_fade.resize(cell_depth.size());
	float *fw = cell_fade.ptrw();
	for (int i = 0; i < cell_depth.size(); i++) {
		fw[i] = WaterSolver::shallow_fade(cell_depth[i], shallow_fade_depth);
	}
}

void WaterBody3D::_configure_solver() {
	_update_cell_fade();
	WaterSolver::Settings s;
	s.grid_resolution = grid_resolution;
	s.domain_size = active_domain_size;
	s.grid_center = grid_center;
	s.cell_depth = cell_depth;
	s.shallow_fade_depth = shallow_fade_depth;
	s.depth = depth;
	s.damping = damping;
	s.ripple_amplitude = ripple_amplitude;
	s.gravity = 9.81f;
	s.water_level = water_level;
	s.ocean_grid_resolution = ocean_grid_resolution;
	s.ocean_domain_size = ocean_domain_size;
	s.wind_speed = wind_speed;
	s.wind_direction = wind_direction;
	s.wave_amplitude = wave_amplitude;
	s.fetch = get_effective_fetch();
	s.caustics_enabled = caustics_enabled;
	// configure() frees the solver's textures and makes new ones: let go of
	// the old ones first (a runtime change -- wave_amplitude, say -- drew with
	// freed textures until the new ones were bound).
	_detach_textures();
	solver.configure(s);
	solver.set_foam_settings(foam_enabled, choppiness, foam_threshold, foam_persistence, shore_foam_band, shore_undertow);
	solver.set_swash_settings(swash_run_up, swash_drain_speed, wet_sand_dry_time);
	if (!solver.has_device() && !warned_no_device) {
		warned_no_device = true;
		WARN_PRINT("WaterBody3D: the water compute solver could not start (no RenderingDevice / compute support).");
	}

	frames_since_refresh = REFRESH_EVERY_FRAMES; // force an immediate CPU cache refresh on the next _update()
}

void WaterBody3D::_detach_textures() {
	const Ref<Texture2DRD> textures[] = { ripple_height_tex, ocean_height_tex, ocean_fade_tex, shore_depth_tex, ocean_disp_tex, ocean_deriv_tex, ocean_foam_tex, shore_foam_tex, swash_tex, caustics_texture };
	for (const Ref<Texture2DRD> &texture : textures) {
		if (texture.is_valid()) {
			texture->set_texture_rd_rid(RID());
		}
	}
	textures_bound = false; // rebound once the solver's (new) textures exist
}

void WaterBody3D::_update_mesh_offset() {
	// Open water past the simulated square: the mesh's vertices spread out
	// with distance from its dense core, and far out they're meters apart --
	// too coarse for the short waves, so what's drawn there missed what
	// sample_height() (and so a boat) sees by tenths of a meter. The core
	// follows the camera instead, smoothly, every frame: near the camera the
	// dense vertices sample the waves finely wherever they sit, and the
	// coarse surface past it reshapes gradually (~1 cm per 5 cm moved, 20-35
	// m out) instead of jumping. (Snapping popped: whole 40 m ocean tiles
	// re-meshed everything past the core at once -- 15 cm jumps -- and
	// core-spacing steps made a constant 6 cm shimmer.)
	if (mesh_instance == nullptr || water_material.is_null() || surface_mesh.is_valid() || render_extent <= MAX(ocean_domain_size.x, ocean_domain_size.y)) {
		return;
	}
	const Viewport *vp = get_viewport();
	const Camera3D *cam = vp != nullptr ? vp->get_camera_3d() : nullptr;
#ifdef TOOLS_ENABLED
	// Editing: the camera the author actually looks through is the 3D view
	// last used -- each split view has its own camera, and the edited scene
	// only lives in viewport 0.
	if (Engine::get_singleton()->is_editor_hint() && Node3DEditor::get_singleton() != nullptr) {
		Node3DEditorViewport *view = Node3DEditor::get_singleton()->get_last_used_viewport();
		if (view != nullptr && view->get_camera_3d() != nullptr && view->get_camera_3d()->get_world_3d() == get_world_3d()) {
			cam = view->get_camera_3d();
		}
	}
#endif
	if (cam == nullptr) {
		return;
	}
	const Vector3 cp = cam->get_global_position();
	// Endless sea out of the box: carry the interactive ripple square (and the
	// node) along with the camera. The FFT ocean is world-anchored, so only
	// local ripples are cleared; no rebuild on this path. Not in the editor:
	// moving (and rebuilding) the authored node because the author panned is
	// not a preview -- the mesh offset below is.
	if (!Engine::get_singleton()->is_editor_hint() && auto_recenter && solver.is_available() && !seabed_from_floor && cell_depth.is_empty()) {
		if (Vector2(cp.x, cp.z).distance_squared_to(grid_center) > 100.0f) {
			recenter(cp);
		}
	}
	const Vector2 rel = Vector2(cp.x, cp.z) - grid_center;
	const Vector2 offset = rel;
	if (offset == mesh_offset) {
		return;
	}
	mesh_offset = offset;
	water_material->set_shader_parameter("mesh_offset", mesh_offset);
	if (water_mesh.is_valid()) {
		AABB box = water_mesh->get_aabb();
		box.position += mesh_instance->get_global_transform().basis.xform_inv(Vector3(offset.x, 0, offset.y));
		mesh_instance->set_custom_aabb(box.grow(MAX(wave_amplitude * 4.0f, 2.0f)));
	}
}

void WaterBody3D::_update(double p_delta) {
	// Forward+/Mobile editors preview the ocean live; a Compatibility editor
	// has no RenderingDevice to run it on (see _rebuild()).
	if (Engine::get_singleton()->is_editor_hint() && RenderingServer::get_singleton()->get_rendering_device() == nullptr) {
		return;
	}
	water_time += p_delta;
	if (water_material.is_valid()) {
		// Wrapped so float precision holds up in long sessions.
		water_material->set_shader_parameter("water_time", (float)Math::fmod(water_time, 3600.0));
		if (sun_light != nullptr) {
			_update_sun();
		} else if (++sun_retry >= 60) {
			sun_retry = 0;
			_resolve_sun();
		}
	}
	if (seabed_pending) {
		seabed_pending = false;
		_sample_seabed();
		_configure_solver();
		return;
	}
	if (!solver.has_device()) {
		return;
	}
	solver.step(p_delta); // no-ops internally until solver.is_available() (see water_solver.cpp)
	if (!textures_bound && solver.is_available() && !solver.uses_local_device()) {
		_bind_textures();
	}
	if (solver.is_available() && caustics_enabled) {
		solver.render_caustics(Vector3(grid_center.x, get_global_position().y, grid_center.y), caustics_sun_direction, caustics_reference_depth, 1.333f, _open_beyond());
	}
	frames_since_refresh++;
	if (frames_since_refresh >= REFRESH_EVERY_FRAMES) {
		frames_since_refresh = 0;
		_refresh_cpu_cache();
	}
}

void WaterBody3D::_bind_textures() {
	// Zero-copy: wrap the solver's own RD texture RIDs directly. Set once --
	// Texture2DRD holds the RID by reference, so whatever the blit shaders
	// write each step (see water_solver.cpp's rt_step()) is what the
	// material samples automatically; no per-frame reassignment needed.
	if (ripple_height_tex.is_null()) {
		ripple_height_tex.instantiate();
	}
	if (ocean_height_tex.is_null()) {
		ocean_height_tex.instantiate();
	}
	if (ocean_fade_tex.is_null()) {
		ocean_fade_tex.instantiate();
	}
	if (ocean_disp_tex.is_null()) {
		ocean_disp_tex.instantiate();
	}
	ocean_disp_tex->set_texture_rd_rid(solver.get_ocean_displacement_texture_rd_rid());
	if (ocean_deriv_tex.is_null()) {
		ocean_deriv_tex.instantiate();
	}
	ocean_deriv_tex->set_texture_rd_rid(solver.get_ocean_derivative_texture_rd_rid());
	if (ocean_foam_tex.is_null()) {
		ocean_foam_tex.instantiate();
	}
	ocean_foam_tex->set_texture_rd_rid(solver.get_ocean_foam_texture_rd_rid());
	if (shore_foam_tex.is_null()) {
		shore_foam_tex.instantiate();
	}
	shore_foam_tex->set_texture_rd_rid(solver.get_shore_foam_texture_rd_rid());
	if (swash_tex.is_null()) {
		swash_tex.instantiate();
	}
	swash_tex->set_texture_rd_rid(solver.get_swash_texture_rd_rid());
	ripple_height_tex->set_texture_rd_rid(solver.get_ripple_height_texture_rd_rid());
	ocean_height_tex->set_texture_rd_rid(solver.get_ocean_height_texture_rd_rid());
	ocean_fade_tex->set_texture_rd_rid(solver.get_ocean_fade_texture_rd_rid());
	if (shore_depth_tex.is_null()) {
		shore_depth_tex.instantiate();
	}
	shore_depth_tex->set_texture_rd_rid(solver.get_shore_depth_texture_rd_rid());
	if (caustics_enabled) {
		if (caustics_texture.is_null()) {
			caustics_texture.instantiate();
		}
		caustics_texture->set_texture_rd_rid(solver.get_caustics_texture_rd_rid());
	}
	if (water_material.is_valid()) {
		water_material->set_shader_parameter("ripple_height_tex", ripple_height_tex);
		water_material->set_shader_parameter("ocean_height_tex", ocean_height_tex);
		water_material->set_shader_parameter("ocean_fade_tex", ocean_fade_tex);
		water_material->set_shader_parameter("shore_depth_tex", shore_depth_tex);
		water_material->set_shader_parameter("ocean_disp_tex", ocean_disp_tex);
		water_material->set_shader_parameter("ocean_deriv_tex", ocean_deriv_tex);
		water_material->set_shader_parameter("ocean_foam_tex", ocean_foam_tex);
		water_material->set_shader_parameter("shore_foam_tex", shore_foam_tex);
		water_material->set_shader_parameter("swash_tex", swash_tex);
	}
	textures_bound = true;
}

void WaterBody3D::_refresh_cpu_cache() {
	// CPU-side cache for sample_height() only -- rendering doesn't touch
	// this at all (see _bind_textures()). Non-blocking: just whatever the
	// solver's last async readback landed, a few frames stale -- fine for
	// buoyancy, see water_solver.h's own note on why this stays a separate
	// path from the zero-copy texture above.
	solver.get_height_grid(cached_ripple_height, cached_ripple_n, cached_ripple_domain);
	solver.get_ocean_height_grid(cached_ocean_height, cached_ocean_dx, cached_ocean_n, cached_ocean_domain);
	solver.get_ocean_dz_grid(cached_ocean_dz);
}

float WaterBody3D::_bilinear_sample(const Vector<float> &p_grid, int p_n, Vector2 p_domain, float p_world_x, float p_world_z, bool p_repeat) const {
	if (p_grid.is_empty() || p_n <= 0) {
		return 0.0f;
	}
	const float gx = (p_world_x / (p_domain.x * 0.5f) + 1.0f) * 0.5f * p_n - 0.5f;
	const float gz = (p_world_z / (p_domain.y * 0.5f) + 1.0f) * 0.5f * p_n - 0.5f;
	int x0, z0, x1, z1;
	float fx, fz;
	if (p_repeat) {
		const float flx = floorf(gx);
		const float flz = floorf(gz);
		auto wrap = [p_n](int p_i) { return ((p_i % p_n) + p_n) % p_n; };
		x0 = wrap((int)flx);
		z0 = wrap((int)flz);
		x1 = wrap((int)flx + 1);
		z1 = wrap((int)flz + 1);
		fx = gx - flx;
		fz = gz - flz;
	} else {
		x0 = CLAMP((int)floorf(gx), 0, p_n - 1);
		z0 = CLAMP((int)floorf(gz), 0, p_n - 1);
		x1 = CLAMP(x0 + 1, 0, p_n - 1);
		z1 = CLAMP(z0 + 1, 0, p_n - 1);
		fx = CLAMP(gx - x0, 0.0f, 1.0f);
		fz = CLAMP(gz - z0, 0.0f, 1.0f);
	}
	const float h00 = p_grid[z0 * p_n + x0];
	const float h10 = p_grid[z0 * p_n + x1];
	const float h01 = p_grid[z1 * p_n + x0];
	const float h11 = p_grid[z1 * p_n + x1];
	return Math::lerp(Math::lerp(h00, h10, fx), Math::lerp(h01, h11, fx), fz);
}

float WaterBody3D::sample_height(Vector3 p_world_pos) const {
	return sample_height_excluding_wake(p_world_pos, ObjectID());
}

int WaterBody3D::register_wake(ObjectID p_wake) {
	for (int i = 0; i < MAX_WAKES; i++) {
		if (wakes[i] == p_wake || ObjectDB::get_instance(wakes[i]) == nullptr) {
			wakes[i] = p_wake;
			return i;
		}
	}
	return -1;
}

void WaterBody3D::unregister_wake(int p_slot) {
	ERR_FAIL_INDEX(p_slot, MAX_WAKES);
	wakes[p_slot] = ObjectID();
	if (water_material.is_valid()) {
		water_material->set_shader_parameter("wake_height_" + itos(p_slot), Variant());
		water_material->set_shader_parameter("wake_rect_" + itos(p_slot), Vector4());
	}
}

void WaterBody3D::set_wake_slot(int p_slot, const Ref<Texture2D> &p_height, const Vector4 &p_rect) {
	ERR_FAIL_INDEX(p_slot, MAX_WAKES);
	if (water_material.is_null()) {
		return;
	}
	const String i = itos(p_slot);
	if (water_material->get_shader_parameter("wake_height_" + i) != Variant(p_height)) {
		water_material->set_shader_parameter("wake_height_" + i, p_height);
	}
	water_material->set_shader_parameter("wake_rect_" + i, p_rect);
}

float WaterBody3D::_sample_height_excluding_wake_bind(Vector3 p_world_pos, Object *p_wake) const {
	return sample_height_excluding_wake(p_world_pos, p_wake != nullptr ? p_wake->get_instance_id() : ObjectID());
}

float WaterBody3D::sample_height_excluding_wake(Vector3 p_world_pos, ObjectID p_wake) const {
	if (!is_wet(p_world_pos)) {
		return -Math::INF;
	}
	const float x = p_world_pos.x - grid_center.x;
	const float z = p_world_pos.z - grid_center.y;
	// The ripple readback already includes water_level (it's the composed
	// height the renderer draws), so only fall back to it before the first
	// readback lands. Both are relative to the node, like the rendered mesh.
	// Past the simulated square the ripples ease back to still water, as the
	// shader draws them.
	float ripple = cached_ripple_height.is_empty() ? water_level : _bilinear_sample(cached_ripple_height, cached_ripple_n, cached_ripple_domain, x, z);
	if (!cached_ripple_height.is_empty()) {
		const float from_center = MAX(Math::abs(x / cached_ripple_domain.x), Math::abs(z / cached_ripple_domain.y));
		const float inside = 1.0f - Math::smoothstep(0.46f, 0.5f, from_center);
		ripple = Math::lerp(water_level, ripple, inside);
	}
	float fade = cell_fade.is_empty() ? 1.0f : _bilinear_sample(cell_fade, grid_resolution, active_domain_size, x, z);
	// Open sea past the square: full chop there, not the edge cell's (the shader's fade_r()).
	const float beyond = (open_sea_beyond && _open_beyond()) ? Math::smoothstep(0.46f, 0.5f, MAX(Math::abs(x / active_domain_size.x), Math::abs(z / active_domain_size.y))) : 0.0f;
	fade = Math::lerp(fade, 1.0f, beyond);
	// With choppy displacement the surface point above (x, z) rested
	// somewhere else: find p0 with p0 + s * D(p0) = (x, z) by fixed-point
	// iteration (s matches the shader's -choppiness * fade), then read the
	// ocean height there.
	// (The ocean layer is anchored to the world, as the shader reads it: its samples are at world xz, so a re-centre
	// of the ripple grid never slides the waves a boat rides.)
	Vector2 p0(x, z);
	const float s = -choppiness * fade;
	if (s != 0.0f && cached_ocean_dx.size() == cached_ocean_height.size() && cached_ocean_dz.size() == cached_ocean_height.size()) {
		for (int i = 0; i < 4; i++) {
			const Vector2 d(_bilinear_sample(cached_ocean_dx, cached_ocean_n, cached_ocean_domain, p0.x + grid_center.x, p0.y + grid_center.y, true),
					_bilinear_sample(cached_ocean_dz, cached_ocean_n, cached_ocean_domain, p0.x + grid_center.x, p0.y + grid_center.y, true));
			p0 = Vector2(x, z) - d * s;
		}
	}
	const float ocean = _bilinear_sample(cached_ocean_height, cached_ocean_n, cached_ocean_domain, p0.x + grid_center.x, p0.y + grid_center.y, true) * fade;
	// Wakes, everywhere (as the shader draws them).
	float wake = 0.0f;
	for (int i = 0; i < MAX_WAKES; i++) {
		if (wakes[i] == p_wake) {
			continue;
		}
		const WaterWake3D *w = ObjectDB::get_instance<WaterWake3D>(wakes[i]);
		if (w != nullptr) {
			const float out = Math::smoothstep(0.46f, 0.5f, MAX(Math::abs(x / active_domain_size.x), Math::abs(z / active_domain_size.y)));
			wake += w->sample_height(p_world_pos) * Math::lerp(1.0f, out, w->get_shore_grid_fade());
		}
	}
	// Not over dry sand (as the shader): fades out over the last 30 cm of depth.
	if (wake != 0.0f && !cell_depth.is_empty()) {
		wake *= Math::smoothstep(0.0f, 0.3f, Math::lerp(_bilinear_sample(cell_depth, grid_resolution, active_domain_size, x, z), 1000.0f, beyond));
	}
	return get_global_position().y + ripple + ocean + wake;
}

bool WaterBody3D::_open_beyond() const {
	return surface_mesh.is_null() && render_extent > MAX(domain_size.x, domain_size.y);
}

bool WaterBody3D::_inside_render_extent(float p_rel_x, float p_rel_z) const {
	return Math::abs(p_rel_x) <= render_extent * 0.5f && Math::abs(p_rel_z) <= render_extent * 0.5f;
}

bool WaterBody3D::is_wet(Vector3 p_world_pos) const {
	const float rel_x = p_world_pos.x - grid_center.x;
	const float rel_z = p_world_pos.z - grid_center.y;
	if (_open_beyond() && !_inside_render_extent(rel_x, rel_z)) {
		return false; // past the drawn surface
	}
	if (cell_depth.is_empty()) {
		return true;
	}
	const int n = grid_resolution;
	if (cell_depth.size() != n * n) {
		return false;
	}
	int gx = (int)Math::floor((rel_x / active_domain_size.x + 0.5f) * n);
	int gz = (int)Math::floor((rel_z / active_domain_size.y + 0.5f) * n);
	if (gx < 0 || gz < 0 || gx >= n || gz >= n) {
		if (!_open_beyond()) {
			return false;
		}
		if (open_sea_beyond) {
			return true; // open, deep sea past the square
		}
		// Open water past the simulated square carries the edge's seabed on
		// outward, as the shader draws it: the sea side stays sea, the beach
		// side stays sand.
		gx = CLAMP(gx, 0, n - 1);
		gz = CLAMP(gz, 0, n - 1);
	}
	return cell_depth[gz * n + gx] > 0.0f;
}

Vector<Vector2> WaterBody3D::_collect_footprint() const {
	Vector<Vector2> tris;
	if (surface_mesh.is_null()) {
		return tris;
	}
	// Only up-facing triangles: a closed mesh (a baked CSG disc has a top and
	// a bottom) would otherwise cover its footprint twice. A mesh with no
	// up-facing triangles at all (flipped winding) falls back to every face.
	const real_t up = plane_mesh_xz_winding();
	const Vector<Face3> faces = surface_mesh->get_faces();
	Vector<Vector2> any_facing;
	for (const Face3 &f : faces) {
		const Vector2 a(f.vertex[0].x, f.vertex[0].z);
		const Vector2 b(f.vertex[1].x, f.vertex[1].z);
		const Vector2 c(f.vertex[2].x, f.vertex[2].z);
		const real_t area = (b - a).cross(c - a);
		if (Math::abs(area) < (real_t)1e-8) {
			continue; // no X/Z area (degenerate, or a vertical face)
		}
		Vector<Vector2> &dst = (area * up > 0) ? tris : any_facing;
		dst.push_back(a);
		dst.push_back(b);
		dst.push_back(c);
	}
	if (tris.is_empty()) {
		return any_facing;
	}
	return tris;
}

PackedByteArray WaterBody3D::_rasterize_wet_mask(const Vector<Vector2> &p_tris, Vector2 p_center, Vector2 p_domain, int p_n) {
	// A cell is wet when its center lies inside any footprint triangle -- the
	// same cell-centre convention as water_ripple.glsl's xz.
	PackedByteArray mask;
	mask.resize(p_n * p_n);
	memset(mask.ptrw(), 0, mask.size());
	uint8_t *mw = mask.ptrw();
	const Vector2 cell = p_domain / (real_t)p_n;
	const Vector2 grid_min = p_center - p_domain * 0.5f;
	for (int t = 0; t + 2 < p_tris.size(); t += 3) {
		const Vector2 a = p_tris[t], b = p_tris[t + 1], c = p_tris[t + 2];
		const Vector2 lo = a.min(b).min(c);
		const Vector2 hi = a.max(b).max(c);
		const int x0 = CLAMP((int)Math::floor((lo.x - grid_min.x) / cell.x), 0, p_n - 1);
		const int x1 = CLAMP((int)Math::floor((hi.x - grid_min.x) / cell.x), 0, p_n - 1);
		const int z0 = CLAMP((int)Math::floor((lo.y - grid_min.y) / cell.y), 0, p_n - 1);
		const int z1 = CLAMP((int)Math::floor((hi.y - grid_min.y) / cell.y), 0, p_n - 1);
		for (int z = z0; z <= z1; z++) {
			for (int x = x0; x <= x1; x++) {
				const Vector2 p = grid_min + Vector2(x + 0.5f, z + 0.5f) * cell;
				if (point_in_triangle(p, a, b, c)) {
					mw[z * p_n + x] = 1;
				}
			}
		}
	}
	return mask;
}

Ref<Mesh> WaterBody3D::_build_extended_plane(Vector2 p_inner, int p_inner_cells, real_t p_extent, real_t p_growth) const {
	// A regular grid of (2K+1)^2 vertices whose square rings are spread out
	// from the center: rings up to K0 are the simulated span at uniform
	// spacing, and each ring past it is p_growth further out than the step
	// before (lod_mesh_growth, 1.15 by default) -- detail where the waves are
	// simulated, a few hundred rings out to the horizon. Spacing grows with
	// the ring (distance from the center), the same along X and Z. (A per-axis
	// spacing left the simulated span's fine columns/rows running out to the
	// horizon as two strips of fine mesh along the axes, and the same waves
	// looked different either side of each strip's edge: a seam straight out
	// to sea.)
	const int k0 = MAX(p_inner_cells / 2, 1);
	const Vector2 half_inner = p_inner * 0.5f;
	const real_t max_half = MAX(half_inner.x, half_inner.y);
	const real_t growth = MAX(p_growth, 1.01f);
	// radius[c]: ring c's half-size, in units of the inner step (ring c <= k0: c).
	LocalVector<real_t> radius;
	for (int c = 0; c <= k0; c++) {
		radius.push_back((real_t)c);
	}
	{
		const real_t limit = (p_extent * 0.5f) / max_half * (real_t)k0;
		real_t r = (real_t)k0;
		real_t step = 1.0f;
		while (r < limit) {
			step *= growth;
			r = MIN(r + step, limit);
			radius.push_back(r);
		}
	}
	const int kmax = (int)radius.size() - 1;
	const int nx = 2 * kmax + 1;
	const int nz = nx;
	auto position = [&](int p_i, int p_j) {
		const int ai = p_i < 0 ? -p_i : p_i;
		const int aj = p_j < 0 ? -p_j : p_j;
		const int ring = ai > aj ? ai : aj;
		const real_t scale = ring == 0 ? (real_t)1.0 : radius[ring] / (real_t)ring;
		return Vector2((real_t)p_i * scale * half_inner.x / (real_t)k0, (real_t)p_j * scale * half_inner.y / (real_t)k0);
	};

	PackedVector3Array verts;
	PackedVector3Array normals;
	PackedVector2Array uvs;
	verts.resize(nx * nz);
	normals.resize(nx * nz);
	uvs.resize(nx * nz);
	for (int j = 0; j < nz; j++) {
		for (int i = 0; i < nx; i++) {
			const int k = j * nx + i;
			const Vector2 xz = position(i - kmax, j - kmax);
			verts.set(k, Vector3(xz.x, 0.0f, xz.y));
			normals.set(k, Vector3(0, 1, 0));
			uvs.set(k, xz / p_extent + Vector2(0.5f, 0.5f));
		}
	}
	const bool flip = plane_mesh_xz_winding() < 0;
	PackedInt32Array indices;
	indices.resize((nx - 1) * (nz - 1) * 6);
	int w = 0;
	for (int j = 0; j < nz - 1; j++) {
		for (int i = 0; i < nx - 1; i++) {
			const int a = j * nx + i;
			const int b = a + 1;
			const int c = a + nx;
			const int d = c + 1;
			// (a, b, d) has a positive X/Z cross product; flip to PlaneMesh's.
			const int tri[6] = { a, flip ? d : b, flip ? b : d, a, flip ? c : d, flip ? d : c };
			for (int t = 0; t < 6; t++) {
				indices.set(w++, tri[t]);
			}
		}
	}
	Ref<ArrayMesh> mesh;
	mesh.instantiate();
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = verts;
	arrays[Mesh::ARRAY_NORMAL] = normals;
	arrays[Mesh::ARRAY_TEX_UV] = uvs;
	arrays[Mesh::ARRAY_INDEX] = indices;
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	return mesh;
}

Ref<Mesh> WaterBody3D::_build_footprint_mesh(const Vector<Vector2> &p_tris, const Rect2 &p_bounds, int p_cells) const {
	// Resample the footprint into an even grid so the height fields have
	// vertices to displace everywhere (a CSG disc is a fan of long slivers
	// with no interior vertices). Cells no triangle edge passes through are
	// whole quads, kept or dropped by their center; cells an edge touches are
	// clipped against each overlapping triangle. A quad cell's edges are
	// never crossed by a triangle edge, so they carry no extra vertices and
	// meet their clipped neighbors without T-junctions.
	const real_t side = MAX(p_bounds.size.x, p_bounds.size.y);
	const real_t cs = side / (real_t)MAX(p_cells, 1);
	const int nx = MAX((int)Math::ceil(p_bounds.size.x / cs), 1);
	const int nz = MAX((int)Math::ceil(p_bounds.size.y / cs), 1);
	const Vector2 origin = p_bounds.position;

	const int tri_count = p_tris.size() / 3;
	LocalVector<LocalVector<int>> cell_tris;
	cell_tris.resize(nx * nz);
	LocalVector<uint8_t> edge_cell;
	edge_cell.resize(nx * nz);
	memset(edge_cell.ptr(), 0, edge_cell.size());
	const real_t eps = cs * (real_t)1e-4;
	auto cell_range = [&](const Vector2 &p_lo, const Vector2 &p_hi, int &r_x0, int &r_x1, int &r_z0, int &r_z1) {
		r_x0 = CLAMP((int)Math::floor((p_lo.x - eps - origin.x) / cs), 0, nx - 1);
		r_x1 = CLAMP((int)Math::floor((p_hi.x + eps - origin.x) / cs), 0, nx - 1);
		r_z0 = CLAMP((int)Math::floor((p_lo.y - eps - origin.y) / cs), 0, nz - 1);
		r_z1 = CLAMP((int)Math::floor((p_hi.y + eps - origin.y) / cs), 0, nz - 1);
	};
	for (int t = 0; t < tri_count; t++) {
		const Vector2 v[3] = { p_tris[t * 3], p_tris[t * 3 + 1], p_tris[t * 3 + 2] };
		int x0, x1, z0, z1;
		cell_range(v[0].min(v[1]).min(v[2]), v[0].max(v[1]).max(v[2]), x0, x1, z0, z1);
		for (int z = z0; z <= z1; z++) {
			for (int x = x0; x <= x1; x++) {
				cell_tris[z * nx + x].push_back(t);
			}
		}
		for (int e = 0; e < 3; e++) {
			const Vector2 p0 = v[e], p1 = v[(e + 1) % 3];
			cell_range(p0.min(p1), p0.max(p1), x0, x1, z0, z1);
			for (int z = z0; z <= z1; z++) {
				for (int x = x0; x <= x1; x++) {
					const Vector2 bmin = origin + Vector2(x, z) * cs - Vector2(eps, eps);
					const Vector2 bmax = origin + Vector2(x + 1, z + 1) * cs + Vector2(eps, eps);
					if (segment_touches_box(p0, p1, bmin, bmax)) {
						edge_cell[z * nx + x] = 1;
					}
				}
			}
		}
	}

	const real_t plane_winding = plane_mesh_xz_winding();

	PackedVector3Array verts;
	PackedVector3Array normals;
	PackedVector2Array uvs;
	PackedInt32Array indices;
	HashMap<Vector2i, int> vertex_ids;
	auto vertex_id = [&](const Vector2 &p_v) {
		const Vector2i key((int)Math::round(p_v.x * 10000.0f), (int)Math::round(p_v.y * 10000.0f));
		if (const int *found = vertex_ids.getptr(key)) {
			return *found;
		}
		const int id = verts.size();
		verts.push_back(Vector3(p_v.x, 0.0f, p_v.y));
		normals.push_back(Vector3(0, 1, 0));
		uvs.push_back((p_v - origin) / side);
		vertex_ids.insert(key, id);
		return id;
	};
	auto add_triangle = [&](const Vector2 &p_a, Vector2 p_b, Vector2 p_c) {
		const real_t area = (p_b - p_a).cross(p_c - p_a);
		if (Math::abs(area) < cs * cs * (real_t)1e-6) {
			return;
		}
		if ((area < 0) != (plane_winding < 0)) {
			SWAP(p_b, p_c);
		}
		indices.push_back(vertex_id(p_a));
		indices.push_back(vertex_id(p_b));
		indices.push_back(vertex_id(p_c));
	};

	LocalVector<Vector2> poly;
	for (int z = 0; z < nz; z++) {
		for (int x = 0; x < nx; x++) {
			const int ci = z * nx + x;
			const Vector2 c00 = origin + Vector2(x, z) * cs;
			const Vector2 c10 = origin + Vector2(x + 1, z) * cs;
			const Vector2 c11 = origin + Vector2(x + 1, z + 1) * cs;
			const Vector2 c01 = origin + Vector2(x, z + 1) * cs;
			if (!edge_cell[ci]) {
				const Vector2 center = (c00 + c11) * 0.5f;
				bool inside = false;
				for (int t : cell_tris[ci]) {
					if (point_in_triangle(center, p_tris[t * 3], p_tris[t * 3 + 1], p_tris[t * 3 + 2])) {
						inside = true;
						break;
					}
				}
				if (inside) {
					add_triangle(c00, c10, c11);
					add_triangle(c00, c11, c01);
				}
				continue;
			}
			for (int t : cell_tris[ci]) {
				Vector2 tri[3] = { p_tris[t * 3], p_tris[t * 3 + 1], p_tris[t * 3 + 2] };
				if ((tri[1] - tri[0]).cross(tri[2] - tri[0]) < 0) {
					SWAP(tri[1], tri[2]);
				}
				poly.clear();
				poly.push_back(c00);
				poly.push_back(c10);
				poly.push_back(c11);
				poly.push_back(c01);
				clip_to_triangle(poly, tri);
				for (uint32_t i = 1; i + 1 < poly.size(); i++) {
					add_triangle(poly[0], poly[i], poly[i + 1]);
				}
			}
		}
	}

	Ref<ArrayMesh> mesh;
	mesh.instantiate();
	if (indices.is_empty()) {
		return mesh;
	}
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = verts;
	arrays[Mesh::ARRAY_NORMAL] = normals;
	arrays[Mesh::ARRAY_TEX_UV] = uvs;
	arrays[Mesh::ARRAY_INDEX] = indices;
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	return mesh;
}

void WaterBody3D::submit_sphere(int p_owner, Vector3 p_world_pos, float p_radius, float p_strength) {
	solver.submit_sphere((uint64_t)p_owner, p_world_pos, p_radius, p_strength);
}

void WaterBody3D::clear_sphere(int p_owner) {
	solver.clear_sphere((uint64_t)p_owner);
}

void WaterBody3D::submit_impulse(Vector3 p_world_pos, float p_radius, float p_strength) {
	solver.submit_impulse(p_world_pos, p_radius, p_strength);
}
