/**************************************************************************/
/*  water_temporal_effect.cpp                                             */
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

#include "water_temporal_effect.h"

#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "scene/resources/3d/world_3d.h"
#include "servers/rendering/renderer_rd/framebuffer_cache_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_data_rd.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_device_binds.h"
#include "servers/rendering/rendering_server.h"
#include "servers/rendering/storage/render_data.h"

HashMap<ObjectID, WaterTemporalEffect::WorldState> WaterTemporalEffect::worlds;
Ref<WaterTemporalEffect> WaterTemporalEffect::shared;
Mutex WaterTemporalEffect::state_mutex;
float WaterTemporalEffect::surface_y = 0.0f;
bool WaterTemporalEffect::surface_ready = false;
WaterTemporalEffect::WaterData WaterTemporalEffect::water_data;

namespace {

// Composites the (accumulated) water target over the frame before the normal
// transparent pass. The water target is premultiplied (it was blended over
// transparent black), so one * src + (1 - src.a) * dst.
const char *COMPOSITE_SHADER = R"(
#[vertex]
#version 450
void main() {
	vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
#[fragment]
#version 450
layout(set = 0, binding = 0) uniform sampler2D water;
layout(location = 0) out vec4 frag;
void main() {
	frag = texelFetch(water, ivec2(gl_FragCoord.xy), 0);
}
)";

// Accumulates the isolated water color. Phase 1 reprojects on the still-water
// plane (waves and ripples are not in the velocity yet); the neighborhood
// clamp and the alpha test keep boats/terrain edges from dragging history in.
const char *RESOLVE_SHADER = R"(
#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D current_water;
layout(set = 0, binding = 1) uniform sampler2D history_water;
layout(set = 0, binding = 2, rgba16f) uniform restrict writeonly image2D output_water;
layout(set = 0, binding = 3, rgba16f) uniform restrict writeonly image2D output_history;
layout(set = 0, binding = 4, std140) uniform Params {
	mat4 inv_view_projection;
	mat4 prev_view_projection;
	vec4 camera_position;
	vec4 settings; // x surface_y, y history weight, z history valid
	vec4 grid_center_ripple_domain;
	vec4 ocean_domain_levels; // xy ocean domain, z node y, w water level
} params;

layout(set = 0, binding = 5) uniform sampler2D ripple_height;
layout(set = 0, binding = 6) uniform sampler2D ocean_height;
layout(set = 0, binding = 7) uniform sampler2D ocean_fade;
layout(set = 0, binding = 8) uniform sampler2D ripple_height_prev;
layout(set = 0, binding = 9) uniform sampler2D ocean_height_prev;

float water_height(vec2 xz, bool prev) {
	vec2 rel = xz - params.grid_center_ripple_domain.xy;
	vec2 cuv_r = clamp(rel / params.grid_center_ripple_domain.zw + 0.5, vec2(0.0), vec2(1.0));
	vec2 cuv_o = xz / params.ocean_domain_levels.xy + 0.5;
	float inside = 1.0 - smoothstep(0.46, 0.5, max(abs(rel.x / params.grid_center_ripple_domain.z), abs(rel.y / params.grid_center_ripple_domain.w)));
	float ripple = prev ? texture(ripple_height_prev, cuv_r).r : texture(ripple_height, cuv_r).r;
	float ocean = prev ? texture(ocean_height_prev, cuv_o).r : texture(ocean_height, cuv_o).r;
	float level = params.ocean_domain_levels.w;
	return params.ocean_domain_levels.z + mix(level, ripple, inside) + ocean * texture(ocean_fade, cuv_r).r;
}

// The surface point at uv, lifted onto the height field (2 fixed-point steps).
// The ray is built from two points through the inverse view-projection, which
// also covers orthographic cameras (the editor's top/front/side views).
vec3 surface_at(vec2 uv) {
	vec4 near = params.inv_view_projection * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
	vec4 far = params.inv_view_projection * vec4(uv * 2.0 - 1.0, 0.0, 1.0);
	vec3 origin = near.xyz / near.w;
	vec3 ray = far.xyz / far.w - origin;
	vec3 hit = origin;
	if (abs(ray.y) < 1e-6) {
		return hit;
	}
	float t = (params.ocean_domain_levels.z + params.ocean_domain_levels.w - origin.y) / ray.y;
	hit = origin + ray * t;
	for (int i = 0; i < 2; i++) {
		t = (water_height(hit.xz, false) - origin.y) / ray.y;
		hit = origin + ray * t;
	}
	return hit;
}

void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(output_water);
	if (p.x >= size.x || p.y >= size.y) {
		return;
	}
	vec4 current = texelFetch(current_water, p, 0);
	float history_weight = params.settings.y * step(0.5, params.settings.z);
	vec4 result = current;

	if (history_weight > 0.0) {
		vec2 uv = (vec2(p) + 0.5) / vec2(size);
		vec3 hit = surface_at(uv);
		// The same water column one frame ago: its height moved with the waves,
		// so reproject the previous height, not the current one.
		vec3 prev_hit = vec3(hit.x, water_height(hit.xz, true), hit.z);
		vec4 prev_clip = params.prev_view_projection * vec4(prev_hit, 1.0);
		if (prev_clip.w > 0.0) {
			vec2 prev_uv = prev_clip.xy / prev_clip.w * 0.5 + 0.5;
			if (all(greaterThanEqual(prev_uv, vec2(0.0))) && all(lessThanEqual(prev_uv, vec2(1.0)))) {
				ivec2 hp = clamp(ivec2(prev_uv * vec2(size)), ivec2(0), size - 1);
				vec4 history = texelFetch(history_water, hp, 0);

				// Clamp history to the current 3x3 neighborhood so a
				// disoccluded or moving edge cannot smear.
				vec4 lo = current;
				vec4 hi = current;
				for (int y = -1; y <= 1; y++) {
					for (int x = -1; x <= 1; x++) {
						vec4 c = texelFetch(current_water, clamp(p + ivec2(x, y), ivec2(0), size - 1), 0);
						lo = min(lo, c);
						hi = max(hi, c);
					}
				}
				history = clamp(history, lo, hi);
				// Empty water in the history means nothing was there (or the
				// waterline moved): trust the current frame.
				history_weight *= step(0.01, history.a);
				result = mix(current, history, history_weight);
			}
		}
	}

	imageStore(output_water, p, result);
	imageStore(output_history, p, result);
}
)";

LocalVector<RID> compositor_own_effects(const Ref<Compositor> &p_compositor) {
	LocalVector<RID> rids;
	const TypedArray<CompositorEffect> effects = p_compositor->get_compositor_effects();
	for (int i = 0; i < effects.size(); i++) {
		Ref<CompositorEffect> effect = effects[i];
		if (effect.is_valid()) {
			rids.push_back(effect->get_rid());
		}
	}
	return rids;
}

bool same_rids(const LocalVector<RID> &p_a, const LocalVector<RID> &p_b) {
	if (p_a.size() != p_b.size()) {
		return false;
	}
	for (uint32_t i = 0; i < p_a.size(); i++) {
		if (p_a[i] != p_b[i]) {
			return false;
		}
	}
	return true;
}

void compositor_push(const Ref<Compositor> &p_compositor, const LocalVector<RID> &p_own, const RID &p_ours) {
	TypedArray<RID> rids;
	for (const RID &rid : p_own) {
		rids.push_back(rid);
	}
	rids.push_back(p_ours);
	RenderingServer::get_singleton()->compositor_set_compositor_effects(p_compositor->get_rid(), rids);
}

void projection_to_float(const Projection &p_projection, float *r_out) {
	for (int c = 0; c < 4; c++) {
		for (int r = 0; r < 4; r++) {
			r_out[c * 4 + r] = (float)p_projection.columns[c][r];
		}
	}
}

struct ResolveParams {
	float inv_view_projection[16];
	float prev_view_projection[16];
	float camera_position[4];
	float settings[4];
	float grid_center_ripple_domain[4];
	float ocean_domain_levels[4];
};

} // namespace

void WaterTemporalEffect::_bind_methods() {
}

WaterTemporalEffect::WaterTemporalEffect() {
	RenderingServer::get_singleton()->compositor_effect_set_callback(get_rid(), RSE::COMPOSITOR_EFFECT_CALLBACK_TYPE_PRE_TRANSPARENT, callable_mp(this, &WaterTemporalEffect::_render));
}

WaterTemporalEffect::~WaterTemporalEffect() {
	// _free_rd_resources() runs from uninstall() while the device is alive;
	// the destructor itself may run after the rendering server is gone.
	pipelines.clear();
}

void WaterTemporalEffect::_free_rd_resources() {
	RenderingServer *rs = RenderingServer::get_singleton();
	RenderingDevice *rd = rs != nullptr ? rs->get_rendering_device() : nullptr;
	if (rd == nullptr) {
		return;
	}
	for (const KeyValue<int64_t, RID> &pipeline : pipelines) {
		rd->free_rid(pipeline.value);
	}
	pipelines.clear();
	if (composite_shader.is_valid()) {
		rd->free_rid(composite_shader);
		composite_shader = RID();
	}
	if (resolve_pipeline.is_valid()) {
		rd->free_rid(resolve_pipeline);
		resolve_pipeline = RID();
	}
	if (resolve_shader.is_valid()) {
		rd->free_rid(resolve_shader);
		resolve_shader = RID();
	}
	if (params_buffer.is_valid()) {
		rd->free_rid(params_buffer);
		params_buffer = RID();
	}
	if (sampler.is_valid()) {
		rd->free_rid(sampler);
		sampler = RID();
	}
	if (sampler_clamp_linear.is_valid()) {
		rd->free_rid(sampler_clamp_linear);
		sampler_clamp_linear = RID();
	}
	if (sampler_repeat_linear.is_valid()) {
		rd->free_rid(sampler_repeat_linear);
		sampler_repeat_linear = RID();
	}
}

bool WaterTemporalEffect::_ensure_composite(RenderingDevice *p_rd) {
	if (composite_shader.is_valid()) {
		return true;
	}
	if (composite_failed) {
		return false;
	}
	Ref<RDShaderFile> file;
	file.instantiate();
	if (file->parse_versions_from_text(COMPOSITE_SHADER) != OK) {
		composite_failed = true;
		ERR_FAIL_V_MSG(false, "WaterTemporalEffect: composite shader failed to compile.");
	}
	composite_shader = p_rd->shader_create_from_spirv(file->get_spirv_stages());
	sampler = p_rd->sampler_create(RD::SamplerState());
	composite_failed = !composite_shader.is_valid() || !sampler.is_valid();
	ERR_FAIL_COND_V_MSG(composite_failed, false, "WaterTemporalEffect: composite shader failed to compile.");
	return true;
}

bool WaterTemporalEffect::_ensure_resolve(RenderingDevice *p_rd) {
	if (resolve_shader.is_valid()) {
		return true;
	}
	if (resolve_failed) {
		return false;
	}
	Ref<RDShaderFile> file;
	file.instantiate();
	if (file->parse_versions_from_text(RESOLVE_SHADER) != OK) {
		resolve_failed = true;
		ERR_FAIL_V_MSG(false, "WaterTemporalEffect: resolve shader failed to compile.");
	}
	resolve_shader = p_rd->shader_create_from_spirv(file->get_spirv_stages());
	if (resolve_shader.is_valid()) {
		resolve_pipeline = p_rd->compute_pipeline_create(resolve_shader);
	}
	params_buffer = p_rd->uniform_buffer_create(sizeof(ResolveParams));
	RD::SamplerState linear;
	linear.mag_filter = RD::SAMPLER_FILTER_LINEAR;
	linear.min_filter = RD::SAMPLER_FILTER_LINEAR;
	linear.mip_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler_clamp_linear = p_rd->sampler_create(linear);
	linear.repeat_u = RD::SAMPLER_REPEAT_MODE_REPEAT;
	linear.repeat_v = RD::SAMPLER_REPEAT_MODE_REPEAT;
	sampler_repeat_linear = p_rd->sampler_create(linear);
	resolve_failed = !resolve_shader.is_valid() || !resolve_pipeline.is_valid() || !params_buffer.is_valid() || !sampler_clamp_linear.is_valid() || !sampler_repeat_linear.is_valid();
	ERR_FAIL_COND_V_MSG(resolve_failed, false, "WaterTemporalEffect: resolve shader failed to compile.");
	return true;
}

void WaterTemporalEffect::set_surface(float p_surface_y) {
	MutexLock lock(state_mutex);
	surface_y = p_surface_y;
	surface_ready = true;
}

void WaterTemporalEffect::set_water_data(const RID &p_ripple_height, const RID &p_ocean_height, const RID &p_ripple_height_prev, const RID &p_ocean_height_prev, const RID &p_ocean_fade, const Vector2 &p_grid_center, const Vector2 &p_ripple_domain, const Vector2 &p_ocean_domain, float p_node_y, float p_water_level) {
	MutexLock lock(state_mutex);
	water_data.ripple_height = p_ripple_height;
	water_data.ocean_height = p_ocean_height;
	water_data.ripple_height_prev = p_ripple_height_prev;
	water_data.ocean_height_prev = p_ocean_height_prev;
	water_data.ocean_fade = p_ocean_fade;
	water_data.grid_center = p_grid_center;
	water_data.ripple_domain = p_ripple_domain;
	water_data.ocean_domain = p_ocean_domain;
	water_data.node_y = p_node_y;
	water_data.water_level = p_water_level;
	water_data.valid = p_ripple_height.is_valid() && p_ocean_height.is_valid() && p_ripple_height_prev.is_valid() && p_ocean_height_prev.is_valid() && p_ocean_fade.is_valid();
}

void WaterTemporalEffect::_render(int p_callback_type, const RenderData *p_render_data) {
	if (p_callback_type != RSE::COMPOSITOR_EFFECT_CALLBACK_TYPE_PRE_TRANSPARENT || p_render_data == nullptr) {
		return;
	}
	RenderSceneBuffersRD *buffers = Object::cast_to<RenderSceneBuffersRD>(p_render_data->get_render_scene_buffers().ptr());
	RenderSceneDataRD *scene = Object::cast_to<RenderSceneDataRD>(p_render_data->get_render_scene_data());
	if (buffers == nullptr || scene == nullptr) {
		return;
	}
	// The renderer only splits the water out with MSAA off and a single view.
	if (buffers->get_msaa_3d() != RSE::VIEWPORT_MSAA_DISABLED || buffers->get_view_count() > 1) {
		return;
	}
	const ObjectID view_id = buffers->get_instance_id();
	ViewState &view = view_states[view_id];
	if (!buffers->has_texture(RB_SCOPE_WATER, RB_TEX_WATER_COLOR)) {
		// No water this frame: drop the history so a later water surface does
		// not inherit a stale frame.
		view.history_valid = false;
		view.have_previous_camera = false;
		return;
	}
	RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
	if (rd == nullptr || !_ensure_composite(rd)) {
		return;
	}
	const RID water = buffers->get_texture(RB_SCOPE_WATER, RB_TEX_WATER_COLOR);
	const RID color = buffers->get_internal_texture();
	if (!water.is_valid() || !color.is_valid()) {
		return;
	}

	const Size2i size = buffers->get_internal_size();
	RID source = water;
	if (_ensure_resolve(rd) && size.x > 0 && size.y > 0) {
		// History wins the frames' ping-pong on creation/resize.
		if (!buffers->has_texture(RB_SCOPE_WATER, SNAME("history_a"))) {
			const uint32_t usage = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
			buffers->create_texture(RB_SCOPE_WATER, SNAME("history_a"), RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage);
			buffers->create_texture(RB_SCOPE_WATER, SNAME("history_b"), RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage);
			buffers->create_texture(RB_SCOPE_WATER, SNAME("temp"), RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage);
			view.history_index = 0;
			view.history_size = Size2i();
			view.history_valid = false;
			view.have_previous_camera = false;
		}
		if (view.history_size != size) {
			view.history_size = size;
			view.history_valid = false;
			view.have_previous_camera = false;
		}

		const RID history_rid = buffers->get_texture(RB_SCOPE_WATER, view.history_index == 0 ? SNAME("history_a") : SNAME("history_b"));
		const RID history_new = buffers->get_texture(RB_SCOPE_WATER, view.history_index == 0 ? SNAME("history_b") : SNAME("history_a"));
		const RID temp = buffers->get_texture(RB_SCOPE_WATER, SNAME("temp"));

		MutexLock lock(state_mutex);
		if (view.last_ripple_height != water_data.ripple_height || view.last_ocean_height != water_data.ocean_height) {
			view.last_ripple_height = water_data.ripple_height;
			view.last_ocean_height = water_data.ocean_height;
			view.history_valid = false;
			view.have_previous_camera = false;
		}
		if (surface_ready && water_data.valid) {
			const Transform3D cam_transform = scene->get_cam_transform();
			const Projection current_vp = scene->get_cam_projection() * Projection(cam_transform.affine_inverse());
			ResolveParams params;
			projection_to_float(current_vp.inverse(), params.inv_view_projection);
			projection_to_float(view.previous_view_projection, params.prev_view_projection);
			const Vector3 eye = cam_transform.origin;
			params.camera_position[0] = eye.x;
			params.camera_position[1] = eye.y;
			params.camera_position[2] = eye.z;
			params.camera_position[3] = 0.0f;
			params.settings[0] = surface_y;
			params.settings[1] = 0.85f; // history weight
			params.settings[2] = (view.history_valid && view.have_previous_camera) ? 1.0f : 0.0f;
			params.settings[3] = 0.0f;
			params.grid_center_ripple_domain[0] = water_data.grid_center.x;
			params.grid_center_ripple_domain[1] = water_data.grid_center.y;
			params.grid_center_ripple_domain[2] = water_data.ripple_domain.x;
			params.grid_center_ripple_domain[3] = water_data.ripple_domain.y;
			params.ocean_domain_levels[0] = water_data.ocean_domain.x;
			params.ocean_domain_levels[1] = water_data.ocean_domain.y;
			params.ocean_domain_levels[2] = water_data.node_y;
			params.ocean_domain_levels[3] = water_data.water_level;
			rd->buffer_update(params_buffer, 0, sizeof(ResolveParams), &params);

			const RID uniform_set = rd->uniform_set_create(
					Vector<RD::Uniform>({
							RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ sampler, water })),
							RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, Vector<RID>({ sampler, history_rid })),
							RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, temp),
							RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, history_new),
							RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 4, params_buffer),
							RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 5, Vector<RID>({ sampler_clamp_linear, water_data.ripple_height })),
							RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 6, Vector<RID>({ sampler_repeat_linear, water_data.ocean_height })),
							RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 7, Vector<RID>({ sampler_clamp_linear, water_data.ocean_fade })),
							RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 8, Vector<RID>({ sampler_clamp_linear, water_data.ripple_height_prev })),
							RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 9, Vector<RID>({ sampler_repeat_linear, water_data.ocean_height_prev })),
					}),
					resolve_shader, 0);
			if (uniform_set.is_valid()) {
				const RD::ComputeListID compute_list = rd->compute_list_begin();
				rd->compute_list_bind_compute_pipeline(compute_list, resolve_pipeline);
				rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
				rd->compute_list_dispatch(compute_list, (size.x + 7) / 8, (size.y + 7) / 8, 1);
				rd->compute_list_end();
				rd->free_rid(uniform_set);
				source = temp;
				view.history_index ^= 1;
				view.history_valid = true;
			}
			view.previous_view_projection = current_vp;
			view.have_previous_camera = true;
		}
	}

	const RID framebuffer = FramebufferCacheRD::get_singleton()->get_cache(color);
	if (!framebuffer.is_valid()) {
		return;
	}
	const int64_t format = (int64_t)rd->framebuffer_get_format(framebuffer);
	RID *pipeline = pipelines.getptr(format);
	if (pipeline == nullptr) {
		RD::PipelineColorBlendState blend = RD::PipelineColorBlendState::create_disabled();
		if (!blend.attachments.is_empty()) {
			RD::PipelineColorBlendState::Attachment &attachment = blend.attachments.write[0];
			attachment.enable_blend = true;
			attachment.color_blend_op = RD::BLEND_OP_ADD;
			attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			attachment.src_color_blend_factor = RD::BLEND_FACTOR_ONE;
			attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
			attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		}
		const RID created = rd->render_pipeline_create(composite_shader, (RD::FramebufferFormatID)format, RD::INVALID_ID, RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), blend);
		ERR_FAIL_COND(!created.is_valid());
		pipeline = &pipelines.insert(format, created)->value;
	}

	const RID uniform_set = rd->uniform_set_create(Vector<RD::Uniform>({ RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ sampler, source })) }), composite_shader, 0);
	ERR_FAIL_COND(!uniform_set.is_valid());
	const RD::DrawListID draw_list = rd->draw_list_begin(framebuffer);
	rd->draw_list_bind_render_pipeline(draw_list, *pipeline);
	rd->draw_list_bind_uniform_set(draw_list, uniform_set, 0);
	rd->draw_list_draw(draw_list, false, 1, 3);
	rd->draw_list_end();
	rd->free_rid(uniform_set);
}

void WaterTemporalEffect::install(World3D *p_world) {
	ERR_FAIL_NULL(p_world);
	const ObjectID world_id = p_world->get_instance_id();
	WorldState &state = worlds[world_id];
	state.users++;
	refresh(p_world);
}

void WaterTemporalEffect::refresh(World3D *p_world) {
	if (p_world == nullptr) {
		return;
	}
	WorldState *state = worlds.getptr(p_world->get_instance_id());
	if (state == nullptr) {
		return;
	}
	Ref<Compositor> compositor = p_world->get_compositor();
	if (compositor.is_null()) {
		if (state->created.is_null()) {
			state->created.instantiate();
		}
		p_world->set_compositor(state->created);
		compositor = state->created;
	}
	// Follow a compositor the user swapped in, or effects they added/removed:
	// the effect lives at the RenderingServer level, so the resource itself is
	// never edited, and its own effects are restored on uninstall.
	const LocalVector<RID> own = compositor_own_effects(compositor);
	if (state->compositor != compositor->get_instance_id() || !same_rids(own, state->own_effects)) {
		state->own_effects = own;
		state->compositor = compositor->get_instance_id();
		compositor_push(compositor, state->own_effects, get_shared()->get_rid());
	}
}

void WaterTemporalEffect::uninstall(World3D *p_world) {
	if (p_world != nullptr) {
		uninstall(p_world->get_instance_id());
	}
}

void WaterTemporalEffect::uninstall(ObjectID p_world_id) {
	const ObjectID world_id = p_world_id;
	WorldState *state = worlds.getptr(world_id);
	if (state == nullptr) {
		return;
	}
	if (state->users > 1) {
		state->users--;
		return;
	}

	// The world/compositor may already be gone at scene teardown: restore only
	// what still exists, but always drop the entry and release the effect.
	World3D *world = Object::cast_to<World3D>(ObjectDB::get_instance(world_id));
	Compositor *compositor = Object::cast_to<Compositor>(ObjectDB::get_instance(state->compositor));
	if (compositor != nullptr) {
		TypedArray<RID> rids;
		for (const RID &rid : state->own_effects) {
			rids.push_back(rid);
		}
		RenderingServer::get_singleton()->compositor_set_compositor_effects(compositor->get_rid(), rids);
	}
	if (world != nullptr && state->created.is_valid() && world->get_compositor() == state->created) {
		world->set_compositor(Ref<Compositor>());
	}
	worlds.erase(world_id);
	if (worlds.is_empty() && shared.is_valid()) {
		// Scene teardown still has a live rendering device: release the
		// effect (and its RIDs) here, not from the module destructor.
		shared->_free_rd_resources();
		shared.unref();
	}
}

void WaterTemporalEffect::free_shared() {
	for (const KeyValue<ObjectID, WorldState> &entry : worlds) {
		World3D *world = Object::cast_to<World3D>(ObjectDB::get_instance(entry.key));
		if (world != nullptr) {
			Compositor *compositor = Object::cast_to<Compositor>(ObjectDB::get_instance(entry.value.compositor));
			if (compositor != nullptr) {
				TypedArray<RID> rids;
				for (const RID &rid : entry.value.own_effects) {
					rids.push_back(rid);
				}
				RenderingServer::get_singleton()->compositor_set_compositor_effects(compositor->get_rid(), rids);
			}
			if (entry.value.created.is_valid() && world->get_compositor() == entry.value.created) {
				world->set_compositor(Ref<Compositor>());
			}
		}
	}
	worlds.clear();
	// Normally uninstall() already released the shared effect while the
	// rendering server was alive; nothing to do here.
}

Ref<WaterTemporalEffect> WaterTemporalEffect::get_shared() {
	if (shared.is_null()) {
		shared.instantiate();
	}
	return shared;
}
