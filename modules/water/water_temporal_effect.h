/**************************************************************************/
/*  water_temporal_effect.h                                               */
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

#pragma once

#include "core/os/mutex.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "scene/resources/compositor.h"

class RenderingDevice;
class World3D;

// Composites the isolated water color (the renderer's `water_temporal` pass)
// back into the frame before the normal transparent pass. The temporal
// accumulation is layered on top of this in later phases.
//
// One effect serves a whole world: WaterBody3D installs/uninstalls it while a
// temporal water surface is in the tree. A Compositor the effect creates is
// removed again; an existing one is only changed at the RenderingServer level
// (the user's resource is never modified), and its own effects are restored.
class WaterTemporalEffect : public CompositorEffect {
	GDCLASS(WaterTemporalEffect, CompositorEffect);

	RID composite_shader;
	RID sampler;
	RID sampler_clamp_linear;
	RID sampler_repeat_linear;
	RID resolve_shader;
	RID resolve_pipeline;
	RID params_buffer;
	HashMap<int64_t, RID> pipelines; // composite pipelines, per framebuffer format
	bool composite_failed = false;
	bool resolve_failed = false;
	bool _ensure_composite(RenderingDevice *p_rd);
	bool _ensure_resolve(RenderingDevice *p_rd);
	void _free_rd_resources();
	void _render(int p_callback_type, const RenderData *p_render_data);

	// Temporal state (render thread only).
	int history_index = 0;
	bool history_valid = false;
	Size2i history_size;
	RID last_ripple_height;
	RID last_ocean_height;
	bool have_previous_camera = false;
	Projection previous_view_projection;

	// Surface parameters published by the water body (main thread), read while
	// resolving (render thread).
	static Mutex state_mutex;
	static float surface_y;
	static bool surface_ready;
	struct WaterData {
		RID ripple_height;
		RID ocean_height;
		RID ripple_height_prev;
		RID ocean_height_prev;
		RID ocean_fade;
		Vector2 grid_center;
		Vector2 ripple_domain;
		Vector2 ocean_domain;
		float node_y = 0.0f;
		float water_level = 0.0f;
		bool valid = false;
	};
	static WaterData water_data;

	struct WorldState {
		Ref<Compositor> created; // ours, set on the world while it has none
		LocalVector<RID> own_effects; // the user's effects, restored on uninstall
		ObjectID compositor;
		int users = 0;
	};
	static HashMap<ObjectID, WorldState> worlds;
	static Ref<WaterTemporalEffect> shared;

protected:
	static void _bind_methods();

public:
	// Keeps the effect in the world's frame; call every frame while a temporal
	// water body may be visible (cheap when nothing changed, and it follows the
	// world/camera compositor if the user swaps it).
	static void install(World3D *p_world);
	static void uninstall(World3D *p_world);
	static void uninstall(ObjectID p_world_id);
	static void refresh(World3D *p_world);
	static void free_shared();
	static Ref<WaterTemporalEffect> get_shared();

	// Called by WaterBody3D every rendered frame while temporal is on.
	static void set_surface(float p_surface_y);
	static void set_water_data(const RID &p_ripple_height, const RID &p_ocean_height, const RID &p_ripple_height_prev, const RID &p_ocean_height_prev, const RID &p_ocean_fade, const Vector2 &p_grid_center, const Vector2 &p_ripple_domain, const Vector2 &p_ocean_domain, float p_node_y, float p_water_level);

	WaterTemporalEffect();
	~WaterTemporalEffect();
};
