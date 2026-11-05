/**************************************************************************/
/*  water_underwater_effect.h                                             */
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

#include "scene/3d/node_3d.h"

class Camera3D;
class Environment;
class WaterBody3D;
class WorldEnvironment;

// Underwater rendering, without a post-process pass: when the camera goes
// under a WaterBody3D, this node swaps the WorldEnvironment's fog and color
// adjustment for an underwater preset, interpolated so crossing the surface
// eases instead of popping. Restores the original environment when it leaves
// the tree.
//
// Cheap and renderer-agnostic on purpose (plain Environment properties, no
// screen copy, no compute), which also means it composes with the water
// surface's own below-water shading. On its own it grades the whole frame,
// including the surface seen from below -- that is what being underwater
// looks like; the surface's Snell's-window/TIR shading comes from
// WaterBody3D's material.
class WaterUnderwaterEffect : public Node3D {
	GDCLASS(WaterUnderwaterEffect, Node3D);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void set_water_body_path(const NodePath &p_path);
	NodePath get_water_body_path() const { return water_body_path; }
	void set_camera_path(const NodePath &p_path);
	NodePath get_camera_path() const { return camera_path; }
	// Fog color underwater (the water body's own tint).
	void set_water_color(const Color &p_color);
	Color get_water_color() const { return water_color; }
	void set_fog_density(float p_density);
	float get_fog_density() const { return fog_density; }
	// Grade applied as the camera goes under (1 = unchanged).
	void set_brightness(float p_brightness);
	float get_brightness() const { return brightness; }
	void set_contrast(float p_contrast);
	float get_contrast() const { return contrast; }
	void set_saturation(float p_saturation);
	float get_saturation() const { return saturation; }
	// 1/s: how fast the grade follows the camera across the surface.
	void set_transition_speed(float p_speed);
	float get_transition_speed() const { return transition_speed; }

	// 0 above water, 1 fully under, smoothed. is_submerged() is the binary
	// state after smoothing (for audio/fx hooks).
	float get_submersion() const { return submersion; }
	bool is_submerged() const { return submersion > 0.5f; }

	// Internal: the hidden instance WaterBody3D adds for out-of-the-box
	// grading stays quiet when the scene has nothing to grade.
	void set_quiet(bool p_quiet) { quiet = p_quiet; }

private:
	NodePath water_body_path;
	NodePath camera_path;
	Color water_color = Color(0.03f, 0.22f, 0.26f);
	float fog_density = 0.08f;
	float brightness = 0.85f;
	float contrast = 1.05f;
	float saturation = 0.8f;
	float transition_speed = 4.0f;
	float submersion = 0.0f;

	// Weak references: either node can be freed before this one exits.
	ObjectID water_id;
	ObjectID world_environment_id;
	Ref<Environment> base_environment;
	Ref<Environment> underwater_environment;
	bool installed = false;
	bool warned_setup = false;
	bool quiet = false;
	bool emitted_submerged = false;
	int resolve_attempts = 0;

	// The WorldEnvironment's values captured when installed.
	bool base_fog_enabled = false;
	Color base_fog_color = Color(0.6f, 0.7f, 0.8f);
	float base_fog_density = 0.0f;
	float base_fog_sky_affect = 0.0f;
	bool base_adjustment_enabled = false;
	float base_brightness = 1.0f;
	float base_contrast = 1.0f;
	float base_saturation = 1.0f;

	void _resolve_nodes();
	void _install();
	void _uninstall();
	void _update_underwater(float p_delta);
	void _apply_grade(float p_amount);
	WaterBody3D *_get_water() const;
	WorldEnvironment *_get_world_environment() const;
	Camera3D *_get_camera() const;
};
