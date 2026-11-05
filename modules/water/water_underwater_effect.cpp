/**************************************************************************/
/*  water_underwater_effect.cpp                                           */
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

#include "water_underwater_effect.h"

#include "water_body_3d.h"

#include "core/object/class_db.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/world_environment.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "scene/main/window.h"
#include "scene/resources/environment.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"

void WaterUnderwaterEffect::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_water_body_path", "path"), &WaterUnderwaterEffect::set_water_body_path);
	ClassDB::bind_method(D_METHOD("get_water_body_path"), &WaterUnderwaterEffect::get_water_body_path);
	ClassDB::bind_method(D_METHOD("set_camera_path", "path"), &WaterUnderwaterEffect::set_camera_path);
	ClassDB::bind_method(D_METHOD("get_camera_path"), &WaterUnderwaterEffect::get_camera_path);
	ClassDB::bind_method(D_METHOD("set_water_color", "color"), &WaterUnderwaterEffect::set_water_color);
	ClassDB::bind_method(D_METHOD("get_water_color"), &WaterUnderwaterEffect::get_water_color);
	ClassDB::bind_method(D_METHOD("set_fog_density", "density"), &WaterUnderwaterEffect::set_fog_density);
	ClassDB::bind_method(D_METHOD("get_fog_density"), &WaterUnderwaterEffect::get_fog_density);
	ClassDB::bind_method(D_METHOD("set_brightness", "brightness"), &WaterUnderwaterEffect::set_brightness);
	ClassDB::bind_method(D_METHOD("get_brightness"), &WaterUnderwaterEffect::get_brightness);
	ClassDB::bind_method(D_METHOD("set_contrast", "contrast"), &WaterUnderwaterEffect::set_contrast);
	ClassDB::bind_method(D_METHOD("get_contrast"), &WaterUnderwaterEffect::get_contrast);
	ClassDB::bind_method(D_METHOD("set_saturation", "saturation"), &WaterUnderwaterEffect::set_saturation);
	ClassDB::bind_method(D_METHOD("get_saturation"), &WaterUnderwaterEffect::get_saturation);
	ClassDB::bind_method(D_METHOD("set_transition_speed", "speed"), &WaterUnderwaterEffect::set_transition_speed);
	ClassDB::bind_method(D_METHOD("get_transition_speed"), &WaterUnderwaterEffect::get_transition_speed);
	ClassDB::bind_method(D_METHOD("get_submersion"), &WaterUnderwaterEffect::get_submersion);
	ClassDB::bind_method(D_METHOD("is_submerged"), &WaterUnderwaterEffect::is_submerged);

	ADD_SIGNAL(MethodInfo("submerged_changed", PropertyInfo(Variant::BOOL, "submerged")));

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "water_body_path", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "WaterBody3D"), "set_water_body_path", "get_water_body_path");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "camera_path", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Camera3D"), "set_camera_path", "get_camera_path");
	ADD_GROUP("Underwater", "");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "water_color"), "set_water_color", "get_water_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fog_density", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater"), "set_fog_density", "get_fog_density");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "brightness", PROPERTY_HINT_RANGE, "0,2,0.01"), "set_brightness", "get_brightness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "contrast", PROPERTY_HINT_RANGE, "0,2,0.01"), "set_contrast", "get_contrast");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "saturation", PROPERTY_HINT_RANGE, "0,2,0.01"), "set_saturation", "get_saturation");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "transition_speed", PROPERTY_HINT_RANGE, "0.1,30,0.1,or_greater"), "set_transition_speed", "get_transition_speed");
}

void WaterUnderwaterEffect::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			set_process(true);
			_resolve_nodes();
			_install();
		} break;
		case NOTIFICATION_EXIT_TREE: {
			_uninstall();
		} break;
		case NOTIFICATION_PROCESS: {
			_update_underwater(get_process_delta_time());
		} break;
	}
}

WaterBody3D *WaterUnderwaterEffect::_get_water() const {
	return ObjectDB::get_instance<WaterBody3D>(water_id);
}

WorldEnvironment *WaterUnderwaterEffect::_get_world_environment() const {
	return ObjectDB::get_instance<WorldEnvironment>(world_environment_id);
}

void WaterUnderwaterEffect::_resolve_nodes() {
	// Setting a property on a scene node happens before it enters the tree:
	// get_tree() asserts there, and a sibling may not be there yet either.
	// Entering the tree and _update_underwater() retry.
	if (!is_inside_tree()) {
		return;
	}
	if (_get_water() == nullptr && !water_body_path.is_empty()) {
		WaterBody3D *found = Object::cast_to<WaterBody3D>(get_node_or_null(water_body_path));
		water_id = found != nullptr ? found->get_instance_id() : ObjectID();
	}
	if (_get_world_environment() == nullptr) {
		Node *root = get_tree() != nullptr ? get_tree()->get_root() : nullptr;
		if (root != nullptr) {
			TypedArray<Node> environments = root->find_children("*", "WorldEnvironment", true, false);
			if (!environments.is_empty()) {
				WorldEnvironment *found = Object::cast_to<WorldEnvironment>(environments[0]);
				world_environment_id = found != nullptr ? found->get_instance_id() : ObjectID();
			}
		}
	}
}

void WaterUnderwaterEffect::_install() {
	WorldEnvironment *world_environment = _get_world_environment();
	if (installed || world_environment == nullptr) {
		return;
	}
	base_environment = world_environment->get_environment();
	if (base_environment.is_null()) {
		if (!quiet) {
			WARN_PRINT("WaterUnderwaterEffect: the WorldEnvironment has no Environment resource to grade.");
		}
		return;
	}
	Ref<Resource> copy = base_environment->duplicate();
	underwater_environment = Object::cast_to<Environment>(copy.ptr());
	if (underwater_environment.is_null()) {
		return;
	}
	base_fog_enabled = base_environment->is_fog_enabled();
	base_fog_color = base_environment->get_fog_light_color();
	base_fog_density = base_environment->get_fog_density();
	base_fog_sky_affect = base_environment->get_fog_sky_affect();
	base_adjustment_enabled = base_environment->is_adjustment_enabled();
	base_brightness = base_environment->get_adjustment_brightness();
	base_contrast = base_environment->get_adjustment_contrast();
	base_saturation = base_environment->get_adjustment_saturation();
	world_environment->set_environment(underwater_environment);
	installed = true;
}

void WaterUnderwaterEffect::_uninstall() {
	if (!installed) {
		return;
	}
	if (WorldEnvironment *world_environment = _get_world_environment()) {
		world_environment->set_environment(base_environment);
	}
	underwater_environment = Ref<Environment>();
	installed = false;
}

Camera3D *WaterUnderwaterEffect::_get_camera() const {
	if (!camera_path.is_empty()) {
		return Object::cast_to<Camera3D>(get_node_or_null(camera_path));
	}
	Viewport *viewport = get_viewport();
	return viewport != nullptr ? viewport->get_camera_3d() : nullptr;
}

void WaterUnderwaterEffect::_update_underwater(float p_delta) {
	WaterBody3D *water = _get_water();
	if (water == nullptr || _get_world_environment() == nullptr) {
		_resolve_nodes();
		water = _get_water();
	}
	if (water == nullptr || _get_world_environment() == nullptr) {
		// Nodes may stream in after this one; warn only if it never resolves.
		if (!warned_setup && !quiet && ++resolve_attempts > 120) {
			warned_setup = true;
			WARN_PRINT(_get_water() == nullptr
							? "WaterUnderwaterEffect: water_body_path is not set or does not point at a WaterBody3D; underwater grade stays off."
							: "WaterUnderwaterEffect: no WorldEnvironment in the scene to grade; underwater grade stays off.");
		}
		return;
	}
	if (!installed) {
		_install();
		if (!installed) {
			return;
		}
	}
	Camera3D *camera = _get_camera();
	if (camera == nullptr) {
		return;
	}
	const Vector3 position = camera->get_global_position();
	// sample_height() is -INF over dry cells, so this is false outside the
	// water body as well as above it.
	const bool underwater = water->is_wet(position) && position.y < water->sample_height(position);
	submersion = Math::move_toward(submersion, underwater ? 1.0f : 0.0f, p_delta * transition_speed);
	_apply_grade(submersion);
	const bool submerged = is_submerged();
	if (submerged != emitted_submerged) {
		emitted_submerged = submerged;
		emit_signal(SNAME("submerged_changed"), submerged);
	}
}

void WaterUnderwaterEffect::_apply_grade(float p_amount) {
	if (underwater_environment.is_null()) {
		return;
	}
	const float t = CLAMP(p_amount, 0.0f, 1.0f);
	underwater_environment->set_fog_enabled(base_fog_enabled || t > 0.001f);
	underwater_environment->set_fog_light_color(base_fog_color.lerp(water_color, t));
	underwater_environment->set_fog_density(Math::lerp(base_fog_density, fog_density, t));
	underwater_environment->set_fog_sky_affect(Math::lerp(base_fog_sky_affect, 0.0f, t));
	underwater_environment->set_adjustment_enabled(base_adjustment_enabled || t > 0.001f);
	underwater_environment->set_adjustment_brightness(Math::lerp(base_brightness, brightness, t));
	underwater_environment->set_adjustment_contrast(Math::lerp(base_contrast, contrast, t));
	underwater_environment->set_adjustment_saturation(Math::lerp(base_saturation, saturation, t));
}

void WaterUnderwaterEffect::set_water_body_path(const NodePath &p_path) {
	water_body_path = p_path;
	water_id = ObjectID();
	resolve_attempts = 0;
	_resolve_nodes();
}

void WaterUnderwaterEffect::set_camera_path(const NodePath &p_path) {
	camera_path = p_path;
}

void WaterUnderwaterEffect::set_water_color(const Color &p_color) {
	water_color = p_color;
}

void WaterUnderwaterEffect::set_fog_density(float p_density) {
	fog_density = MAX(p_density, 0.0f);
}

void WaterUnderwaterEffect::set_brightness(float p_brightness) {
	brightness = MAX(p_brightness, 0.0f);
}

void WaterUnderwaterEffect::set_contrast(float p_contrast) {
	contrast = MAX(p_contrast, 0.0f);
}

void WaterUnderwaterEffect::set_saturation(float p_saturation) {
	saturation = MAX(p_saturation, 0.0f);
}

void WaterUnderwaterEffect::set_transition_speed(float p_speed) {
	transition_speed = MAX(p_speed, 0.0f);
}
