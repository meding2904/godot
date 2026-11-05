shader_type spatial;
// Seabed material that receives the water's caustics: the light pattern the
// WaterBody3D projects along the sun direction, tinted onto the sand.
render_mode diffuse_burley, specular_schlick_ggx;

uniform vec4 sand_color : source_color = vec4(0.78, 0.70, 0.55, 1.0);
uniform float sand_roughness : hint_range(0.0, 1.0) = 0.9;
uniform sampler2D caustics_tex : repeat_enable, filter_linear, hint_default_black;
uniform vec3 caustics_origin = vec3(0.0);
uniform vec3 caustics_right = vec3(1.0, 0.0, 0.0);
uniform vec3 caustics_up = vec3(0.0, 0.0, 1.0);
uniform float caustics_extent = 20.0;
uniform float caustics_strength : hint_range(0.0, 8.0) = 2.0;
uniform vec4 caustics_color : source_color = vec4(0.55, 0.8, 0.85, 1.0);

void fragment() {
	vec3 world_pos = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
	ALBEDO = sand_color.rgb;
	ROUGHNESS = sand_roughness;
	vec2 p = vec2(dot(world_pos - caustics_origin, caustics_right), dot(world_pos - caustics_origin, caustics_up));
	vec2 uv = p / (2.0 * caustics_extent) + 0.5;
	vec4 c = texture(caustics_tex, uv);
	EMISSION = caustics_color.rgb * (c.r * c.g * caustics_strength);
}
