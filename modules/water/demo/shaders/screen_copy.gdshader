shader_type spatial;
// 1:1 screen-color copy probe. If the engine's screen-texture capture is
// correct, this quad is invisible: it reproduces exactly what is behind it.
// A washed-out, white or shifted quad means the capture path is broken.
render_mode unshaded, blend_mix, cull_disabled, depth_draw_never;

uniform sampler2D screen_copy : hint_screen_texture, filter_nearest;

void fragment() {
	ALBEDO = texture(screen_copy, SCREEN_UV).rgb;
	ALPHA = 1.0;
}
