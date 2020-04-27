// Shawn Halayka
// This code and data is in the public domain.


#include "uv_camera.h"

const float pi = 4.0f*atan(1.0f);
const float pi_half = 0.5f*pi;
const float pi_2 = 2.0f*pi;
const float epsilon = 1e-6f;



uv_camera::uv_camera(void)
{
	u = v = 0;
	w = 4;
	fov = 45.0f;
	near_plane = 0.1f;
	far_plane = 1000.0f;
	win_x = win_y = 0;

	transform();
}

void uv_camera::calculate_camera_matrices(const int width_px, const int height_px)
{
	static const float lock = epsilon * 1000.0f;

	if(u < -pi_half + lock)
		u = -pi_half + lock;
	else if(u > pi_half - lock)
		u = pi_half - lock;

	while(v < 0)
		v += pi_2;

	while(v > pi_2)
		v -= pi_2;

	if(w < 0)
		w = 0;
	else if(w > 10000)
		w = 10000;

	win_x = width_px;
	win_y = height_px;

	transform();
}

void uv_camera::transform(void)
{
	reset();
	rotate();
	translate();

	init_perspective_camera(fov, float(win_x)/float(win_y), near_plane, far_plane,
							eye.x, eye.y, eye.z, // eye position
							look_at.x, look_at.y, look_at.z, // look at position
							up.x, up.y, up.z, // up direction vector
							model_mat,
							view_mat,
							projection_mat);
}

void uv_camera::reset(void)
{
	eye.zero();
	look_at.zero();
	up.zero();

	look_at.z = -1;
	up.y = 1;
}

void uv_camera::rotate(void)
{
	// Rotate about the world x axis
	look_at.rotate_x(u);
	up.rotate_x(u);

	// Rotate about the world y axis
	look_at.rotate_y(v);
	up.rotate_y(v);
}

void uv_camera::translate(void)
{
	// Place the eye directly across the sphere from the look-at vector's "tip",
	// Then scale the sphere radius by w
	eye.x = -look_at.x*w;
	eye.y = -look_at.y*w;
	eye.z = -look_at.z*w;
}
