#ifndef uv_camera_h
#define uv_camera_h

#include <cstdlib>




#include "primitives.h"
#include "matrix_utils.h"



// UV camera
//
// latitude:     | longitude:    | radius:       |
//       *_*_    |        ___    |        ___    |
//      */   \   |       /   \   |       /   \   |
// u:  *|  x  |  |  v:  |**x**|  |  w:  |  x**|  |
//      *\___/   |       \___/   |       \___/   |
//       * *     |               |               |
// 

class uv_camera
{
public:
	// Use as read-only
	float u, v, w, fov;
	int win_x, win_y;
	vertex_3 eye, look_at, up;
	float near_plane;
	float far_plane;

	float model_mat[16];
	float view_mat[16];
	float projection_mat[16];

public:
	uv_camera(void);
	void calculate_camera_matrices(const int width_px, const int height_px);

protected:
	void transform(void);
	void reset(void);
	void rotate(void);
	void translate(void);
};


#endif
