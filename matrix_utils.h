#ifndef __matrix_utils__
#define __matrix_utils__

#include <cstdio>
#include "primitives.h"



void get_model_matrix(float (&in_a)[16]);
void get_view_matrix(float eyex, float eyey, float eyez, float centrex, float centrey, float centrez, float upx, float upy, float upz, float (&mat)[16]);
void get_projection_matrix(float fovy, float aspect, float znear, float zfar, float (&in_a)[16]);
void multiply_4x4_matrices(float (&in_a)[16], float (&in_b)[16], float (&out)[16]);
void init_perspective_camera(float fovy_degrees, float aspect, float znear, float zfar,
                             float eyex, float eyey, float eyez, float centrex, float centrey,
                             float centrez, float upx, float upy, float upz,
                             float (&model_matrix)[16],
							 float (&view_matrix)[16],
							 float (&projection_matrix)[16]);


#endif