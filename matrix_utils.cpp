#include "matrix_utils.h"


void multiply_4x4_matrices(float (&in_a)[16], float (&in_b)[16], float (&out)[16])
{
	//   matrix layout:
	//    
	//   [0 4  8 12]
	//   [1 5  9 13]
	//   [2 6 10 14]
	//   [3 7 11 15]

	out[0] = in_a[0] * in_b[0] + in_a[4] * in_b[1] + in_a[8] * in_b[2] + in_a[12] * in_b[3];
	out[1] = in_a[1] * in_b[0] + in_a[5] * in_b[1] + in_a[9] * in_b[2] + in_a[13] * in_b[3];
	out[2] = in_a[2] * in_b[0] + in_a[6] * in_b[1] + in_a[10] * in_b[2] + in_a[14] * in_b[3];
	out[3] = in_a[3] * in_b[0] + in_a[7] * in_b[1] + in_a[11] * in_b[2] + in_a[15] * in_b[3];
	out[4] = in_a[0] * in_b[4] + in_a[4] * in_b[5] + in_a[8] * in_b[6] + in_a[12] * in_b[7];
	out[5] = in_a[1] * in_b[4] + in_a[5] * in_b[5] + in_a[9] * in_b[6] + in_a[13] * in_b[7];
	out[6] = in_a[2] * in_b[4] + in_a[6] * in_b[5] + in_a[10] * in_b[6] + in_a[14] * in_b[7];
	out[7] = in_a[3] * in_b[4] + in_a[7] * in_b[5] + in_a[11] * in_b[6] + in_a[15] * in_b[7];
	out[8] = in_a[0] * in_b[8] + in_a[4] * in_b[9] + in_a[8] * in_b[10] + in_a[12] * in_b[11];
	out[9] = in_a[1] * in_b[8] + in_a[5] * in_b[9] + in_a[9] * in_b[10] + in_a[13] * in_b[11];
	out[10] = in_a[2] * in_b[8] + in_a[6] * in_b[9] + in_a[10] * in_b[10] + in_a[14] * in_b[11];
	out[11] = in_a[3] * in_b[8] + in_a[7] * in_b[9] + in_a[11] * in_b[10] + in_a[15] * in_b[11];
	out[12] = in_a[0] * in_b[12] + in_a[4] * in_b[13] + in_a[8] * in_b[14] + in_a[12] * in_b[15];
	out[13] = in_a[1] * in_b[12] + in_a[5] * in_b[13] + in_a[9] * in_b[14] + in_a[13] * in_b[15];
	out[14] = in_a[2] * in_b[12] + in_a[6] * in_b[13] + in_a[10] * in_b[14] + in_a[14] * in_b[15];
	out[15] = in_a[3] * in_b[12] + in_a[7] * in_b[13] + in_a[11] * in_b[14] + in_a[15] * in_b[15];
}

/*
void multiply_4x4_matrices(float (&in_a)[16], float (&in_b)[16], float (&out)[16])
{
for(int i = 0; i < 4; i++)
{
for(int j = 0; j < 4; j++)
{
out[4*i + j] = 0;

for (int k = 0; k < 4; k++)
out[4*i + j] += in_a[4*k + j] * in_b[4*i + k];
}
}
}
*/


void init_perspective_camera(float fovy, float aspect, float znear, float zfar,
							 float eyex, float eyey, float eyez, float centrex, float centrey,
							 float centrez, float upx, float upy, float upz,
							 float (&model_matrix)[16],
							 float (&view_matrix)[16],
							 float (&projection_matrix)[16])
{
	get_model_matrix(model_matrix);
	get_view_matrix(eyex, eyey, eyez, centrex, centrey, centrez, upx, upy, upz, view_matrix);
	get_projection_matrix(fovy, aspect, znear, zfar, projection_matrix);
}

void get_model_matrix(float (&in_a)[16])
{
	// Identity matrix
	in_a[0] = 1; in_a[4] = 0; in_a[8] =  0; in_a[12] = 0;
	in_a[1] = 0; in_a[5] = 1; in_a[9] =  0; in_a[13] = 0;
	in_a[2] = 0; in_a[6] = 0; in_a[10] = 1; in_a[14] = 0;
	in_a[3] = 0; in_a[7] = 0; in_a[11] = 0; in_a[15] = 1;
}

void get_view_matrix(float eyex, float eyey, float eyez, float centrex, float centrey, float centrez, float upx, float upy, float upz, float (&in_a)[16])
{
	vertex_3 f, up, s, u;

	f.x = centrex - eyex;
	f.y = centrey - eyey;
	f.z = centrez - eyez;
	f.normalize();

	up.x = upx;
	up.y = upy;
	up.z = upz;
	up.normalize();

	s = f.cross(up);
	s.normalize();

	u = s.cross(f);
	u.normalize();

	in_a[0] = s.x;  in_a[4] = s.y;  in_a[8] = s.z;   in_a[12] = 0;
	in_a[1] = u.x;  in_a[5] = u.y;  in_a[9] = u.z;   in_a[13] = 0;
	in_a[2] = -f.x; in_a[6] = -f.y; in_a[10] = -f.z; in_a[14] = 0;
	in_a[3] = 0;    in_a[7] = 0;    in_a[11] = 0;    in_a[15] = 1;

	float translate[16];
	translate[0] = 1; translate[4] = 0; translate[8] = 0;  translate[12] = -eyex;
	translate[1] = 0; translate[5] = 1; translate[9] = 0;  translate[13] = -eyey;
	translate[2] = 0; translate[6] = 0; translate[10] = 1; translate[14] = -eyez;
	translate[3] = 0; translate[7] = 0; translate[11] = 0; translate[15] = 1;

	float temp[16];
	multiply_4x4_matrices(in_a, translate, temp);

	for(size_t i = 0; i < 16; i++)
		in_a[i] = temp[i];
}

void get_projection_matrix(float fovy_degrees, float aspect, float znear, float zfar, float (&in_a)[16])
{
	const float pi = 4.0f*atanf(1.0);

	// Convert fovy to radians, then divide by 2
	float f = 1.0f / tan(fovy_degrees/360.0f*pi);

	in_a[0] = f/aspect; in_a[4] = 0; in_a[8] = 0;                              in_a[12] = 0;
	in_a[1] = 0;        in_a[5] = f; in_a[9] = 0;                              in_a[13] = 0;
	in_a[2] = 0;        in_a[6] = 0; in_a[10] = (zfar + znear)/(znear - zfar); in_a[14] = (2.0f*zfar*znear)/(znear - zfar);
	in_a[3] = 0;        in_a[7] = 0; in_a[11] = -1;                            in_a[15] = 0;
}


