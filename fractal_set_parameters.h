#ifndef FRACTAL_SET_PARAMETERS_H
#define F

#include <string>
using std::string;

class fractal_set_parameters
{
public:
	string equation_text;
	bool randomize_c;
	bool use_pedestal;
	float pedestal_y_start;
	float pedestal_y_end;
	float C_x, C_y, C_z, C_w;
	float Z_w;
	short unsigned int max_iterations;
	short unsigned int resolution;
	float infinity;
	float x_min, x_max;
	float y_min, y_max;
	float z_min, z_max;
};

#endif