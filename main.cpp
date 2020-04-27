#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "dear imgui/imgui.h"
#include "dear imgui/imgui_impl_glfw.h"
#include "dear imgui/imgui_impl_opengl3.h"

#include "uv_camera.h"
#include "vertex_fragment_shader.h"
#include "bmp.h"
#include "fractal_set_parameters.h"
#include "logging_system.h"
#include "eqparse.h"
#include "primitives.h"
#include "marching_cubes.h"
using namespace marching_cubes;

#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <chrono>
#include <set>
using namespace std;

// Automatically link in the GLFW and GLEW libraries if compiling on MSVC++
#ifdef _MSC_VER
#pragma comment(lib, "glew32")
#pragma comment(lib, "glfw3") // https://github.com/glfw/glfw/releases/download/3.3.2/glfw-3.3.2.bin.WIN64.zip
#endif

#include <stdio.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif


bool lmb_down = false;
bool mmb_down = false;
bool rmb_down = false;
GLuint mouse_x = 0;
GLuint mouse_y = 0;
float u_spacer = 0.01f;
float v_spacer = 0.5f * u_spacer;
float w_spacer = 0.1f;
uv_camera main_camera;

logging_system log_system;


vector<triangle> triangles;
vector<vertex_3_with_normal> vertices_with_face_normals;

GLuint      render_fbo = 0;
GLuint      fbo_textures[3] = { 0, 0, 0 };
GLuint      quad_vao = 0;
GLuint      points_buffer = 0;

thread* gen_thread = 0;
atomic_bool stop = false;
atomic_bool thread_is_running = false;
atomic_bool vertex_data_refreshed = false;
vector<string> string_log;
mutex thread_mutex;

bool is_amd_gpu = false;

bool generate_button = true;
unsigned int triangle_buffer = 0;
unsigned int axis_buffer = 0;

std::chrono::high_resolution_clock::time_point start_time, end_time;





class RGB
{
public:
    unsigned char r, g, b;
};


vertex_fragment_shader render;
vertex_fragment_shader ssao;
vertex_fragment_shader flat;



struct
{
    struct
    {
        GLint           mv_matrix;
        GLint           proj_matrix;
        GLint           shading_level;
    } render;
    struct
    {
        GLint           ssao_level;
        GLint           object_level;
        GLint           ssao_radius;
        GLint           weight_by_angle;
        GLint           randomize_points;
        GLint           point_count;
    } ssao;
    struct
    {
        GLint           mv_matrix;
        GLint           proj_matrix;
        GLint			flat_colour;
    } flat;
} uniforms;

bool  show_shading;
bool  show_ao;
float ssao_level;
float ssao_radius;
bool  weight_by_angle;
bool randomize_points;
unsigned int point_count;


struct SAMPLE_POINTS
{
    vertex_4 point[256];
    vertex_4 random_vectors[256];
};


static unsigned int seed = 0x13371337;

static inline float random_float()
{
    float res;
    unsigned int tmp;

    seed *= 16807;

    tmp = seed ^ (seed >> 4) ^ (seed << 15);

    *((unsigned int*)&res) = (tmp >> 9) | 0x3F800000;

    return (res - 1.0f);
}


bool load_shaders()
{
    // Set up shader
    if (false == render.init("render.vs.glsl", "render.fs.glsl"))
    {
        cout << "Could not load render shader" << endl;
        return false;
    }

    if (false == flat.init("flat.vs.glsl", "flat.fs.glsl"))
    {
        cout << "Could not load flat shader" << endl;
        return false;
    }

    // Set up shader
    if (false == ssao.init("ssao.vs.glsl", "ssao.fs.glsl"))
    {
        cout << "Could not load SSAO shader" << endl;
        return false;
    }

    uniforms.render.mv_matrix = glGetUniformLocation(render.get_program(), "mv_matrix");
    uniforms.render.proj_matrix = glGetUniformLocation(render.get_program(), "proj_matrix");
    uniforms.render.shading_level = glGetUniformLocation(render.get_program(), "shading_level");

    uniforms.ssao.ssao_radius = glGetUniformLocation(ssao.get_program(), "ssao_radius");
    uniforms.ssao.ssao_level = glGetUniformLocation(ssao.get_program(), "ssao_level");
    uniforms.ssao.object_level = glGetUniformLocation(ssao.get_program(), "object_level");
    uniforms.ssao.weight_by_angle = glGetUniformLocation(ssao.get_program(), "weight_by_angle");
    uniforms.ssao.randomize_points = glGetUniformLocation(ssao.get_program(), "randomize_points");
    uniforms.ssao.point_count = glGetUniformLocation(ssao.get_program(), "point_count");

    uniforms.flat.mv_matrix = glGetUniformLocation(flat.get_program(), "mv_matrix");
    uniforms.flat.proj_matrix = glGetUniformLocation(flat.get_program(), "proj_matrix");
    uniforms.flat.flat_colour = glGetUniformLocation(flat.get_program(), "flat_colour");

    return true;
}




class monochrome_image
{
public:
	size_t width;
	size_t height;
	vector<unsigned char> pixel_data;
};

vector<monochrome_image> mimgs;

const size_t num_chars = 256;
const size_t image_width = 256;
const size_t image_height = 256;
const size_t char_width = 16;
const size_t char_height = 16;
const size_t num_chars_wide = image_width / char_width;
const size_t num_chars_high = image_height / char_height;




void print_char(vector<unsigned char>& fbpixels, const size_t fb_width, const size_t fb_height, const size_t char_x_pos, const size_t char_y_pos, const unsigned char c, const RGB& text_colour)
{
	for (size_t i = 0; i < mimgs[c].width; i++)
	{
		for (size_t j = 0; j < mimgs[c].height; j++)
		{
			size_t y = mimgs[c].height - j;

			size_t fb_x = char_x_pos + i;
			size_t fb_y = fb_height - char_y_pos + y;

			// If out of bounds, skip this pixel
			if (fb_x >= fb_width || fb_y >= fb_height)
				continue;

			size_t fb_index = 4 * (fb_y * fb_width + fb_x);
			size_t img_index = j * mimgs[c].width + i;

			RGB background_colour;
			background_colour.r = fbpixels[fb_index + 0];
			background_colour.g = fbpixels[fb_index + 1];
			background_colour.b = fbpixels[fb_index + 2];

			const unsigned char alpha = mimgs[c].pixel_data[img_index];
			const float alpha_float = alpha / 255.0f;

			RGB target_colour;
			target_colour.r = static_cast<unsigned char>(alpha_float * double(text_colour.r - background_colour.r) + background_colour.r);
			target_colour.g = static_cast<unsigned char>(alpha_float * double(text_colour.g - background_colour.g) + background_colour.g);
			target_colour.b = static_cast<unsigned char>(alpha_float * double(text_colour.b - background_colour.b) + background_colour.b);

			fbpixels[fb_index + 0] = target_colour.r;
			fbpixels[fb_index + 1] = target_colour.g;
			fbpixels[fb_index + 2] = target_colour.b;
			fbpixels[fb_index + 3] = 255;
		}
	}
}

void print_sentence(vector<unsigned char>& fbpixels, const size_t fb_width, const size_t fb_height, size_t char_x_pos, const size_t char_y_pos, const string s, const RGB& text_colour)
{
	for (size_t i = 0; i < s.size(); i++)
	{
		print_char(fbpixels, fb_width, fb_height, char_x_pos, char_y_pos, s[i], text_colour);

		size_t char_width = mimgs[s[i]].width;

		char_x_pos += char_width + 2;
	}
}


bool is_all_zeroes(size_t width, size_t height, const vector<unsigned char>& pixel_data)
{
	bool all_zeroes = true;

	for (size_t i = 0; i < width * height; i++)
	{
		if (pixel_data[i] != 0)
		{
			all_zeroes = false;
			break;
		}
	}

	return all_zeroes;
}

bool is_column_all_zeroes(size_t column, size_t width, size_t height, const vector<unsigned char>& pixel_data)
{
	bool all_zeroes = true;

	for (size_t y = 0; y < height; y++)
	{
		size_t index = y * width + column;

		if (pixel_data[index] != 0)
		{
			all_zeroes = false;
			break;
		}
	}

	return all_zeroes;
}



bool compile_and_link_compute_shader(const char* const file_name, GLuint& program)
{
	// Read in compute shader contents
	ifstream infile(file_name);

	ostringstream oss;

	if (infile.fail())
	{
		oss.clear();
		oss.str("");
		oss << "Could not open compute shader source file " << file_name;
		thread_mutex.lock();
		log_system.add_string_to_contents(oss.str());
		thread_mutex.unlock();

		return false;
	}

	string shader_code;
	string line;

	while (getline(infile, line))
	{
		shader_code += line;
		shader_code += "\n";
	}

	// Compile compute shader
	const char* cch = 0;
	GLint status = GL_FALSE;

	GLuint shader = glCreateShader(GL_COMPUTE_SHADER);

	glShaderSource(shader, 1, &(cch = shader_code.c_str()), NULL);


	glCompileShader(shader);

	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

	if (GL_FALSE == status)
	{
		string status_string = "Compute shader compile error.\n";
		vector<GLchar> buf(4096, '\0');
		glGetShaderInfoLog(shader, 4095, 0, &buf[0]);

		for (size_t i = 0; i < buf.size(); i++)
			if ('\0' != buf[i])
				status_string += buf[i];

		status_string += '\n';

		cout << status_string << endl;

		glDeleteShader(shader);

		return false;
	}

	// Link compute shader
	program = glCreateProgram();
	glAttachShader(program, shader);
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &status);

	if (GL_FALSE == status)
	{
		string status_string = "Program link error.\n";
		vector<GLchar> buf(4096, '\0');
		glGetShaderInfoLog(program, 4095, 0, &buf[0]);

		for (size_t i = 0; i < buf.size(); i++)
			if ('\0' != buf[i])
				status_string += buf[i];

		status_string += '\n';

		cout << status_string << endl;

		glDetachShader(program, shader);
		glDeleteShader(shader);
		glDeleteProgram(program);

		return false;
	}

	// The shader is no longer needed now that the program
	// has been linked
	glDetachShader(program, shader);
	glDeleteShader(shader);

	return true;
}


bool write_triangles_to_binary_stereo_lithography_file(const char* const file_name, vector<triangle>& triangles)
{
	ostringstream oss;

	oss.clear();
	oss.str("");
	oss << "Triangle count: " << triangles.size();
	thread_mutex.lock();
	log_system.add_string_to_contents(oss.str());
	thread_mutex.unlock();

	if (0 == triangles.size())
		return false;


	// Write to file.
	ofstream out(file_name, ios_base::binary);

	if (out.fail())
		return false;

	const size_t header_size = 80;
	vector<char> buffer(header_size, 0);
	unsigned int num_triangles = static_cast<unsigned int>(triangles.size()); // Must be 4-byte unsigned int.
	vertex_3 normal;


	// Copy everything to a single buffer.
	// We do this here because calling ofstream::write() only once PER MESH is going to 
	// send the data to disk faster than if we were to instead call ofstream::write()
	// thirteen times PER TRIANGLE.
	// Of course, the trade-off is that we are using 2x the RAM than what's absolutely required,
	// but the trade-off is often very much worth it (especially so for meshes with millions of triangles).

	oss.clear();
	oss.str("");
	oss << "Generating normal/vertex/attribute buffer";
	thread_mutex.lock();
	log_system.add_string_to_contents(oss.str());
	thread_mutex.unlock();

	// Enough bytes for twelve 4-byte floats plus one 2-byte integer, per triangle.
	const size_t data_size = (12 * sizeof(float) + sizeof(short unsigned int)) * num_triangles;
	buffer.resize(data_size, 0);

	// Use a pointer to assist with the copying.
	// Should probably use std::copy() instead, but memcpy() does the trick, so whatever...
	char* cp = &buffer[0];

	for (vector<triangle>::const_iterator i = triangles.begin(); i != triangles.end(); i++)
	{
		if (stop)
			break;

		// Get face normal.
		vertex_3 v0 = i->vertex[1] - i->vertex[0];
		vertex_3 v1 = i->vertex[2] - i->vertex[0];
		normal = v0.cross(v1);
		normal.normalize();

		memcpy(cp, &normal.x, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &normal.y, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &normal.z, sizeof(float)); cp += sizeof(float);

		memcpy(cp, &i->vertex[0].x, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[0].y, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[0].z, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[1].x, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[1].y, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[1].z, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[2].x, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[2].y, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[2].z, sizeof(float)); cp += sizeof(float);

		cp += sizeof(short unsigned int);
	}

	// Write blank header.
	out.write(reinterpret_cast<const char*>(&(buffer[0])), header_size);

	if (stop)
		num_triangles = 0;

	// Write number of triangles.
	out.write(reinterpret_cast<const char*>(&num_triangles), sizeof(unsigned int));

	oss.clear();
	oss.str("");
	oss << "Writing " << data_size / 1048576.0 << " MB of data to STL file: " << file_name;
	thread_mutex.lock();
	log_system.add_string_to_contents(oss.str());
	thread_mutex.unlock();

	if (false == stop)
		out.write(reinterpret_cast<const char*>(&buffer[0]), data_size);

	oss.clear();
	oss.str("");
	oss << "Done writing out.stl";
	thread_mutex.lock();
	log_system.add_string_to_contents(oss.str());
	thread_mutex.unlock();

	out.close();

	return true;
}



void get_vertices_with_face_normals_from_triangles(vector<vertex_3_with_normal>& vertices_with_face_normals, vector<triangle>& triangles)
{
	vector<vertex_3_with_index> v;

	vertices_with_face_normals.clear();

	if (0 == triangles.size())
		return;

	ostringstream oss;

	oss.clear();
	oss.str("");
	oss << "Welding vertices";
	thread_mutex.lock();
	log_system.add_string_to_contents(oss.str());
	thread_mutex.unlock();

	// Insert unique vertices into set.
	set<vertex_3_with_index> vertex_set;

	for (vector<triangle>::const_iterator i0 = triangles.begin(); i0 != triangles.end(); i0++)
	{
		if (stop)
			return;

		vertex_set.insert(i0->vertex[0]);
		vertex_set.insert(i0->vertex[1]);
		vertex_set.insert(i0->vertex[2]);
	}

	oss.clear();
	oss.str("");
	oss << "Vertices: " << vertex_set.size();
	thread_mutex.lock();
	log_system.add_string_to_contents(oss.str());
	thread_mutex.unlock();


	oss.clear();
	oss.str("");
	oss << "Generating vertex indices";
	thread_mutex.lock();
	log_system.add_string_to_contents(oss.str());
	thread_mutex.unlock();


	// Add indices to the vertices.
	for (set<vertex_3_with_index>::const_iterator i1 = vertex_set.begin(); i1 != vertex_set.end(); i1++)
	{
		if (stop)
			return;

		size_t index = v.size();
		v.push_back(*i1);
		v[index].index = static_cast<GLuint>(index);
	}

	vertex_set.clear();

	// Re-insert modified vertices into set.
	for (vector<vertex_3_with_index>::const_iterator i2 = v.begin(); i2 != v.end(); i2++)
	{
		if (stop)
			return;

		vertex_set.insert(*i2);
	}

	oss.clear();
	oss.str("");
	oss << "Assigning vertex indices to triangles";
	thread_mutex.lock();
	log_system.add_string_to_contents(oss.str());
	thread_mutex.unlock();


	// Find the three vertices for each triangle, by index.
	set<vertex_3_with_index>::iterator find_iter;

	for (vector<triangle>::iterator i3 = triangles.begin(); i3 != triangles.end(); i3++)
	{
		if (stop)
			return;

		find_iter = vertex_set.find(i3->vertex[0]);
		i3->vertex[0].index = find_iter->index;

		find_iter = vertex_set.find(i3->vertex[1]);
		i3->vertex[1].index = find_iter->index;

		find_iter = vertex_set.find(i3->vertex[2]);
		i3->vertex[2].index = find_iter->index;
	}

	vertex_set.clear();

	oss.clear();
	oss.str("");
	oss << "Calculating normals";
	thread_mutex.lock();
	log_system.add_string_to_contents(oss.str());
	thread_mutex.unlock();

	vertices_with_face_normals.resize(v.size());

	// Assign per-triangle face normals
	for (vector<triangle>::iterator i4 = triangles.begin(); i4 != triangles.end(); i4++)
	{
		if (stop)
			return;

		vertex_3 v0 = i4->vertex[1] - i4->vertex[0];
		vertex_3 v1 = i4->vertex[2] - i4->vertex[0];
		vertex_3 fn = v0.cross(v1);
		fn.normalize();

		vertices_with_face_normals[i4->vertex[0].index].nx += fn.x;
		vertices_with_face_normals[i4->vertex[0].index].ny += fn.y;
		vertices_with_face_normals[i4->vertex[0].index].nz += fn.z;
		vertices_with_face_normals[i4->vertex[1].index].nx += fn.x;
		vertices_with_face_normals[i4->vertex[1].index].ny += fn.y;
		vertices_with_face_normals[i4->vertex[1].index].nz += fn.z;
		vertices_with_face_normals[i4->vertex[2].index].nx += fn.x;
		vertices_with_face_normals[i4->vertex[2].index].ny += fn.y;
		vertices_with_face_normals[i4->vertex[2].index].nz += fn.z;
	}

	oss.clear();
	oss.str("");
	oss << "Generating final index/vertex data";
	thread_mutex.lock();
	log_system.add_string_to_contents(oss.str());
	thread_mutex.unlock();

	for (size_t i5 = 0; i5 < v.size(); i5++)
	{
		if (stop)
			return;

		// Assign vertex spatial comoponents
		vertices_with_face_normals[i5].x = v[i5].x;
		vertices_with_face_normals[i5].y = v[i5].y;
		vertices_with_face_normals[i5].z = v[i5].z;

		// Normalize face normal
		vertex_3 temp_face_normal(vertices_with_face_normals[i5].nx, vertices_with_face_normals[i5].ny, vertices_with_face_normals[i5].nz);
		temp_face_normal.normalize();

		vertices_with_face_normals[i5].nx = temp_face_normal.x;
		vertices_with_face_normals[i5].ny = temp_face_normal.y;
		vertices_with_face_normals[i5].nz = temp_face_normal.z;
	}

	oss.clear();
	oss.str("");
	oss << "Done";
	thread_mutex.lock();
	log_system.add_string_to_contents(oss.str());
	thread_mutex.unlock();
}



void thread_func_cpu(fractal_set_parameters p, vector<triangle>& triangles, vector<vertex_3_with_normal>& vertices_with_face_normals)
{
	thread_is_running = true;

	triangles.clear();
	vertices_with_face_normals.clear();

	bool make_border = true;

	quaternion C;
	C.x = p.C_x;
	C.y = p.C_y;
	C.z = p.C_z;
	C.w = p.C_w;

	ostringstream oss;

	string error_string;
	quaternion_julia_set_equation_parser eqparser;

	if (false == eqparser.setup(p.equation_text, error_string, C))
	{
		oss.clear();
		oss.str("");
		oss << "Equation error: " << error_string;
		thread_mutex.lock();
		log_system.add_string_to_contents(oss.str());
		thread_mutex.unlock();
		thread_is_running = false;
		return;
	}

	// When adding a border, use a value that is "much" greater than the threshold.
	const float border_value = 1.0f + p.infinity;

	size_t num_voxels = p.resolution * p.resolution;
	vector<float> xyplane0(num_voxels, 0);
	vector<float> xyplane1(num_voxels, 0);

	const float step_size_x = (p.x_max - p.x_min) / (p.resolution - 1);
	const float step_size_y = (p.y_max - p.y_min) / (p.resolution - 1);
	const float step_size_z = (p.z_max - p.z_min) / (p.resolution - 1);

	size_t z = 0;

	quaternion Z(p.x_min, p.y_min, p.z_min, p.Z_w);

	// Calculate 0th xy plane.
	for (size_t x = 0; x < p.resolution; x++, Z.x += step_size_x)
	{
		Z.y = p.y_min;

		for (size_t y = 0; y < p.resolution; y++, Z.y += step_size_y)
		{
			if (stop)
			{
				thread_is_running = false;
				return;
			}

			if (true == make_border && (x == 0 || y == 0 || z == 0 || x == p.resolution - 1 || y == p.resolution - 1 || z == p.resolution - 1))
			{
				xyplane0[x * p.resolution + y] = border_value;
			}
			else
			{
				const float y_span = (p.y_max - p.y_min);
				const float curr_span = 1.0f - static_cast<float>(p.y_max - Z.y) / y_span;

				if (p.use_pedestal == true && curr_span >= p.pedestal_y_start && curr_span <= p.pedestal_y_end)
				{
					xyplane0[x * p.resolution + y] = p.infinity - 0.00001f;
				}
				else
				{
					xyplane0[x * p.resolution + y] = eqparser.iterate(Z, p.max_iterations, p.infinity);
				}
			}
		}
	}

	// Prepare for 1st xy plane.
	z++;
	Z.z += step_size_z;



	size_t box_count = 0;


	// Calculate 1st and subsequent xy planes.
	for (; z < p.resolution; z++, Z.z += step_size_z)
	{
		Z.x = p.z_min;

		oss.clear();
		oss.str("");
		oss << "Calculating triangles from xy-plane pair " << z << " of " << p.resolution - 1;
		thread_mutex.lock();
		log_system.add_string_to_contents(oss.str());
		thread_mutex.unlock();

		for (size_t x = 0; x < p.resolution; x++, Z.x += step_size_x)
		{
			Z.y = p.y_min;

			for (size_t y = 0; y < p.resolution; y++, Z.y += step_size_y)
			{
				if (stop)
				{
					thread_is_running = false;
					return;
				}

				if (true == make_border && (x == 0 || y == 0 || z == 0 || x == p.resolution - 1 || y == p.resolution - 1 || z == p.resolution - 1))
				{
					xyplane1[x * p.resolution + y] = border_value;
				}
				else
				{
					const float y_span = (p.y_max - p.y_min);
					const float curr_span = 1.0f - static_cast<float>(p.y_max - Z.y) / y_span;

					if (p.use_pedestal == true && curr_span >= p.pedestal_y_start && curr_span <= p.pedestal_y_end)
					{
						xyplane1[x * p.resolution + y] = p.infinity - 0.00001f;
					}
					else
					{
						xyplane1[x * p.resolution + y] = eqparser.iterate(Z, p.max_iterations, p.infinity);
					}
				}
			}
		}

		// Calculate triangles for the xy-planes corresponding to z - 1 and z by marching cubes.
		tesselate_adjacent_xy_plane_pair(stop,
			box_count,
			xyplane0, xyplane1,
			z - 1,
			triangles,
			p.infinity, // Use threshold as isovalue.
			p.x_min, p.x_max, p.resolution,
			p.y_min, p.y_max, p.resolution,
			p.z_min, p.z_max, p.resolution);


		if (stop)
		{
			thread_is_running = false;
			return;
		}

		// Swap memory pointers (fast) instead of performing a memory copy (slow).
		xyplane1.swap(xyplane0);
	}

	if (false == stop)
	{
		get_vertices_with_face_normals_from_triangles(vertices_with_face_normals, triangles);
		write_triangles_to_binary_stereo_lithography_file("out.stl", triangles);
	}

	thread_is_running = false;
	return;
}



void thread_func_gpu(fractal_set_parameters p, quaternion_julia_set_equation_parser eqparser, quaternion C, vector<triangle>& triangles, vector<vertex_3_with_normal>& vertices_with_face_normals)
{
	thread_is_running = true;

	triangles.clear();
	vertices_with_face_normals.clear();

	//glutInitDisplayMode(GLUT_RGB);
	//glutInitWindowSize(glutGet(GLUT_SCREEN_WIDTH), 1);
	//glutInitWindowPosition(0, 0);
	//win_id2 = glutCreateWindow("Julia 4D 3 GPU acceleration window");
	//glutDisplayFunc(display_func2);

	GLuint compute_shader_program = 0;
	GLuint tex_output = 0;
	GLuint tex_input = 0;

	compile_and_link_compute_shader("julia.cs.glsl", compute_shader_program);

	glGenTextures(1, &tex_output);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, p.resolution, p.resolution, 0, GL_RED, GL_FLOAT, NULL);
	glBindImageTexture(0, tex_output, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

	// Generate input texture
	glGenTextures(1, &tex_input);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, tex_input);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);


	// Set up quaternion Julia set parameters

	// Set up grid parameters
	const float step_size_x = (p.x_max - p.x_min) / (p.resolution - 1);
	const float step_size_y = (p.y_max - p.y_min) / (p.resolution - 1);
	const float step_size_z = (p.z_max - p.z_min) / (p.resolution - 1);

	// Set up input quaternion
	quaternion Z(p.x_min, p.y_min, p.z_min, 0.0);

	// Set up output/input data
	const size_t num_output_channels = 1;
	vector<float> output_pixels(p.resolution * p.resolution * num_output_channels, 0.0f);
	const size_t num_input_channels = 4;
	vector<float> input_pixels(p.resolution * p.resolution * num_input_channels, 0.0f);

	// We must keep track of both the current and the previous slices, 
	// so that they can be used as input for the Marching Cubes algorithm
	vector<float> previous_slice = output_pixels;

	// The result of the Marching Cubes algorithm is triangles

	// For each z slice
	for (size_t z = 0; z < p.resolution; z++, Z.z += step_size_z)
	{
		Z.x = p.x_min;

		// Create pixel array to be used as input for the compute shader
		for (size_t x = 0; x < p.resolution; x++, Z.x += step_size_x)
		{
			Z.y = p.y_min;

			for (size_t y = 0; y < p.resolution; y++, Z.y += step_size_y)
			{
				if (stop)
				{
					thread_is_running = false;
					glDeleteTextures(1, &tex_output);
					glDeleteTextures(1, &tex_input);
					glDeleteProgram(compute_shader_program);
					//glutDestroyWindow(win_id2);
					return;
				}

				const size_t index = num_input_channels * (x * p.resolution + y);

				input_pixels[index + 0] = Z.x;
				input_pixels[index + 1] = Z.y;
				input_pixels[index + 2] = Z.z;
				input_pixels[index + 3] = Z.w;
			}
		}

		// Run the compute shader
		glActiveTexture(GL_TEXTURE1);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, p.resolution, p.resolution, 0, GL_RGBA, GL_FLOAT, &input_pixels[0]);
		glBindImageTexture(1, tex_input, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);

		glUseProgram(compute_shader_program);

		// Pass in the input image and quaternion Julia set parameters as uniforms
		glUniform1i(glGetUniformLocation(compute_shader_program, "input_image"), 1); // use GL_TEXTURE1
		glUniform4f(glGetUniformLocation(compute_shader_program, "c"), C.x, C.y, C.z, C.w);
		glUniform1i(glGetUniformLocation(compute_shader_program, "max_iterations"), p.max_iterations);
		glUniform1f(glGetUniformLocation(compute_shader_program, "threshold"), p.infinity);

		// Run compute shader
		glDispatchCompute(p.resolution, p.resolution, 1);

		// Wait for compute shader to finish
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		// Copy output pixel array to CPU as texture 0
		glActiveTexture(GL_TEXTURE0);
		glBindImageTexture(0, tex_output, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, &output_pixels[0]);

		quaternion tempq;
		tempq.x = p.x_min;
		tempq.y = p.y_min;
		tempq.z = Z.z;
		tempq.w = 1.0f;


		// Make a border, so that the mesh is closed around the edges
		for (size_t x = 0; x < p.resolution; x++, tempq.x += step_size_x)
		{
			tempq.y = p.y_min;

			for (size_t y = 0; y < p.resolution; y++, tempq.y += step_size_y)
			{
				if (stop)
				{
					thread_is_running = false;
					glDeleteTextures(1, &tex_output);
					glDeleteTextures(1, &tex_input);
					glDeleteProgram(compute_shader_program);
					//glutDestroyWindow(win_id2);
					return;
				}

				if (z == 0 || z == p.resolution - 1 ||
					x == 0 || x == p.resolution - 1 ||
					y == 0 || y == p.resolution - 1)
				{
					const size_t index = num_output_channels * (y * p.resolution + x);
					output_pixels[index] = p.infinity + 1.0f;
				}
				else
				{
					const float y_span = (p.y_max - p.y_min);
					const float curr_span = 1.0f - static_cast<float>(p.y_max - tempq.y) / y_span;
					const size_t index = num_output_channels * (x * p.resolution + y);

					if (p.use_pedestal == true && curr_span >= p.pedestal_y_start && curr_span <= p.pedestal_y_end)
					{
						output_pixels[index] = p.infinity - 0.00001f;
					}
				}
			}
		}

		// Use the Marching Cubes algorithm to convert the output data
		// into triangles, that is, if this isn't the first loop iteration
		if (z > 0)
		{
			size_t box_count = 0;

			ostringstream oss;

			oss.clear();
			oss.str("");
			oss << "Calculating triangles from xy-plane pair " << z << " of " << p.resolution - 1;
			thread_mutex.lock();
			log_system.add_string_to_contents(oss.str());
			thread_mutex.unlock();

			// Calculate triangles for the xy-planes corresponding to z - 1 and z by marching cubes.
			tesselate_adjacent_xy_plane_pair(stop,
				box_count,
				previous_slice, output_pixels,
				z - 1,
				triangles,
				p.infinity, // Use threshold as isovalue.
				p.x_min, p.x_max, p.resolution,
				p.y_min, p.y_max, p.resolution,
				p.z_min, p.z_max, p.resolution);

			if (stop)
			{
				thread_is_running = false;
				glDeleteTextures(1, &tex_output);
				glDeleteTextures(1, &tex_input);
				glDeleteProgram(compute_shader_program);
				//glutDestroyWindow(win_id2);
				return;
			}
		}

		previous_slice.swap(output_pixels);
	}

	if (false == stop)
	{
		get_vertices_with_face_normals_from_triangles(vertices_with_face_normals, triangles);
		write_triangles_to_binary_stereo_lithography_file("out.stl", triangles);
	}

	thread_is_running = false;
	glDeleteTextures(1, &tex_output);
	glDeleteTextures(1, &tex_input);
	glDeleteProgram(compute_shader_program);
	//glutDestroyWindow(win_id2);
	return;
}








bool obtain_control_contents(fractal_set_parameters& p)
{
	p.equation_text = "Z = sin(Z) + C*sin(Z)";
	p.use_pedestal = true;
	p.pedestal_y_start = 0.0f;
	p.pedestal_y_end = 0.15f;
	p.C_x = 0.2866f;
	p.C_y = 0.5133f;
	p.C_z = 0.46f;
	p.C_w = 0.2467f;
	p.Z_w = 0.0;
	p.max_iterations = 8;
	p.resolution = 100;
	p.infinity = 4.0f;
	p.x_min = p.y_min = p.z_min = -1.5f;
	p.x_max = p.y_max = p.z_max = 1.5f;

	//ostringstream oss;

	//if (p.randomize_c = randomize_c_checkbox->get_int_val())
	//{
	//	float c_x = rand() / static_cast<float>(RAND_MAX);
	//	float c_y = rand() / static_cast<float>(RAND_MAX);
	//	float c_z = rand() / static_cast<float>(RAND_MAX);
	//	float c_w = rand() / static_cast<float>(RAND_MAX);

	//	if (rand() % 2 == 0)
	//		c_x = -c_x;

	//	if (rand() % 2 == 0)
	//		c_y = -c_y;

	//	if (rand() % 2 == 0)
	//		c_z = -c_z;

	//	if (rand() % 2 == 0)
	//		c_w = -c_w;

	//	oss.clear();
	//	oss.str("");
	//	oss << c_x;
	//	c_x_edittext->set_text(oss.str());

	//	oss.clear();
	//	oss.str("");
	//	oss << c_y;
	//	c_y_edittext->set_text(oss.str());

	//	oss.clear();
	//	oss.str("");
	//	oss << c_z;
	//	c_z_edittext->set_text(oss.str());

	//	oss.clear();
	//	oss.str("");
	//	oss << c_w;
	//	c_w_edittext->set_text(oss.str());
	//}


	//p.equation_text = equation_edittext->text;

	//if (p.equation_text == "")
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "blank equation text";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}

	//p.randomize_c = randomize_c_checkbox->get_int_val();
	//p.use_pedestal = use_pedestal_checkbox->get_int_val();

	//string temp_string;

	//temp_string = pedestal_y_start_edittext->text;

	//if (false == is_real_number(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "pedestal y start is not a real number";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.pedestal_y_start;
	//}

	//temp_string = pedestal_y_end_edittext->text;

	//if (false == is_real_number(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "pedestal y end is not a real number";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.pedestal_y_end;
	//}

	//if (p.pedestal_y_start < 0 || p.pedestal_y_start > 1)
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "pedestal y start must be between 0 and 1";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}

	//if (p.pedestal_y_end < 0 || p.pedestal_y_end > 1)
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "pedestal y end must be between 0 and 1";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}

	//if (p.pedestal_y_start >= p.pedestal_y_end)
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "Y start must be smaller than y_end";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}




	//temp_string = c_x_edittext->text;

	//if (false == is_real_number(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "c.x  is not a real number";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.C_x;
	//}

	//temp_string = c_y_edittext->text;

	//if (false == is_real_number(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "c.y  is not a real number";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.C_y;
	//}

	//temp_string = c_z_edittext->text;

	//if (false == is_real_number(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "c.z  is not a real number";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.C_z;
	//}

	//temp_string = c_w_edittext->text;

	//if (false == is_real_number(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "c.w  is not a real number";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.C_w;
	//}

	//temp_string = x_min_edittext->text;

	//if (false == is_real_number(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "x min  is not a real number";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.x_min;
	//}

	//temp_string = y_min_edittext->text;

	//if (false == is_real_number(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "y min  is not a real number";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.y_min;
	//}

	//temp_string = z_min_edittext->text;

	//if (false == is_real_number(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "z min  is not a real number";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.z_min;
	//}




	//temp_string = x_max_edittext->text;

	//if (false == is_real_number(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "x max  is not a real number";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.x_max;
	//}

	//temp_string = y_max_edittext->text;

	//if (false == is_real_number(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "y max  is not a real number";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.y_max;
	//}

	//temp_string = z_max_edittext->text;

	//if (false == is_real_number(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "z max  is not a real number";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.z_max;
	//}

	//if (p.x_min >= p.x_max)
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "x min must be less than x max";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}

	//if (p.y_min >= p.y_max)
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "y min must be less than y max";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}

	//if (p.z_min >= p.z_max)
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "z min must be less than z max";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}

	//temp_string = z_w_edittext->text;

	//if (false == is_real_number(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "z.w  is not a real number";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.Z_w;
	//}


	//temp_string = infinity_edittext->text;

	//if (false == is_real_number(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "infinity  is not a real number";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.infinity;
	//}


	//temp_string = iterations_edittext->text;

	//if (false == is_unsigned_int(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "max iterations is not an unsigned int";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.max_iterations;
	//}

	//temp_string = resolution_edittext->text;

	//if (false == is_unsigned_int(temp_string))
	//{
	//	oss.clear();
	//	oss.str("");
	//	oss << "resolution is not an unsigned int";
	//	thread_mutex.lock();
	//	log_system.add_string_to_contents(oss.str());
	//	thread_mutex.unlock();

	//	return false;
	//}
	//else
	//{
	//	istringstream iss(temp_string);
	//	iss >> p.resolution;

	//	if (p.resolution < 3)
	//	{
	//		oss.clear();
	//		oss.str("");
	//		oss << "resolution must be greater than or equal to 3";
	//		thread_mutex.lock();
	//		log_system.add_string_to_contents(oss.str());
	//		thread_mutex.unlock();

	//		return false;
	//	}
	//}

	return true;
}


static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}


void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    int prev_mouse_x = mouse_x;
    int prev_mouse_y = mouse_y;

    mouse_x = xpos;
    mouse_y = ypos;

    if (ImGui::IsAnyWindowHovered())
        return;

    int mouse_delta_x = mouse_x - prev_mouse_x;
    int mouse_delta_y = prev_mouse_y - mouse_y;

    if (true == lmb_down && (0 != mouse_delta_x || 0 != mouse_delta_y))
    {
        // Rotate camera
        main_camera.u -= static_cast<float>(mouse_delta_y) * u_spacer;
        main_camera.v += static_cast<float>(mouse_delta_x) * v_spacer;
    }
    else if (true == rmb_down && (0 != mouse_delta_y))
    {
        // Move camera
        main_camera.w -= static_cast<float>(mouse_delta_y) * w_spacer;

        if (main_camera.w < 1.1f)
            main_camera.w = 1.1f;
        else if (main_camera.w > 20.0f)
            main_camera.w = 20.0f;
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (ImGui::IsAnyWindowHovered())
        return;

    if (GLFW_MOUSE_BUTTON_LEFT == button)
    {
        if (action == GLFW_PRESS)
            lmb_down = true;
        else
            lmb_down = false;
    }
    else if (GLFW_MOUSE_BUTTON_MIDDLE == button)
    {
        if (action == GLFW_PRESS)
            mmb_down = true;
        else
            mmb_down = false;
    }
    else if (GLFW_MOUSE_BUTTON_RIGHT == button)
    {
        if (action == GLFW_PRESS)
            rmb_down = true;
        else
            rmb_down = false;
    }
}




void generate_cancel_button_func(void)
{
    if (generate_button == false)
    {
        ostringstream oss;

        oss.clear();
        oss.str("");
        oss << "Aborting";
        thread_mutex.lock();
        log_system.add_string_to_contents(oss.str());
        thread_mutex.unlock();

        stop = true;
        vertex_data_refreshed = false;

        if (gen_thread != 0)
        {
            stop = true;

            gen_thread->join();

            delete gen_thread;
            gen_thread = 0;
            stop = true;
        }

        generate_button = true;
        //generate_mesh_button->set_name(const_cast<char*>("Generate mesh"));
    }
    else
    {
        fractal_set_parameters p;

        if (false == obtain_control_contents(p))
        {
            ostringstream oss;

            oss.clear();
            oss.str("");
            oss << "Aborting";
            thread_mutex.lock();
            log_system.add_string_to_contents(oss.str());
            thread_mutex.unlock();

            return;
        }

        stop = false;
        vertex_data_refreshed = false;

        if (gen_thread != 0)
        {
            stop = true;
            gen_thread->join();

            delete gen_thread;
            gen_thread = 0;
            stop = false;
        }

        if (0)//gpu_acceleration_checkbox->get_int_val())
        {
            GLint global_workgroup_count[2];
            glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &global_workgroup_count[0]);
            glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &global_workgroup_count[1]);

            ostringstream oss;

            if (p.resolution > global_workgroup_count[0])
            {
                //oss.clear();
                //oss.str("");
                //oss << "Texture width " << p.resolution << " is larger than max " << global_workgroup_count[0];
                //thread_mutex.lock();
                //log_system.add_string_to_contents(oss.str());
                //thread_mutex.unlock();

                return;
            }

            if (p.resolution > global_workgroup_count[1])
            {
                //oss.clear();
                //oss.str("");
                //oss << "Texture height " << p.resolution << " is larger than max " << global_workgroup_count[1];
                //thread_mutex.lock();
                //log_system.add_string_to_contents(oss.str());
                //thread_mutex.unlock();

                return;
            }

            string error_string;
            quaternion_julia_set_equation_parser eqparser;
            quaternion C(p.C_x, p.C_y, p.C_z, p.C_w);

            if (false == eqparser.setup(p.equation_text, error_string, C))
            {
                //oss.clear();
                //oss.str("");
                //oss << "Equation error: " << error_string;
                //thread_mutex.lock();
                //log_system.add_string_to_contents(oss.str());
                //thread_mutex.unlock();

                return;
            }

            string code = eqparser.emit_compute_shader_code(p.resolution, p.resolution);

            ofstream of("julia.cs.glsl");
            of << code;
            of.close();

            gen_thread = new thread(thread_func_gpu, p, eqparser, C, ref(triangles), ref(vertices_with_face_normals));
        }
        else
        {
            gen_thread = new thread(thread_func_cpu, p, ref(triangles), ref(vertices_with_face_normals));
        }

        generate_button = false;
       // generate_mesh_button->set_name(const_cast<char*>("Cancel"));

        start_time = std::chrono::high_resolution_clock::now();
    }
}



RGB HSBtoRGB(unsigned short int hue_degree, unsigned char sat_percent, unsigned char bri_percent)
{
	float R = 0.0f;
	float G = 0.0f;
	float B = 0.0f;

	if (hue_degree > 359)
		hue_degree = 359;

	if (sat_percent > 100)
		sat_percent = 100;

	if (bri_percent > 100)
		bri_percent = 100;

	float hue_pos = 6.0f - ((static_cast<float>(hue_degree) / 359.0f) * 6.0f);

	if (hue_pos >= 0.0f && hue_pos < 1.0f)
	{
		R = 255.0f;
		G = 0.0f;
		B = 255.0f * hue_pos;
	}
	else if (hue_pos >= 1.0f && hue_pos < 2.0f)
	{
		hue_pos -= 1.0f;

		R = 255.0f - (255.0f * hue_pos);
		G = 0.0f;
		B = 255.0f;
	}
	else if (hue_pos >= 2.0f && hue_pos < 3.0f)
	{
		hue_pos -= 2.0f;

		R = 0.0f;
		G = 255.0f * hue_pos;
		B = 255.0f;
	}
	else if (hue_pos >= 3.0f && hue_pos < 4.0f)
	{
		hue_pos -= 3.0f;

		R = 0.0f;
		G = 255.0f;
		B = 255.0f - (255.0f * hue_pos);
	}
	else if (hue_pos >= 4.0f && hue_pos < 5.0f)
	{
		hue_pos -= 4.0f;

		R = 255.0f * hue_pos;
		G = 255.0f;
		B = 0.0f;
	}
	else
	{
		hue_pos -= 5.0f;

		R = 255.0f;
		G = 255.0f - (255.0f * hue_pos);
		B = 0.0f;
	}

	if (100 != sat_percent)
	{
		if (0 == sat_percent)
		{
			R = 255.0f;
			G = 255.0f;
			B = 255.0f;
		}
		else
		{
			if (255.0f != R)
				R += ((255.0f - R) / 100.0f) * (100.0f - sat_percent);
			if (255.0f != G)
				G += ((255.0f - G) / 100.0f) * (100.0f - sat_percent);
			if (255.0f != B)
				B += ((255.0f - B) / 100.0f) * (100.0f - sat_percent);
		}
	}

	if (100 != bri_percent)
	{
		if (0 == bri_percent)
		{
			R = 0.0f;
			G = 0.0f;
			B = 0.0f;
		}
		else
		{
			if (0.0f != R)
				R *= static_cast<float>(bri_percent) / 100.0f;
			if (0.0f != G)
				G *= static_cast<float>(bri_percent) / 100.0f;
			if (0.0f != B)
				B *= static_cast<float>(bri_percent) / 100.0f;
		}
	}

	if (R < 0.0f)
		R = 0.0f;
	else if (R > 255.0f)
		R = 255.0f;

	if (G < 0.0f)
		G = 0.0f;
	else if (G > 255.0f)
		G = 255.0f;

	if (B < 0.0f)
		B = 0.0f;
	else if (B > 255.0f)
		B = 255.0f;

	RGB rgb;

	rgb.r = static_cast<unsigned char>(R);
	rgb.g = static_cast<unsigned char>(G);
	rgb.b = static_cast<unsigned char>(B);

	return rgb;
}


vector<GLfloat> vertex_data;
vector<GLfloat> flat_data;



void refresh_vertex_data_blue(void)
{
	vertex_data.clear();

	for (size_t i = 0; i < triangles.size(); i++)
	{
		if (stop)
		{
			vertex_data.clear();
			return;
		}

		vertex_3 colour(0.0f, 0.8f, 1.0f);

		size_t v0_index = triangles[i].vertex[0].index;
		size_t v1_index = triangles[i].vertex[1].index;
		size_t v2_index = triangles[i].vertex[2].index;

		vertex_3 v0_fn(vertices_with_face_normals[v0_index].nx, vertices_with_face_normals[v0_index].ny, vertices_with_face_normals[v0_index].nz);
		vertex_3 v1_fn(vertices_with_face_normals[v1_index].nx, vertices_with_face_normals[v1_index].ny, vertices_with_face_normals[v1_index].nz);
		vertex_3 v2_fn(vertices_with_face_normals[v2_index].nx, vertices_with_face_normals[v2_index].ny, vertices_with_face_normals[v2_index].nz);

		vertex_3 v0(triangles[i].vertex[0].x, triangles[i].vertex[0].y, triangles[i].vertex[0].z);
		vertex_3 v1(triangles[i].vertex[1].x, triangles[i].vertex[1].y, triangles[i].vertex[1].z);
		vertex_3 v2(triangles[i].vertex[2].x, triangles[i].vertex[2].y, triangles[i].vertex[2].z);

		vertex_data.push_back(v0.x);
		vertex_data.push_back(v0.y);
		vertex_data.push_back(v0.z);
		vertex_data.push_back(v0_fn.x);
		vertex_data.push_back(v0_fn.y);
		vertex_data.push_back(v0_fn.z);
		vertex_data.push_back(colour.x);
		vertex_data.push_back(colour.y);
		vertex_data.push_back(colour.z);

		vertex_data.push_back(v1.x);
		vertex_data.push_back(v1.y);
		vertex_data.push_back(v1.z);
		vertex_data.push_back(v1_fn.x);
		vertex_data.push_back(v1_fn.y);
		vertex_data.push_back(v1_fn.z);
		vertex_data.push_back(colour.x);
		vertex_data.push_back(colour.y);
		vertex_data.push_back(colour.z);

		vertex_data.push_back(v2.x);
		vertex_data.push_back(v2.y);
		vertex_data.push_back(v2.z);
		vertex_data.push_back(v2_fn.x);
		vertex_data.push_back(v2_fn.y);
		vertex_data.push_back(v2_fn.z);
		vertex_data.push_back(colour.x);
		vertex_data.push_back(colour.y);
		vertex_data.push_back(colour.z);
	}

}



void refresh_vertex_data_rainbow(void)
{
	vertex_data.clear();

	float min_3d_length = FLT_MAX;
	float max_3d_length = FLT_MIN;

	for (size_t i = 0; i < triangles.size(); i++)
	{
		if (stop)
			return;

		size_t v0_index = triangles[i].vertex[0].index;
		size_t v1_index = triangles[i].vertex[1].index;
		size_t v2_index = triangles[i].vertex[2].index;

		vertex_3 v0(triangles[i].vertex[0].x, triangles[i].vertex[0].y, triangles[i].vertex[0].z);
		vertex_3 v1(triangles[i].vertex[1].x, triangles[i].vertex[1].y, triangles[i].vertex[1].z);
		vertex_3 v2(triangles[i].vertex[2].x, triangles[i].vertex[2].y, triangles[i].vertex[2].z);

		float vertex_length = v0.length();

		if (vertex_length > max_3d_length)
			max_3d_length = vertex_length;

		if (vertex_length < min_3d_length)
			min_3d_length = vertex_length;

		vertex_length = v1.length();

		if (vertex_length > max_3d_length)
			max_3d_length = vertex_length;

		if (vertex_length < min_3d_length)
			min_3d_length = vertex_length;

		vertex_length = v2.length();

		if (vertex_length > max_3d_length)
			max_3d_length = vertex_length;

		if (vertex_length < min_3d_length)
			min_3d_length = vertex_length;
	}

	double max_rainbow = 360.0;
	double min_rainbow = 360.0;





	for (size_t i = 0; i < triangles.size(); i++)
	{
		if (stop)
		{
			vertex_data.clear();
			return;
		}

		vertex_3 colour(1.0f, 0.5f, 0.0);

		size_t v0_index = triangles[i].vertex[0].index;
		size_t v1_index = triangles[i].vertex[1].index;
		size_t v2_index = triangles[i].vertex[2].index;

		vertex_3 v0_fn(vertices_with_face_normals[v0_index].nx, vertices_with_face_normals[v0_index].ny, vertices_with_face_normals[v0_index].nz);
		vertex_3 v1_fn(vertices_with_face_normals[v1_index].nx, vertices_with_face_normals[v1_index].ny, vertices_with_face_normals[v1_index].nz);
		vertex_3 v2_fn(vertices_with_face_normals[v2_index].nx, vertices_with_face_normals[v2_index].ny, vertices_with_face_normals[v2_index].nz);

		vertex_3 v0(triangles[i].vertex[0].x, triangles[i].vertex[0].y, triangles[i].vertex[0].z);
		vertex_3 v1(triangles[i].vertex[1].x, triangles[i].vertex[1].y, triangles[i].vertex[1].z);
		vertex_3 v2(triangles[i].vertex[2].x, triangles[i].vertex[2].y, triangles[i].vertex[2].z);

		float vertex_length = v0.length() - min_3d_length;

		RGB rgb = HSBtoRGB(static_cast<unsigned short int>(
			max_rainbow - ((vertex_length / (max_3d_length - min_3d_length)) * min_rainbow)),
			static_cast<unsigned char>(50),
			static_cast<unsigned char>(100));

		colour.x = rgb.r / 255.0f;
		colour.y = rgb.g / 255.0f;
		colour.z = rgb.b / 255.0f;

		vertex_data.push_back(v0.x);
		vertex_data.push_back(v0.y);
		vertex_data.push_back(v0.z);
		vertex_data.push_back(v0_fn.x);
		vertex_data.push_back(v0_fn.y);
		vertex_data.push_back(v0_fn.z);
		vertex_data.push_back(colour.x);
		vertex_data.push_back(colour.y);
		vertex_data.push_back(colour.z);

		vertex_length = v1.length() - min_3d_length;

		rgb = HSBtoRGB(static_cast<unsigned short int>(
			max_rainbow - ((vertex_length / (max_3d_length - min_3d_length)) * min_rainbow)),
			static_cast<unsigned char>(50),
			static_cast<unsigned char>(100));

		colour.x = rgb.r / 255.0f;
		colour.y = rgb.g / 255.0f;
		colour.z = rgb.b / 255.0f;

		vertex_data.push_back(v1.x);
		vertex_data.push_back(v1.y);
		vertex_data.push_back(v1.z);
		vertex_data.push_back(v1_fn.x);
		vertex_data.push_back(v1_fn.y);
		vertex_data.push_back(v1_fn.z);
		vertex_data.push_back(colour.x);
		vertex_data.push_back(colour.y);
		vertex_data.push_back(colour.z);


		vertex_length = v2.length() - min_3d_length;

		rgb = HSBtoRGB(static_cast<unsigned short int>(
			max_rainbow - ((vertex_length / (max_3d_length - min_3d_length)) * min_rainbow)),
			static_cast<unsigned char>(50),
			static_cast<unsigned char>(100));

		colour.x = rgb.r / 255.0f;
		colour.y = rgb.g / 255.0f;
		colour.z = rgb.b / 255.0f;

		vertex_data.push_back(v2.x);
		vertex_data.push_back(v2.y);
		vertex_data.push_back(v2.z);
		vertex_data.push_back(v2_fn.x);
		vertex_data.push_back(v2_fn.y);
		vertex_data.push_back(v2_fn.z);
		vertex_data.push_back(colour.x);
		vertex_data.push_back(colour.y);
		vertex_data.push_back(colour.z);
	}

}

void refresh_vertex_data(void)
{
	ostringstream oss;

	oss.clear();
	oss.str("");
	oss << "Refreshing vertex data";
	thread_mutex.lock();
	log_system.add_string_to_contents(oss.str());
	thread_mutex.unlock();

	//int do_rainbow = rainbow_colouring_checkbox->get_int_val();

	if (0)//do_rainbow)
		refresh_vertex_data_rainbow();
	else
		refresh_vertex_data_blue();

	if (stop)
	{
		oss.clear();
		oss.str("");
		oss << "Cancelled refreshing vertex data";
		thread_mutex.lock();
		log_system.add_string_to_contents(oss.str());
		thread_mutex.unlock();
	}
	else
	{
		oss.clear();
		oss.str("");
		oss << "Done refreshing vertex data";
		thread_mutex.lock();
		log_system.add_string_to_contents(oss.str());
		thread_mutex.unlock();
	}
}







int main(int, char**)
{
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 430";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, "Julia 4D 3 Multithreaded", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    // glfwSwapInterval(1); // Enable vsync

    bool err = glewInit() != GLEW_OK;

    if (err)
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 1;
    }

    if (false == load_shaders())
    {
        return 1;
    }

    BMP font;

    if (false == font.load("font.bmp"))
    {
        cout << "could not load font.bmp" << endl;
        return false;
    }

    size_t char_index = 0;

    vector< vector<GLubyte> > char_data;
    vector<unsigned char> char_template(char_width * char_height);
    char_data.resize(num_chars, char_template);

    for (size_t i = 0; i < num_chars_wide; i++)
    {
        for (size_t j = 0; j < num_chars_high; j++)
        {
            size_t left = i * char_width;
            size_t right = left + char_width - 1;
            size_t top = j * char_height;
            size_t bottom = top + char_height - 1;

            for (size_t k = left, x = 0; k <= right; k++, x++)
            {
                for (size_t l = top, y = 0; l <= bottom; l++, y++)
                {
                    size_t img_pos = 4 * (k * image_height + l);
                    size_t sub_pos = x * char_height + y;

                    char_data[char_index][sub_pos] = font.Pixels[img_pos]; // Assume grayscale, only use r component
                }
            }

            char_index++;
        }
    }

    for (size_t n = 0; n < num_chars; n++)
    {
        if (is_all_zeroes(char_width, char_height, char_data[n]))
        {
            monochrome_image img;

            img.width = char_width / 4;
            img.height = char_height;

            img.pixel_data.resize(img.width * img.height, 0);

            mimgs.push_back(img);
        }
        else
        {
            size_t first_non_zeroes_column = 0;
            size_t last_non_zeroes_column = char_width - 1;

            for (size_t x = 0; x < char_width; x++)
            {
                bool all_zeroes = is_column_all_zeroes(x, char_width, char_height, char_data[n]);

                if (false == all_zeroes)
                {
                    first_non_zeroes_column = x;
                    break;
                }
            }

            for (size_t x = first_non_zeroes_column + 1; x < char_width; x++)
            {
                bool all_zeroes = is_column_all_zeroes(x, char_width, char_height, char_data[n]);

                if (false == all_zeroes)
                {
                    last_non_zeroes_column = x;
                }
            }

            size_t cropped_width = last_non_zeroes_column - first_non_zeroes_column + 1;

            monochrome_image img;
            img.width = cropped_width;
            img.height = char_height;
            img.pixel_data.resize(img.width * img.height, 0);

            for (size_t i = 0; i < num_chars_wide; i++)
            {
                for (size_t j = 0; j < num_chars_high; j++)
                {
                    const size_t left = first_non_zeroes_column;
                    const size_t right = left + cropped_width - 1;
                    const size_t top = 0;
                    const size_t bottom = char_height - 1;

                    for (size_t k = left, x = 0; k <= right; k++, x++)
                    {
                        for (size_t l = top, y = 0; l <= bottom; l++, y++)
                        {
                            const size_t img_pos = l * char_width + k;
                            const size_t sub_pos = y * cropped_width + x;

                            img.pixel_data[sub_pos] = char_data[n][img_pos];
                        }
                    }
                }
            }

            mimgs.push_back(img);
        }
    }


    ssao_level = 1.0f;
    ssao_radius = 0.05f;
    show_shading = true;
    show_ao = true;
    weight_by_angle = true;
    randomize_points = true;
    point_count = 10;

    load_shaders();

    glGenFramebuffers(1, &render_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, render_fbo);
    glGenTextures(3, fbo_textures);

    glBindTexture(GL_TEXTURE_2D, fbo_textures[0]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB16F, 2048, 2048);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, fbo_textures[1]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 2048, 2048);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, fbo_textures[2]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32F, 2048, 2048);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, fbo_textures[0], 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, fbo_textures[1], 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, fbo_textures[2], 0);

    static const GLenum draw_buffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };

    glDrawBuffers(2, draw_buffers);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenVertexArrays(1, &quad_vao);
    glBindVertexArray(quad_vao);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    int i;
    SAMPLE_POINTS point_data;

    for (i = 0; i < 256; i++)
    {
        do
        {
            point_data.point[i].x = random_float() * 2.0f - 1.0f;
            point_data.point[i].y = random_float() * 2.0f - 1.0f;
            point_data.point[i].z = random_float(); //  * 2.0f - 1.0f;
            point_data.point[i].w = 0.0f;
        } while (length(point_data.point[i]) > 1.0f);
        normalize(point_data.point[i]);
    }
    for (i = 0; i < 256; i++)
    {
        point_data.random_vectors[i].x = random_float();
        point_data.random_vectors[i].y = random_float();
        point_data.random_vectors[i].z = random_float();
        point_data.random_vectors[i].w = random_float();
    }

    glGenBuffers(1, &points_buffer);
    glBindBuffer(GL_UNIFORM_BUFFER, points_buffer);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(SAMPLE_POINTS), &point_data, GL_STATIC_DRAW);





    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableSetMousePos;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.WantCaptureMouse = true;



    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    cout << io.WantCaptureMouse << endl;

    // Setup Dear ImGui style
    //ImGui::StyleColorsDark();
    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Read 'docs/FONTS.txt' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != NULL);

    // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);





    // Main loop
    while (!glfwWindowShouldClose(window))
    {
		if (false == thread_is_running && false == generate_button)
		{
			if (false == vertex_data_refreshed && false == stop && triangles.size() > 0)
			{
				refresh_vertex_data();
				vertex_data_refreshed = true;
			}

			generate_button = true;
			//generate_mesh_button->set_name(const_cast<char*>("Generate mesh"));

			end_time = std::chrono::high_resolution_clock::now();

			std::chrono::duration<float, std::milli> elapsed = end_time - start_time;

			ostringstream oss;
			oss.clear();
			oss.str("");
			oss << "Duration: " << elapsed.count() / 1000.0f << " seconds";
			thread_mutex.lock();
			log_system.add_string_to_contents(oss.str());
			thread_mutex.unlock();
		}
        
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        //// 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        //if (show_demo_window)
        //    ImGui::ShowDemoWindow(&show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
		{
			static float f = 0.0f;

			ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

			if (generate_button)
			{
				if (ImGui::Button("Generate mesh"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
					generate_cancel_button_func();
			}
			else
			{
				if (ImGui::Button("Cancel"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
					generate_cancel_button_func();
			}

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        

        glClearColor(1, 0.5f, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDisable(GL_BLEND);

        const GLfloat black[] = { 0.0f, 0.0f, 0.0f, 0.0f };
        static const GLfloat one = 1.0f;
        static const GLenum draw_buffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };

        glViewport(0, 0, display_w, display_h);

        glBindFramebuffer(GL_FRAMEBUFFER, render_fbo);
        glEnable(GL_DEPTH_TEST);

        glClearBufferfv(GL_COLOR, 0, black);
        glClearBufferfv(GL_COLOR, 1, black);
        glClearBufferfv(GL_DEPTH, 0, &one);

        glBindBufferBase(GL_UNIFORM_BUFFER, 0, points_buffer);

        glUseProgram(render.get_program());

        main_camera.calculate_camera_matrices(display_w, display_h);
        glUniformMatrix4fv(uniforms.render.proj_matrix, 1, GL_FALSE, main_camera.projection_mat);
        glUniformMatrix4fv(uniforms.render.mv_matrix, 1, GL_FALSE, main_camera.view_mat);
        glUniform1f(uniforms.render.shading_level, show_shading ? (show_ao ? 0.7f : 1.0f) : 0.0f);

        if (1)//draw_axis_checkbox->get_int_val())
        {
            glLineWidth(2.0);

            glUseProgram(flat.get_program());

            main_camera.calculate_camera_matrices(display_w, display_h);
            glUniformMatrix4fv(uniforms.flat.proj_matrix, 1, GL_FALSE, main_camera.projection_mat);
            glUniformMatrix4fv(uniforms.flat.mv_matrix, 1, GL_FALSE, main_camera.view_mat);

            const GLuint components_per_vertex = 3;
            const GLuint components_per_position = 3;
            vector<float> flat_data;
            flat_data.clear();

            flat_data.push_back(0);
            flat_data.push_back(0);
            flat_data.push_back(0);
            flat_data.push_back(1);
            flat_data.push_back(0);
            flat_data.push_back(0);

            glUniform3f(uniforms.flat.flat_colour, 1.0, 0.0, 0.0);

            glDeleteBuffers(1, &axis_buffer);
            glGenBuffers(1, &axis_buffer);

            GLuint num_vertices = static_cast<GLuint>(flat_data.size()) / components_per_vertex;

            glBindBuffer(GL_ARRAY_BUFFER, axis_buffer);
            glBufferData(GL_ARRAY_BUFFER, flat_data.size() * sizeof(GLfloat), &flat_data[0], GL_DYNAMIC_DRAW);

            glEnableVertexAttribArray(glGetAttribLocation(render.get_program(), "position"));
            glVertexAttribPointer(glGetAttribLocation(render.get_program(), "position"),
                components_per_position,
                GL_FLOAT,
                GL_FALSE,
                components_per_vertex * sizeof(GLfloat),
                NULL);

            glDrawArrays(GL_LINES, 0, num_vertices);

            flat_data.clear();

            flat_data.push_back(0);
            flat_data.push_back(0);
            flat_data.push_back(0);
            flat_data.push_back(0);
            flat_data.push_back(1);
            flat_data.push_back(0);

            glUniform3f(uniforms.flat.flat_colour, 0.0, 1.0, 0.0);

            glDeleteBuffers(1, &axis_buffer);
            glGenBuffers(1, &axis_buffer);

            num_vertices = static_cast<GLuint>(flat_data.size()) / components_per_vertex;

            glBindBuffer(GL_ARRAY_BUFFER, axis_buffer);
            glBufferData(GL_ARRAY_BUFFER, flat_data.size() * sizeof(GLfloat), &flat_data[0], GL_DYNAMIC_DRAW);

            glEnableVertexAttribArray(glGetAttribLocation(render.get_program(), "position"));
            glVertexAttribPointer(glGetAttribLocation(render.get_program(), "position"),
                components_per_position,
                GL_FLOAT,
                GL_FALSE,
                components_per_vertex * sizeof(GLfloat),
                NULL);

            glDrawArrays(GL_LINES, 0, num_vertices);

            flat_data.clear();

            flat_data.push_back(0);
            flat_data.push_back(0);
            flat_data.push_back(0);
            flat_data.push_back(0);
            flat_data.push_back(0);
            flat_data.push_back(1);

            glUniform3f(uniforms.flat.flat_colour, 0.0, 0.0, 1.0);

            glDeleteBuffers(1, &axis_buffer);
            glGenBuffers(1, &axis_buffer);

            num_vertices = static_cast<GLuint>(flat_data.size()) / components_per_vertex;

            glBindBuffer(GL_ARRAY_BUFFER, axis_buffer);
            glBufferData(GL_ARRAY_BUFFER, flat_data.size() * sizeof(GLfloat), &flat_data[0], GL_DYNAMIC_DRAW);

            glEnableVertexAttribArray(glGetAttribLocation(render.get_program(), "position"));
            glVertexAttribPointer(glGetAttribLocation(render.get_program(), "position"),
                components_per_position,
                GL_FLOAT,
                GL_FALSE,
                components_per_vertex * sizeof(GLfloat),
                NULL);

            glDrawArrays(GL_LINES, 0, num_vertices);
        }


        if (vertex_data_refreshed && vertex_data.size() > 0)
        {
            glUseProgram(render.get_program());

            const GLuint components_per_vertex = 9;
            const GLuint components_per_normal = 3;
            const GLuint components_per_position = 3;
            const GLuint components_per_colour = 3;

            glDeleteBuffers(1, &triangle_buffer);
            glGenBuffers(1, &triangle_buffer);

            const GLuint num_vertices = static_cast<GLuint>(vertex_data.size()) / components_per_vertex;

            glBindBuffer(GL_ARRAY_BUFFER, triangle_buffer);
            glBufferData(GL_ARRAY_BUFFER, vertex_data.size() * sizeof(GLfloat), &vertex_data[0], GL_DYNAMIC_DRAW);

            glEnableVertexAttribArray(glGetAttribLocation(render.get_program(), "position"));
            glVertexAttribPointer(glGetAttribLocation(render.get_program(), "position"),
                components_per_position,
                GL_FLOAT,
                GL_FALSE,
                components_per_vertex * sizeof(GLfloat),
                NULL);

            glEnableVertexAttribArray(glGetAttribLocation(render.get_program(), "normal"));
            glVertexAttribPointer(glGetAttribLocation(render.get_program(), "normal"),
                components_per_normal,
                GL_FLOAT,
                GL_TRUE,
                components_per_vertex * sizeof(GLfloat),
                (const GLvoid*)(components_per_position * sizeof(GLfloat)));

            glEnableVertexAttribArray(glGetAttribLocation(render.get_program(), "colour"));
            glVertexAttribPointer(glGetAttribLocation(render.get_program(), "colour"),
                components_per_colour,
                GL_FLOAT,
                GL_TRUE,
                components_per_vertex * sizeof(GLfloat),
                (const GLvoid*)(components_per_normal * sizeof(GLfloat) + components_per_position * sizeof(GLfloat)));

            // Draw 12 vertices per card
            glDrawArrays(GL_TRIANGLES, 0, num_vertices);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glUseProgram(ssao.get_program());

        glUniform1f(uniforms.ssao.ssao_radius, ssao_radius * float(display_w) / 1000.0f);
        glUniform1f(uniforms.ssao.ssao_level, show_ao ? (show_shading ? 0.3f : 1.0f) : 0.0f);
        glUniform1i(uniforms.ssao.weight_by_angle, weight_by_angle ? 1 : 0);
        glUniform1i(uniforms.ssao.randomize_points, randomize_points ? 1 : 0);
        glUniform1ui(uniforms.ssao.point_count, point_count);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, fbo_textures[0]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, fbo_textures[1]);

        glDisable(GL_DEPTH_TEST);
        glBindVertexArray(quad_vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);


        if (0)//draw_console_checkbox->get_int_val() && log_system.get_contents_size() > 0)
        {
            size_t char_x_pos = 10;
            size_t char_y_pos = 30;

            RGB text_colour;
            text_colour.r = 255;
            text_colour.g = 255;
            text_colour.b = 255;

            vector<unsigned char> fbpixels(4 * static_cast<size_t>(display_w) * static_cast<size_t>(display_h));

            glReadPixels(0, 0, display_w, display_h, GL_RGBA, GL_UNSIGNED_BYTE, &fbpixels[0]);

            // Do anything you like here... for instance, use OpenCV for convolution

            thread_mutex.lock();
  /*          for (size_t i = 0; i < log_system.get_contents_size(); i++)
            {
                string s;
                log_system.get_string_from_contents(i, s);
                print_sentence(fbpixels, win_x, win_y, char_x_pos, char_y_pos, s, text_colour);
                char_y_pos += 20;
            }*/
            thread_mutex.unlock();

            glDrawPixels(display_w, display_h, GL_RGBA, GL_UNSIGNED_BYTE, &fbpixels[0]);
        }












        
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
