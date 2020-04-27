#version 410 core


// Per-vertex inputs
layout (location = 0) in vec4 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 colour;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;

// Inputs from vertex shader
out VS_OUT
{
    vec3 N;
    vec3 L;
    vec3 V;
    vec3 vertex_colour;
} vs_out;

// Position of light
uniform vec3 light_pos = vec3(100.0, 100.0, 100.0);

void main(void)
{	float max_rainbow = 0.0;
	float min_rainbow = 100.0;
        float min_3d_length = 1.0;
        float max_3d_length = 4.0;
		float vertex_length = sqrt(position.x * position.x + position.y * position.y + position.z * position.z) - min_3d_length;

		vs_out.vertex_colour = colour;

//		vs_out.vertex_colour = 
//		HSBtoRGB(uint(
//			max_rainbow - ((vertex_length / (max_3d_length - min_3d_length)) * min_rainbow)),
//			uint(50),
//			uint(100));

    // Calculate view-space coordinate
    vec4 P = mv_matrix * position;

    // Calculate normal in view-space
    vs_out.N = mat3(mv_matrix) * normal;

    // Calculate light vector
    vs_out.L = light_pos - P.xyz;

    // Calculate view vector
    vs_out.V = -P.xyz;

    // Calculate the clip-space position of each vertex
    gl_Position = proj_matrix * P;
}
