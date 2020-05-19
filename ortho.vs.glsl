#version 430
layout(location = 0) in vec4 position;
layout(location = 1) in vec2 texcoord;

out vec2 ftexcoord;

void main()
{
    ftexcoord = texcoord;
    gl_Position = position;
}