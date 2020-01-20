#version 410 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 offset;
uniform float scale;

void main()
{
    vec3 pt = scale * aPos + offset;
	gl_Position = projection * view * model * vec4(pt, 1.0);
}