#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 3) in vec3 aOffset;
layout (location = 4) in vec3 aColor;

out vec3 objectColor;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

void main()
{
    vec3 latticePoint = aPos + aOffset;
    gl_Position = projection * view * model * vec4(latticePoint, 1.0);
    objectColor = aColor;
}