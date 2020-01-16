#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aOffset;
layout (location = 2) in vec3 aScale;
layout (location = 3) in vec3 aColor;
layout (location = 4) in float aAlpha;


out vec4 objectColor;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    vec3 point = aScale * aPos + aOffset;
    gl_Position = projection * view * vec4(point, 1.0);
    objectColor = vec4(aColor, aAlpha);
}