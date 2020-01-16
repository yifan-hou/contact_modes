#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 3) in float aX;
layout (location = 4) in float aY;
layout (location = 5) in float aZ;
layout (location = 6) in float aDist;

out vec3 objectColor;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    vec3 latticePoint = aPos + vec3(aX, aY, aZ);
    gl_Position = projection * view * vec4(latticePoint, 1.0);
    if (aDist < 0)
        objectColor = vec3(-2.0*aDist, 0.0, 0.0);
    else
        objectColor = vec3(0.0, aDist, 0.0);
}