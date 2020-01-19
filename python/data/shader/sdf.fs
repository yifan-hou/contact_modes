#version 410 core
out vec4 FragColor;

in vec3 objectColor;

void main()
{
    FragColor = vec4(objectColor, 0.9);
}