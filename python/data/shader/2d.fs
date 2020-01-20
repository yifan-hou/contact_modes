#version 410 core
out vec4 FragColor;

uniform vec3 objectColor;

void main()
{
    // apply gamma correction
    float gamma = 2.2;
    FragColor = vec4(pow(objectColor, vec3(1.0/gamma)), 1.);
}