#version 330 core
out vec4 FragColor;

in vec4 objectColor;

void main()
{
    // FragColor = vec4(objectColor, 0.25);
    FragColor = objectColor;
}