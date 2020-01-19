from contact_modes.viewer.backend import *


class Shader(object):
    def __init__(self, vertex_sources, fragment_sources, geometry_source=None):
        # Load source code from file paths.
        vs = []
        for vert_src in vertex_sources:
            with open(vert_src, 'r') as source:
                vs.append(source.read())
        fs = []
        for frag_src in fragment_sources:
            with open(frag_src, 'r') as source:
                fs.append(source.read())
        if geometry_source is not None:
            with open(geometry_source, 'r') as source:
                gs = source.read()
        # Compile shaders.
        vertex = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex, vs)
        glCompileShader(vertex)
        fragment = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment, fs)
        glCompileShader(fragment)
        if geometry_source is not None:
            geometry = glCreateShader(GL_GEOMETRY_SHADER)
            glShaderSource(geometry, [gs])
            glCompileShader(geometry)
        # Compile program.
        self.id = glCreateProgram()
        glAttachShader(self.id, vertex)
        glAttachShader(self.id, fragment)
        if geometry_source is not None:
            glAttachShader(self.id, geometry)
        glLinkProgram(self.id)
        if not glGetProgramiv(self.id, GL_LINK_STATUS):
            print(glGetProgramInfoLog(self.id))
        assert(glGetProgramiv(self.id, GL_LINK_STATUS))
        # Delete shaders as they're linked into our program.
        glDeleteShader(vertex)
        glDeleteShader(fragment)
        if geometry_source is not None:
            glDeleteShader(geometry)

    def use(self):
        glUseProgram(self.id)
    
    def disable(self):
        glUseProgram(0)
    
    def set_bool(self, name, value):
        glUniform1i(glGetUniformLocation(self.id, name), value)

    def set_int(self, name, value):
        glUniform1i(glGetUniformLocation(self.id, name), value)

    def set_float(self, name, value):
        glUniform1f(glGetUniformLocation(self.id, name), value)

    def set_vec3(self, name, value):
        glUniform3fv(glGetUniformLocation(self.id, name), 1, value)

    def set_vec4(self, name, value):
        glUniform4fv(glGetUniformLocation(self.id, name), 1, value)

    def set_mat3(self, name, value):
        glUniformMatrix3fv(glGetUniformLocation(self.id, name), 1, GL_FALSE, value)

    def set_mat4(self, name, value):
        glUniformMatrix4fv(glGetUniformLocation(self.id, name), 1, GL_FALSE, value)

    def bind_texture_rect(self, name, value, id):
        self.set_int(name, value)
        glActiveTexture(GL_TEXTURE0 + value)
        glBindTexture(GL_TEXTURE_RECTANGLE, id)