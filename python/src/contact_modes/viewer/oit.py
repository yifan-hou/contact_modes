import numpy as np
from contact_modes import get_data

from .backend import *
from .shader import *
from .quad import Quad

DEBUG = True

class OITRenderer(object):
    def __init__(self, window):
        self.window = window
        self.peel_depth = 16
        self.max_peel_depth = 64
        self.opacity = 0.6

    def set_draw_func(self, draw_func):
        self.draw_func = draw_func

    def init_opengl(self):
        # ----------------------------------------------------------------------
        # Initialize buffers
        # ----------------------------------------------------------------------
        width = self.window.width
        height = self.window.height

        if DEBUG:
            print('DepthBufferFloatNV: ', glInitDepthBufferFloatNV())

        front_depth_tex_ids = []
        front_color_tex_ids = []
        front_fbo_ids = []
        front_depth_tex_ids = glGenTextures(2)
        front_color_tex_ids = glGenTextures(2)
        front_fbo_ids = glGenFramebuffers(2)
        if DEBUG:
            print('front depth tex ids', front_depth_tex_ids)
            print('front color tex ids', front_color_tex_ids)
            print('front fbo ids', front_fbo_ids)
        for i in range(2):
            glBindTexture(GL_TEXTURE_RECTANGLE, front_depth_tex_ids[i])
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_DEPTH_COMPONENT32F_NV,
                         width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)

            glBindTexture(GL_TEXTURE_RECTANGLE, front_color_tex_ids[i])
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, width, height,
                         0, GL_RGBA, GL_FLOAT, None)

            glBindFramebuffer(GL_FRAMEBUFFER, front_fbo_ids[i])
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                   GL_TEXTURE_RECTANGLE, front_depth_tex_ids[i], 0)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_RECTANGLE, front_color_tex_ids[i], 0)
        
        front_color_blender_tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_RECTANGLE, front_color_blender_tex_id)
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, width, height,
                     0, GL_RGBA, GL_FLOAT, None)

        front_color_blender_fbo_id = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, front_color_blender_fbo_id)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_RECTANGLE, front_depth_tex_ids[0], 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_RECTANGLE, front_color_blender_tex_id, 0)

        self.front_depth_tex_ids = front_depth_tex_ids
        self.front_color_tex_ids = front_color_tex_ids
        self.front_fbo_ids = front_fbo_ids
        self.front_color_blender_tex_id = front_color_blender_tex_id
        self.front_color_blender_fbo_id = front_color_blender_fbo_id

        self.query_id = glGenQueries(1)

        # ----------------------------------------------------------------------
        # Initialize shaders.
        # ----------------------------------------------------------------------
        shader_path = os.path.join(get_data(), 'shader', 'oit')

        vertex_source = os.path.join(shader_path, 'base_vertex.glsl')
        fragment_source = os.path.join(shader_path, 'front_peeling_blend_fragment.glsl')
        self.shader_peeling_blend = Shader([vertex_source], [fragment_source])

        vertex_source = os.path.join(shader_path, 'base_vertex.glsl')
        fragment_source = os.path.join(shader_path, 'front_peeling_final_fragment.glsl')
        self.shader_peeling_final = Shader([vertex_source], [fragment_source])

        vertex_source = os.path.join(shader_path, 'base_vertex.glsl')
        fragment_source = os.path.join(shader_path, 'weighted_final_fragment.glsl')
        self.shader_weighted_final = Shader([vertex_source], [fragment_source])

        fragment_sources = ['']*2
        fragment_sources[0] = os.path.join(shader_path, 'shade_fragment.glsl')

        vertex_source = os.path.join(shader_path, 'base_shade_vertex.glsl')
        fragment_sources[1] = os.path.join(shader_path, 'front_peeling_init_fragment.glsl')
        self.shader_peeling_init = Shader([vertex_source], fragment_sources)

        vertex_source = os.path.join(shader_path, 'base_shade_vertex.glsl')
        fragment_sources[1] = os.path.join(shader_path, 'front_peeling_peel_fragment.glsl')
        self.shader_peeling_peel = Shader([vertex_source], fragment_sources)

        vertex_source = os.path.join(shader_path, 'base_shade_vertex.glsl')
        fragment_sources[1] = os.path.join(shader_path, 'weighted_blend_fragment.glsl')
        self.shader_weighted_blend = Shader([vertex_source], fragment_sources)

        self.quad = Quad()
        self.quad.init_opengl()

    def render(self):
        # ----------------------------------------------------------------------
        # Peel the first layer
        # ----------------------------------------------------------------------
        glBindFramebuffer(GL_FRAMEBUFFER, self.front_color_blender_fbo_id)
        glDrawBuffer(GL_COLOR_ATTACHMENT0)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_MULTISAMPLE)

        self.shader_peeling_init.use()
        self.shader_peeling_init.set_float('uAlpha', self.opacity)
        self.draw_func(self.shader_peeling_init)
        self.shader_peeling_init.disable()

        # ----------------------------------------------------------------------
        # Depth peeling + blending
        # ----------------------------------------------------------------------
        for layer in range(1, self.peel_depth + 1):

            # ------------------------------------------------------------------
            # Peel the next depth layer
            # ------------------------------------------------------------------
            curr_id = layer % 2
            prev_id = 1 - curr_id
            # if DEBUG:
            #     print('curr id:', curr_id)
            #     print('prev id:', prev_id)
            
            glBindFramebuffer(GL_FRAMEBUFFER, self.front_fbo_ids[curr_id])
            glDrawBuffer(GL_COLOR_ATTACHMENT0)

            glClearColor(0, 0, 0, 0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glDisable(GL_BLEND)
            glEnable(GL_DEPTH_TEST)

            glBeginQuery(GL_SAMPLES_PASSED, self.query_id)

            self.shader_peeling_peel.use()
            self.shader_peeling_peel.bind_texture_rect('DepthTex', 0, self.front_depth_tex_ids[prev_id])
            self.shader_peeling_peel.set_float('uAlpha', self.opacity)
            self.draw_func(self.shader_peeling_peel)
            self.shader_peeling_peel.disable()

            glEndQuery(GL_SAMPLES_PASSED)

            # ------------------------------------------------------------------
            # Blend the current layer
            # ------------------------------------------------------------------
            glBindFramebuffer(GL_FRAMEBUFFER, self.front_color_blender_fbo_id)
            glDrawBuffer(GL_COLOR_ATTACHMENT0)

            glDisable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)

            glBlendEquation(GL_FUNC_ADD)
            glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, 
                                GL_ZERO, GL_ONE_MINUS_SRC_ALPHA)
            
            self.shader_peeling_blend.use()
            self.shader_peeling_blend.bind_texture_rect('TempTex', 0, 
                                                        self.front_color_tex_ids[curr_id])
            self.quad.draw(self.shader_peeling_blend)
            self.shader_peeling_blend.disable()

            glDisable(GL_BLEND)

            sample_count = glGetQueryObjectuiv(self.query_id, GL_QUERY_RESULT)
            # if DEBUG:
            #     print('sample count:', sample_count)
            if sample_count == 0:
                break

        # ----------------------------------------------------------------------
        # Compositing pass
        # ----------------------------------------------------------------------
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDisable(GL_DEPTH_TEST)

        self.shader_peeling_final.use()
        self.shader_peeling_final.set_vec3('uBackgroundColor', 
                                    np.array([0.2, 0.3, 0.3], dtype='float32'))
        self.shader_peeling_final.bind_texture_rect('ColorTex', 0, 
                                                    self.front_color_blender_tex_id)
        self.quad.draw(self.shader_peeling_final)
        self.shader_peeling_final.disable()

