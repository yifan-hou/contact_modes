#!/usr/bin/env python3
import argparse
import os
import sys
from time import time

import glm
import numpy as np
import imgui

import contact_modes
import contact_modes.collision
import contact_modes.dynamics
import contact_modes.geometry
import contact_modes.shape
import contact_modes.viewer
from contact_modes.viewer.backend import *

import multi_sim


class Planning2DDemo(contact_modes.viewer.Application):
    def __init__(self):
        super().__init__()

    def init(self, viewer):
        super().init(viewer)

        window = self.window
        window.set_on_init(self.init_win_p2d)
        window.set_on_draw(self.draw)
        window.set_on_key_press(self.on_key_press_p2d)

    def init_win_p2d(self):
        super().init_win()

        # Initialize GUI.
        self.init_gui()
        
        # Create scene.
        self.system = multi_sim.box_slide_2d()
        # self.box = contact_modes.shape.Box()

        # Initialize renderer.
        self.renderer = contact_modes.viewer.OITRenderer(self.window)
        # self.renderer = contact_modes.viewer.BasicLightingRenderer(self.window)
        self.renderer.init_opengl()
        self.renderer.set_draw_func(self.draw_scene)

    def step(self):
        pass

    def draw(self):
        # Clear frame.
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)

        # Step.
        self.step()

        # Render scene.
        self.renderer.render()

        # Render GUI.
        self.imgui_impl.process_inputs()
        imgui.new_frame()

        self.draw_menu()
        self.draw_gui()

        imgui.render()
        self.imgui_impl.render(imgui.get_draw_data())
    
    def draw_scene(self, shader):
        # ----------------------------------------------------------------------
        # 1. Setup shader uniforms
        # ----------------------------------------------------------------------
        shader.use()

        # model view projection
        model = glm.mat4(1.0)
        shader.set_mat4('model', np.asarray(model))

        view = self.camera.get_view()
        shader.set_mat4('view', np.asarray(view))

        width = self.window.width
        height = self.window.height
        projection = glm.perspective(glm.radians(50.0), width/height, 0.1, 100.0)
        shader.set_mat4('projection', np.asarray(projection))

        # lighting
        lightPos = np.array(self.light_pos)
        shader.set_vec3('lightPos', lightPos)
        shader.set_vec3('lightColor', np.array([1.0, 1.0, 1.0], 'f'))

        cameraPos = glm.vec3(glm.column(glm.inverse(view), 3))
        shader.set_vec3('viewPos', np.asarray(cameraPos))

        # ----------------------------------------------------------------------
        # 2. Draw scene
        # ----------------------------------------------------------------------
        self.system.draw(shader)

    def draw_menu(self):
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_quit, selected_quit = imgui.menu_item(
                        "Quit", 'Cmd+Q', False, True
                    )
                if clicked_quit:
                    exit(1)
                imgui.end_menu()
            if imgui.begin_menu("Controls", True):
                clicked_scene, selected_scene = imgui.menu_item(
                    "Scene", "Cmd+S", False, True
                )
                if clicked_scene:
                    self.scene_controls = True
                imgui.end_menu()
            imgui.end_main_menu_bar()
    
    def draw_gui(self):
        self.draw_scene_gui()

    def init_gui(self):
        self.init_scene_gui()

    def init_scene_gui(self):
        self.scene_controls = True
        self.load_scene = False
        self.object_color = contact_modes.get_color('clay')
        self.object_color[3] = 0.5
        self.obstacle_color = contact_modes.get_color('teal')
        self.light_pos = [0, 2.0, 10.0]

    def draw_scene_gui(self):
        if not self.scene_controls:
            return
        _, opened = imgui.begin("Scene", True)
        if not opened:
            self.scene_controls = False
        
        changed, new_color = imgui.color_edit4('object', *self.object_color)
        if changed or self.load_scene:
            [body.set_color(np.array(new_color)) for body in self.system.bodies]
            self.object_color = new_color

        changed, new_color = imgui.color_edit4('obs', *self.obstacle_color)
        if changed or self.load_scene:
            [o.set_color(np.array(new_color)) for o in self.system.obstacles]
            self.obstacle_color = new_color

        changed, new_pos = imgui.slider_float3('light', *self.light_pos, -10.0, 10.0)
        if changed or self.load_scene:
            self.light_pos = new_pos

        self.load_scene = True

        imgui.end()

    def on_key_press_p2d(self, win, key, scancode, action, mods):
        pass

viewer = contact_modes.viewer.Viewer()
viewer.add_application(Planning2DDemo())
viewer.start()
