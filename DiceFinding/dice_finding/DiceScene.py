# Copyright 2021 Dice Finding Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Describes setup of dice within a pyrender scene, and handles loading of meshes into the scene."""

import numpy as np
import trimesh
from trimesh.transformations import quaternion_about_axis, quaternion_multiply
import pyrender

from typing import List

class DiceSetup(object):
    """Setup pose information for a given die."""
    def __init__(self, exists : bool, x : float, z : float, side_rotation_times : int, side_rotation_around_z : bool, rotation_y_degrees : float):
        self.exists = exists
        self.x = x
        self.z = z
        self.side_rotation_times = side_rotation_times
        self.side_rotation_around_z = side_rotation_around_z
        self.rotation_y_degrees = rotation_y_degrees

class SceneSetup(object):
    """Setup information for all die in a scene."""
    def __init__(self, dice_setups : List[DiceSetup]):
        self.dice_setups = dice_setups

    def get_number_of_dice(self):
        """Gets number of dice in the setup."""
        return len(self.dice_setups)

class SceneSetupLoader(object):
    """Loads die models into a scene, based on a given SceneSetup."""
    def load_mesh_if_needed(self):
        """Load die 3D model mesh if needed."""
        if not self.dice_trimesh:
            self.dice_trimesh = trimesh.load(self.trimesh_location)
        if not self.dice_mesh:
            self.dice_mesh = pyrender.Mesh.from_trimesh(self.dice_trimesh)

    def __init__(self, trimesh_location):
        self.trimesh_location = trimesh_location
        self.dice_trimesh = None
        self.dice_mesh = None
        self.dice_nodes = []

        self.load_mesh_if_needed()

    def remove_nodes_from_scene(self, scene : pyrender.Scene):
        """Remove nodes in scene corresponding to internal list of dice nodes."""
        for mesh_node in self.dice_nodes:
            if scene.has_node(mesh_node):
                scene.remove_node(mesh_node)

    def clear_dice_nodes(self):
        """Clear internal list of dice nodes."""
        self.dice_nodes = []

    def load_dice_nodes(self, scene_setup : SceneSetup):
        """Creates internal list of dice nodes, including calculating pose rotation and translation, from a given SceneSetup."""
        for dice_setup in [dice_setup for dice_setup in scene_setup.dice_setups if dice_setup.exists]:
            dice_node = pyrender.Node(mesh = self.dice_mesh, matrix = np.eye(4))
            quaternion_y_rotation = quaternion_about_axis(dice_setup.rotation_y_degrees * np.pi / 180, [0, 1, 0])
            side_axis = [0, 0, 1] if dice_setup.side_rotation_around_z else [1, 0, 0]
            quaternion_side_rotation = quaternion_about_axis(dice_setup.side_rotation_times * np.pi / 2, side_axis)
            dice_node.rotation = np.roll(quaternion_multiply(quaternion_y_rotation, quaternion_side_rotation), -1)
            dice_node.translation = (dice_setup.x, 0, dice_setup.z)
            self.dice_nodes.append(dice_node)

    def add_loaded_to_scene(self, scene : pyrender.Scene):
        """Add to scene all dice nodes in loaded internal list."""
        for node in self.dice_nodes:
            scene.add_node(node)



