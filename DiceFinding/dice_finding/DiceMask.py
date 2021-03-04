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
"""Generates depth image and per-dice segmentation mask."""

import numpy as np
import pyrender

import DiceScene

def get_image_depth_and_mask(scene : pyrender.Scene, scene_setup_loader : DiceScene.SceneSetupLoader, width : int, height : int, keep_nodes_in_scene : bool):
    """Renders an image ggiven a scene and seetup, along with the depth and segmentation mask labelling each die."""
    r = pyrender.OffscreenRenderer(width, height)
    color_bg, depth_bg = r.render(scene)

    depth_nodes = []
    for node in scene_setup_loader.dice_nodes:
        scene.add_node(node)
        color_node, depth_node = r.render(scene)
        depth_nodes.append(depth_node)
        scene.remove_node(node)

    scene_setup_loader.add_loaded_to_scene(scene)
    color_final, depth_final = r.render(scene)
    if not keep_nodes_in_scene:
        scene_setup_loader.remove_nodes_from_scene(scene)

    #Initialize labels of pixels to -1 (for background)
    labels_mask = np.ones((height, width), dtype = np.int8) * -1

    for index, depth_for_node in enumerate(depth_nodes):
        depth_not_background = np.not_equal(depth_bg, depth_for_node)
        depth_at_foreground = np.equal(depth_final, depth_for_node)
        depth_at_dice = np.logical_and(depth_not_background, depth_at_foreground)
        labels_mask[depth_at_dice] = index

    return color_final, depth_final, labels_mask

def get_mask_image_bounding_box(full_mask : np.ndarray, mask_index : int):
    """Given a mask image, gets the bounding box encompassing the given mask index."""
    matching_mask = np.where(full_mask == mask_index)
    min_xy = np.array([np.min(matching_mask[1]), np.min(matching_mask[0])])
    max_xy = np.array([np.max(matching_mask[1]), np.max(matching_mask[0])])
    width_height = max_xy - min_xy
    min_x_y_width_height = np.hstack((min_xy, width_height))
    return min_x_y_width_height

def get_mask_image_bounding_boxes(full_mask : np.ndarray, number_of_dice : int):
    """Given a mask image, gets the bounding box encompassing the given mask index for each die."""
    boxes = []
    for index in range(number_of_dice):
        boxes.append(get_mask_image_bounding_box(full_mask, index))
    return boxes