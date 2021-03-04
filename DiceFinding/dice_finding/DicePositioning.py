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
"""Sets up group of dice position and rotation within a pyrender scene based on Halton sequence."""

import numpy as np

import DiceScene
import DiceConfig

from MathUtil import Halton

def get_positions_unconstrained(spread_radius : float, number_of_dice : int, sequence_offset : int):
    """Based on the spread radius and Halton sequence offset, generates corresponding positions from Halton sequence."""
    sequence = Halton.sequence(dim = 2, n_sample = number_of_dice, offset = sequence_offset)
    radii = sequence[:, 0] * spread_radius
    angles = sequence[:,1] * np.pi * 2
    positions = radii * np.cos(angles) * np.array([1, 0, 0])[:,np.newaxis] + radii * np.sin(angles) * np.array([0, 0, 1])[:,np.newaxis]
    return positions

def constrain_positions(positions : np.ndarray):
    """Given positions, moves dice away from each other that would be overlapping."""
    dice_radius = DiceConfig.get_local_bounding_box_horizontal_radius()
    min_distance = 2 * dice_radius
    step_distance = min_distance * 0.1

    needs_iteration = True
    while needs_iteration:
        needs_iteration = False
        for i in range(positions.shape[1]):
            for j in range(i+1, positions.shape[1]):
                diff = positions[:,i] - positions[:,j]
                distance = np.linalg.norm(diff)
                if distance < min_distance:
                    needs_iteration = True
                    direction = diff / distance
                    positions[:,i] += step_distance * 0.5 * direction
                    positions[:,j] -= step_distance * 0.5 * direction
    return positions

def get_positions_constrained(spread_radius : float, number_of_dice : int, sequence_offset : int):
    """Based on the spread radius and Halton sequence offset, generates corresponding positions from Halton sequence,
    with additional constraining to prevent overlapping of dice."""
    positions = get_positions_unconstrained(spread_radius, number_of_dice, sequence_offset)
    positions_constrained = constrain_positions(positions)
    return positions_constrained

def get_rotations(number_of_dice : int, sequence_offset : int):
    """Based on the Halton sequence offset, generates dice rotations (defining vertical axis, and rotation around that vertical axis)."""
    sequence = Halton.sequence(dim = 3, n_sample = number_of_dice, offset = sequence_offset)
    rotation_axes_are_z = sequence[:,1] > 0.5
    y_rotations_degrees = sequence[:,2] * 360.0 - 180.0
    side_rotations_prob = sequence[:,0]
    rotation_indices = np.floor(side_rotations_prob * 6).astype(int) - 1
    rotation_indices[rotation_indices == 3] = -1
    rotation_indices[rotation_indices == 4] = 1
    return rotation_indices, rotation_axes_are_z, y_rotations_degrees

def get_scene_setup(spread_radius : float, number_of_dice : int, sequence_offset : int):
    """Given a spread radius and Halton sequence offset, generates a scene setup for dice positions and rotations."""
    positions = get_positions_constrained(spread_radius, number_of_dice, sequence_offset)
    side_rotations, rotation_axes, y_rotations_degrees = get_rotations(number_of_dice, sequence_offset)
    dice_setups = []
    for i in range(number_of_dice):
        position = positions[:,i]
        rotation_index = side_rotations[i]
        rotation_is_z = rotation_axes[i]
        y_rotation_degrees = y_rotations_degrees[i]
        dice_setup = DiceScene.DiceSetup(True, position[0], position[2], rotation_index, rotation_is_z, y_rotation_degrees)
        dice_setups.append(dice_setup)

    scene_setup = DiceScene.SceneSetup(dice_setups)
    return scene_setup 
