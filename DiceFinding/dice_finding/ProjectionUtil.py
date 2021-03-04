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
"""Utilities for transforming points to/from homogeneous coordinates, between screen normalized and pixel coordinates."""

import numpy as np
import pyrender

from typing import Tuple

def from_homogeneous(vectors, at_infinity):
    """Converts vectors from homogeneous (from point or direction)."""
    if at_infinity:
        return vectors[:-1,...]
    else:
        divided = vectors / vectors[-1,...]
        return divided[:-1,...]

def to_homogeneous(vectors, at_infinity):
    """Converts vectors to homogeneous (as point or direction)."""
    if at_infinity:
        return np.r_[vectors, np.zeros((1, vectors.shape[1]))]
    else:
        return np.r_[vectors, np.ones((1, vectors.shape[1]))]

def project_to_screen_normalized_coords(camera : pyrender.Camera, eye_space_points : np.ndarray):
    """Project eye space points to screen-normalized coordinates."""
    projection_matrix = camera.get_projection_matrix()
    points_projected_homogeneous = projection_matrix @ to_homogeneous(eye_space_points, False)
    points_projected = from_homogeneous(points_projected_homogeneous, False)
    points_screen = (points_projected + 1) / 2
    return points_screen

def screen_normalized_to_pixel(screen_points : np.ndarray, image_shape : Tuple[int, int]):
    """Converts screen-normalized points to pixel coordinates."""
    points_px = screen_points[0:2, ...] * np.flip(np.array(image_shape)).T[:,np.newaxis]
    points_px[1,...] = image_shape[0] - points_px[1,...]
    return points_px

def project_to_image_px_coords(camera : pyrender.Camera, eye_space_points : np.ndarray, image_shape : Tuple[int, int]):
    """Project eye space points to pixel coordinates."""
    points_screen = project_to_screen_normalized_coords(camera, eye_space_points)
    points_px = screen_normalized_to_pixel(points_screen, image_shape)
    return points_px