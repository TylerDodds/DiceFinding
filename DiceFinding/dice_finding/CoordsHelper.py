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
"""Defines a Pose class (3d position androtation), allowing conversion to/from transformation matrices and between opencv and pyrender coordinate systems."""

import numpy as np
import pyrender
import cv2


class Pose(object):
    """Representation of a pose (translation and rotation as Rodrigues vector)."""
    def __init__(self, found : bool, rotation_rodrigues : np.ndarray, translation : np.ndarray):
        if found and np.isfinite(rotation_rodrigues).all() and np.isfinite(translation).all():
            self.found = True
        else:
            self.found = False
        self.rotation_rodrigues = None if not found else np.squeeze(rotation_rodrigues)[:, np.newaxis]
        self.translation = None if not found else np.squeeze(translation)[:, np.newaxis]
    def __iter__(self):
        yield from [self.found, self.rotation_rodrigues, self.translation]
    def __bool__(self):
        return self.found
    def __repr__(self):
        return "{} {} {}".format(self.round, self.rotation_rodrigues, self.translation)

    def as_transformation_matrix(self):
        """Convers pose to 4x4 transformation matrix."""
        transformation = np.eye(4)
        rotation_matrix, rotation_jacobian = cv2.Rodrigues(self.rotation_rodrigues)
        transformation[0:3, 0:3] = rotation_matrix
        transformation[0:3, 3] = np.squeeze(self.translation)
        return transformation

    @staticmethod
    def create_from_transformation_matrix(transformation_matrix : np.ndarray):
        """Creates a pose from a 4x4 transformation matrix."""
        rotation_vector, rotation_jacobian = cv2.Rodrigues(transformation_matrix[0:3, 0:3])
        translation = transformation_matrix[0:3, 3]
        return Pose(True, rotation_vector, translation)

    def get_converted_between_cv_and_pyrender_coords(self):
        """Gets a pose converted between OpenCV and pyrender coordinates (or vice-versa)."""
        def convert_transformation_cv_to_pyrender_coords(transformation_matrix_cv : np.ndarray):
            transformed = np.diag([1,-1,-1,1]) @ transformation_matrix_cv
            return transformed
        matrix = self.as_transformation_matrix()
        matrix_converted = convert_transformation_cv_to_pyrender_coords(matrix)
        return Pose.create_from_transformation_matrix(matrix_converted)

    def as_numpy_array(self):
        """Returns stacked rotation (as Rodrigues vector) and translation length-six numpy array."""
        stacked = np.vstack((self.rotation_rodrigues, self.translation))
        return stacked

    @staticmethod
    def create_from_numpy_array(contiguous_array : np.ndarray):
        """Parses a length-six numpy array into Rodrigues rotation and translation components of a Pose."""
        return Pose(True, contiguous_array[0:3], contiguous_array[3:6])

    @staticmethod
    def create_from_cv_results(cv_results):
        """Create a Pose from OpenCV methods returning pose components."""
        return Pose(cv_results[0], cv_results[1], cv_results[2])
