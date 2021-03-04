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
"""Configuration information for dice dot points. Ensure this matches positions of dots on the rendered 3d mesh."""

import numpy as np

def get_dot_indices(face_index_from_1):
    """Get indices (from 0 through 20) corresponding to dots on the given face (from 1 through 6)."""
    if face_index_from_1 == 1:
        return [0]
    elif face_index_from_1 == 2:
        return [1,2]
    elif face_index_from_1 == 3:
        return [3,4,5]
    elif face_index_from_1 == 4:
        return [6,7,8,9]
    elif face_index_from_1 == 5:
        return [10,11,12,13,14]
    elif face_index_from_1 == 6:
        return [15,16,17,18,19,20]
    else:
        return []

def get_all_dot_indices(face_indices_from_1):
    """Gets all dot indices for a given set of face indices."""
    stacked = np.hstack([get_dot_indices(face_id) for face_id in face_indices_from_1])
    return stacked

def get_local_dot_positions_sunken():
    """Gets dot positions in local coordinate space of a die. 
    Considers height of the center of each dot, as sunken relative to surrounding die face.
    Note that these positions are representative of a given die model, and may not be fully accurate for all types of dice."""
    return np.array([[0.00074,0.01146,-0.49888],
                     [-0.21756,0.48146,-0.24718],
                     [0.21904,0.48146,0.18942],
                     [0.47074,0.22976,0.18942],
                     [0.47074,0.01146,-0.02888],
                     [0.47074,-0.20684,-0.24718],
                     [0.21904,-0.45854,0.18942],
                     [-0.21756,-0.45854,0.18942],
                     [0.21904,-0.45854,-0.24718],
                     [-0.21756,-0.45854,-0.24718],
                     [-0.46926,0.01146,-0.02888],
                     [-0.46926,-0.20684,0.18942],
                     [-0.46926,0.22976,0.18942],
                     [-0.46926,0.22976,-0.24718],
                     [-0.46926,-0.20684,-0.24718],
                     [-0.21756,-0.20684,0.44112],
                     [0.00074,-0.20684,0.44112],
                     [0.21904,-0.20684,0.44112],
                     [0.21904,0.22976,0.44112],
                     [0.00074,0.22976,0.44112],
                     [-0.21756,0.22976,0.44112]
                      ]).T

def get_local_dot_positions_unsunken_at_die_face():
    """Gets dot positions in local coordinate space of a die. 
    Considers height of the center of each dot, at the height of the surrounding die face plane.
    Note that these positions are representative of a given die model, and may not be fully accurate for all types of dice."""
    return np.array([[0.00074,0.01146,-0.52908],
                     [-0.21756,0.51166,-0.24718],
                     [0.21904,0.51166,0.18942],
                     [0.50094,0.22976,0.18942],
                     [0.50094,0.01146,-0.02888],
                     [0.50094,-0.20684,-0.24718],
                     [0.21904,-0.48874,0.18942],
                     [-0.21756,-0.48874,0.18942],
                     [0.21904,-0.48874,-0.24718],
                     [-0.21756,-0.48874,-0.24718],
                     [-0.49946,0.01146,-0.02888],
                     [-0.49946,-0.20684,0.18942],
                     [-0.49946,0.22976,0.18942],
                     [-0.49946,0.22976,-0.24718],
                     [-0.49946,-0.20684,-0.24718],
                     [-0.21756,-0.20684,0.47132],
                     [0.00074,-0.20684,0.47132],
                     [0.21904,-0.20684,0.47132],
                     [0.21904,0.22976,0.47132],
                     [0.00074,0.22976,0.47132],
                     [-0.21756,0.22976,0.47132]
                      ]).T

def get_local_dot_positions_unsunken_at_dot_edge():
    """Gets dot positions in local coordinate space of a die. 
    Considers height of the center of each dot, at the height of the edge loop of each dot.
    Note that these positions are representative of a given die model, and may not be fully accurate for all types of dice."""
    return np.array([[0.00074,0.01146,-0.52335],
                     [-0.21756,0.50592,-0.24718],
                     [0.21904,0.50592,0.18942],
                     [0.4952,0.22976,0.18942],
                     [0.4952,0.01146,-0.02888],
                     [0.4952,-0.20684,-0.24718],
                     [0.21904,-0.483,0.18942],
                     [-0.21756,-0.483,0.18942],
                     [0.21904,-0.483,-0.24718],
                     [-0.21756,-0.483,-0.24718],
                     [-0.49372,0.01146,-0.02888],
                     [-0.49372,-0.20684,0.18942],
                     [-0.49372,0.22976,0.18942],
                     [-0.49372,0.22976,-0.24718],
                     [-0.49372,-0.20684,-0.24718],
                     [-0.21756,-0.20684,0.46558],
                     [0.00074,-0.20684,0.46558],
                     [0.21904,-0.20684,0.46558],
                     [0.21904,0.22976,0.46558],
                     [0.00074,0.22976,0.46558],
                     [-0.21756,0.22976,0.46558]
                      ]).T

def get_unsunken_at_dot_edge_instead_of_die_face():
    """Configuration if unsunken dots should be returned at height of dot edge instead of surrounding face."""
    return True

def get_local_dot_positions(unsunken = True):
    """Gets the dot positions in the local coordinate space of a die."""
    if unsunken:
        if get_unsunken_at_dot_edge_instead_of_die_face():
            return get_local_dot_positions_unsunken_at_dot_edge()
        else:
            return get_local_dot_positions_unsunken_at_die_face()
    else:
        return get_local_dot_positions_sunken()

def get_local_dot_normals():
    """Gets the dot normals in the local coordinate space of a die."""
    return np.array([[0,0,-1.0],
                     [0,1.0,0],
                     [0,1.0,0],
                     [1.0,0,0],
                     [1.0,0,0],
                     [1.0,0,0],
                     [0,-1.0,0],
                     [0,-1.0,0],
                     [0,-1.0,0],
                     [0,-1.0,0],
                     [-1.0,0,0],
                     [-1.0,0,0],
                     [-1.0,0,0],
                     [-1.0,0,0],
                     [-1.0,0,0],
                     [0,0,1.0],
                     [0,0,1.0],
                     [0,0,1.0],
                     [0,0,1.0],
                     [0,0,1.0],
                     [0,0,1.0]
                      ]).T

def get_local_face_normals():
    """Gets the six face normals in the local coordinate space of a die."""
    return np.array([[0,0,-1.0],
                     [0,1.0,0],
                     [1.0,0,0],
                     [0,-1.0,0],
                     [-1.0,0,0],
                     [0,0,1.0],
                      ]).T

def get_local_face_forward_index_from_1(face_up_index_from_1 : int):
    """Given an upward face index, gets the corresponding face index for the horizontal forward-facing face (defined at zero rotation around vertical axis)."""
    index_of_forward = None
    if face_up_index_from_1 == 1:
        index_of_forward = 2
    elif face_up_index_from_1 == 2:
        index_of_forward = 1
    elif face_up_index_from_1 == 3:
        index_of_forward = 1
    elif face_up_index_from_1 == 4:
        index_of_forward = 1
    elif face_up_index_from_1 == 5:
        index_of_forward = 1
    elif face_up_index_from_1 == 6:
        index_of_forward = 2
    return index_of_forward

def get_local_face_forward(face_up_index_from_1 : int):
    """Given an upward face index, gets the corresponding face normal for the horizontal forward-facing face (defined at zero rotation around vertical axis)."""
    forward_index_from_1 = get_local_face_forward_index_from_1(face_up_index_from_1)
    forward_index = forward_index_from_1 - 1
    local_face_normals = get_local_face_normals()
    local_forward = local_face_normals[:, forward_index]
    return local_forward

def get_local_bounding_box_min_max():
    """Gets an Axis-Aligned Bounding Box for the canonical die model, in local coordinate space."""
    return np.array([[-0.49946,-0.48874,-0.52908],
                     [0.50094,0.51166,0.47132]]).T

def get_local_bounding_box_corners():
    """Gets corners of the canonical Axis-Aligned Bounding Box in local coordinate space."""
    min_max = get_local_bounding_box_min_max().T
    return np.array([[min_max[0,0],min_max[0,1],min_max[0,2]],
                     [min_max[0,0],min_max[1,1],min_max[0,2]],
                     [min_max[1,0],min_max[1,1],min_max[0,2]],
                     [min_max[1,0],min_max[0,1],min_max[0,2]],
                     [min_max[0,0],min_max[0,1],min_max[1,2]],
                     [min_max[0,0],min_max[1,1],min_max[1,2]],
                     [min_max[1,0],min_max[1,1],min_max[1,2]],
                     [min_max[1,0],min_max[0,1],min_max[1,2]]
                     ]).T

def get_local_bounding_box_horizontal_radius():
    """Gets the horizontal radius of the canonical Axis-Aligned Bounding Box in local coordinate space."""
    min_max = get_local_bounding_box_min_max()
    min_max_xz = min_max[[0,2],:]
    abs_min_max_xz = np.abs(min_max_xz)
    max_xz = np.max(abs_min_max_xz, axis = 1)
    radius = np.linalg.norm(max_xz)
    return radius

def get_dot_radius():
    """Gets the canonical dot radius."""
    radius = 0.067989 #diameter 0.13598
    return radius

def _get_circle_points(point, normal, radius):
    """Gets points around the edge of a circle, given a point, normal, and radius."""
    import GramSchmidt
    axis_a, axis_b = GramSchmidt.get_orthogonal_axes(normal)
    num_angles = 360
    angles = np.array([np.deg2rad(i) for i in range(num_angles)])
    a_axis_coeffs, b_axis_coeffs = (np.cos(angles) * radius, np.sin(angles) * radius)
    points_around_circle = axis_a[...,np.newaxis] * a_axis_coeffs + axis_b[...,np.newaxis] * b_axis_coeffs 
    points = points_around_circle + point[..., np.newaxis]
    return points

def get_dot_edge_points(points, normals, unsunken):
    """Gets points corresponding to the edge of dot circles."""
    sunken_shift = 0.0245
    unsunken_shift = 0.0 if get_unsunken_at_dot_edge_instead_of_die_face() else -0.00454
    shift = unsunken_shift if unsunken else sunken_shift
    center_points = points + normals * shift
    list_of_points = [_get_circle_points(center_points[:,i], normals[:,i], get_dot_radius()) for i in range(center_points.shape[1])]
    return list_of_points
