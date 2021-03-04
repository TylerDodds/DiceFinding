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
"""Transfoorms/projects visible dice dots to camera or screen space."""

import numpy as np

import pyrender

import DiceConfig
import ProjectionUtil
from TransformUtil import transform_points_3d
from CoordsHelper import Pose

from typing import Tuple

def pose_inverse(pose):
    return np.linalg.inv(pose)

def _get_dice_to_camera_transform(camera_node : pyrender.Node, dice_node : pyrender.Node, scene: pyrender.Scene):
    """Gets the transformation matrix from dice node to camera node coordinates."""
    dice_scene_pose = scene.get_pose(dice_node)
    camera_scene_pose = scene.get_pose(camera_node)
    scene_to_camera_pose = pose_inverse(camera_scene_pose)
    dice_to_camera_transform = scene_to_camera_pose @ dice_scene_pose
    return dice_to_camera_transform

def get_eye_space_dot_point_normals(camera_node : pyrender.Node, dice_node : pyrender.Node, scene: pyrender.Scene):
    """Gets eye-space points and normals of dots given a camera node and a dice node."""
    dice_to_camera_transform = _get_dice_to_camera_transform(camera_node, dice_node, scene)
    
    points = DiceConfig.get_local_dot_positions()
    normals = DiceConfig.get_local_dot_normals()
    points_transformed = transform_points_3d(points, dice_to_camera_transform, False)
    normals_transformed = transform_points_3d(normals, dice_to_camera_transform, True)

    return points_transformed, normals_transformed

def get_are_eye_space_point_normals_facing_camera(points : np.ndarray, normals : np.ndarray):
    """Gets a true-false mask of which eye-space points and normals are facing the camera."""
    points_in_front_of_camera = points[2, :] < 0 #Since positive z-axis is 'backwards'
    #normals_pointing_towards_camera = normals[2,:] > 0 #Since positive z-axis is 'backwards' #NB This normal test is not correct! It depends not on z-axis direction, but relative to direction from camera to point
    dot_normals_points = np.sum(points * normals, axis = 0)
    normals_pointing_towards_camera = dot_normals_points < 0 #Since positive z-axis is 'backwards'
    are_visible = np.logical_and(points_in_front_of_camera, normals_pointing_towards_camera)
    return are_visible

def filter_eye_space_camera_facing_dot_point_normals(points : np.ndarray, normals : np.ndarray):
    """Returns eye-space points and normals, filtering out points not facing the camera."""
    are_visible = get_are_eye_space_point_normals_facing_camera(points, normals)
    visible_points = points[:, are_visible]
    visible_normals = normals[:, are_visible]
    return visible_points, visible_normals

def get_eye_space_camera_facing_dot_point_normals(camera_node : pyrender.Node, dice_node : pyrender.Node, scene: pyrender.Scene):
    """Gets eye-space points and normals facing the camera, given a camera node and dice node."""
    points, normals = get_eye_space_dot_point_normals(camera_node, dice_node, scene)
    visible_points, visible_normals = filter_eye_space_camera_facing_dot_point_normals(points, normals)
    return visible_points, visible_normals

def get_dots_mask_facing_camera_from_eye_space_pose(die_eye_space_pose : Pose):
    """Gets a true-false mask of which dots face the camera, given the die's eye-space pose."""
    dice_transform = die_eye_space_pose.as_transformation_matrix()
    dot_points = DiceConfig.get_local_dot_positions()
    dot_normals = DiceConfig.get_local_dot_normals()
    dot_points_transformed =  transform_points_3d(dot_points, dice_transform, at_infinity=False)
    dot_normals_transformed =  transform_points_3d(dot_normals, dice_transform, at_infinity=True)
    dots_are_visible = get_are_eye_space_point_normals_facing_camera(dot_points_transformed, dot_normals_transformed)
    return dots_are_visible

def get_local_dots_facing_camera_from_eye_space_pose(die_eye_space_pose : Pose, transpose : bool = True):
    """Gets local-to-die-coordinate dots that are facing the camera, given the die's eye-space pose.
    Returns Nx3 matrix if transposed, 3xN otherwise."""
    dots_are_visible = get_dots_mask_facing_camera_from_eye_space_pose(die_eye_space_pose)
    dot_points = DiceConfig.get_local_dot_positions()
    local_dots_visible_in_eye_space = dot_points[:, dots_are_visible]
    return local_dots_visible_in_eye_space.T if transpose else local_dots_visible_in_eye_space

def get_points_within_image_bounds(points_px : np.ndarray, image_shape : Tuple[int, int]):
    """Get px-space points within bounds of given image_shape."""
    points_above_corner = np.all(points_px > 0, axis = 0)
    points_below_corner = np.all(points_px < np.array(image_shape)[::-1,np.newaxis], axis = 0)
    points_within_image = np.logical_and(points_above_corner, points_below_corner)
    return points_within_image

def get_eye_px_space_dot_pt_normals(camera_node : pyrender.Node, dice_node : pyrender.Node, scene: pyrender.Scene, image_shape : Tuple[int, int]):
    """Get px-coordinate points of die dots facing camera, and within image, given a camera and dice node."""
    facing_points, facing_normals = get_eye_space_camera_facing_dot_point_normals(camera_node, dice_node, scene)

    points_px = ProjectionUtil.project_to_image_px_coords(camera_node.camera, facing_points, image_shape)
    points_within_image = get_points_within_image_bounds(points_px, image_shape)

    facing_points = facing_points[:,points_within_image]
    facing_normals = facing_normals[:,points_within_image]
    points_px = points_px[:, points_within_image]
    return facing_points, facing_normals, points_px

def sample_buffer(points_px : np.ndarray, buffer : np.ndarray, get_interpolated : bool):
    """Sample a buffer at a set of pixel coordinates, with potential interpolation."""
    if get_interpolated:
        from scipy.interpolate import interp2d
        interpolator = interp2d(np.arange(buffer.shape[1]), np.arange(buffer.shape[0]), buffer)
        interpolated = np.hstack([interpolator(points_px[0,i], points_px[1,i]) for i in range(points_px.shape[1])])
        result = interpolated
    else:
        points_round = np.rint(points_px).astype(int)
        points_round[0,:] = np.clip(points_round[0,:], 0, buffer.shape[0] - 1)
        points_round[1,:] = np.clip(points_round[1,:], 0, buffer.shape[1] - 1)
        nearest = np.array([buffer[pt[1], pt[0]] for pt in points_round.T])
        result = nearest
    return result

def get_eye_space_dot_point_normals_visible_at_mask(camera_node : pyrender.Node, dice_node : pyrender.Node, scene: pyrender.Scene, mask : np.ndarray, dice_mask_index : int, also_return_points_px : bool):
    """Gets eye-space dot points and normals of die dots facing camera, and within image, given a camera and dice node, corresponding to a given die's mask index and mask image array."""
    facing_points, facing_normals, points_px = get_eye_px_space_dot_pt_normals(camera_node, dice_node , scene, mask.shape)

    mask_values = sample_buffer(points_px, mask, get_interpolated = False)
    points_equal_mask = mask_values == dice_mask_index

    not_occluded_points = facing_points[:, points_equal_mask]
    not_occluded_normals = facing_normals[:, points_equal_mask]

    if also_return_points_px:
        return not_occluded_points, not_occluded_normals, points_px
    else:
        return not_occluded_points, not_occluded_normals

def get_image_space_dot_points(camera_node : pyrender.Node, dice_node : pyrender.Node, scene: pyrender.Scene, mask : np.ndarray, dice_mask_index : int):
    """Gets pixel-coordinate dot points given a camera node, dice node, mask and mask index."""
    pts, normals, pts_px = get_eye_space_dot_point_normals_visible_at_mask(camera_node, dice_node, scene, mask, dice_mask_index, also_return_points_px = True)
    return pts_px

def get_eye_space_corner_points(camera_node : pyrender.Node, dice_node : pyrender.Node, scene: pyrender.Scene):
    """Gets eye-space corner points of a dice node's bounding box."""
    dice_to_camera_transform = _get_dice_to_camera_transform(camera_node, dice_node, scene)
    points = DiceConfig.get_local_bounding_box_corners()
    points_transformed = transform_points_3d(points, dice_to_camera_transform, False)
    return points_transformed

def get_eye_space_mesh_points(camera_node : pyrender.Node, dice_node : pyrender.Node, scene: pyrender.Scene):
    """Gets pixel-coordinate die 3D mesh vertex points, given a camera node, dice node."""
    dice_to_camera_transform = _get_dice_to_camera_transform(camera_node, dice_node, scene)
    points = np.vstack([primitive.positions for primitive in dice_node.mesh.primitives]).T
    points_transformed = transform_points_3d(points, dice_to_camera_transform, False)
    return points_transformed

def get_image_space_bounding_box(points_px : np.ndarray):
    """Gets an encapsulating bounding box in 2D."""
    points_px_min = np.min(points_px, axis = 1)
    points_px_max = np.max(points_px, axis = 1)
    width_height_px = points_px_max - points_px_min
    min_x_y_width_height_px = np.hstack((points_px_min, width_height_px))
    return min_x_y_width_height_px

def get_image_space_dot_bounds(camera_node : pyrender.Node, dice_node : pyrender.Node, scene: pyrender.Scene, mask : np.ndarray, dice_mask_index : int, unsunken : bool):
    """Returns image-space bounding boxes for dots, given a dice and camera node, as well as mask image and index.
    :param bool unsunken: If dot locations should be considered on the plane of unsunken edges, rather than the sunken center.
    """
    pts, normals, pts_px = get_eye_space_dot_point_normals_visible_at_mask(camera_node, dice_node, scene, mask, dice_mask_index, also_return_points_px = True)
    eye_space_dot_circles = DiceConfig.get_dot_edge_points(pts, normals, unsunken)
    eye_space_dot_circles_px = [ProjectionUtil.project_to_image_px_coords(camera_node.camera, points, mask.shape) for points in eye_space_dot_circles]
    def get_within_mask(pts_px):
        mask_values = sample_buffer(pts_px, mask, get_interpolated = False)
        points_equal_mask = mask_values == dice_mask_index
        pts_px_at_mask = pts_px[:, points_equal_mask]
        return pts_px_at_mask
    eye_space_dot_circles_px_in_mask = [get_within_mask(pts) for pts in eye_space_dot_circles_px]
    bounding_boxes = [get_image_space_bounding_box(pts) for pts in eye_space_dot_circles_px_in_mask]
    return bounding_boxes

def get_image_space_full_bounding_box_from_eye_space_points(camera_node : pyrender.Node, image_shape : np.ndarray, eye_space_points : np.ndarray):
    """Gets the image space bounding box containing given eye-space points, given the camera node and image shape."""
    eye_points_screen = ProjectionUtil.project_to_screen_normalized_coords(camera_node.camera, eye_space_points)
    eye_points_px = ProjectionUtil.screen_normalized_to_pixel(eye_points_screen, image_shape)
    min_and_width = get_image_space_bounding_box(eye_points_px)
    return min_and_width

def get_image_space_full_bounding_box_from_mesh(camera_node : pyrender.Node, dice_node : pyrender.Node, scene: pyrender.Scene, image_shape : np.ndarray):
    """Gets the image space bounding box containing mesh points of the dice model, given the camera and dice nodes."""
    eye_space_mesh_points = get_eye_space_mesh_points(camera_node, dice_node, scene)
    return get_image_space_full_bounding_box_from_eye_space_points(camera_node, image_shape, eye_space_mesh_points)

def get_scene_space_up_face_index(dice_node : pyrender.Node, scene: pyrender.Scene):
    """Gets the face-up index of a given dice node.
    Indices start at 1, for face with 1 dot up"""
    face_normals = DiceConfig.get_local_face_normals()
    dice_pose = scene.get_pose(dice_node)
    scene_face_normals = transform_points_3d(face_normals, dice_pose, True)
    scene_face_normals_y = scene_face_normals[1,:]
    index = np.argmax(scene_face_normals_y)
    index_starting_at_1 = index + 1
    return index_starting_at_1

def get_y_rotation_angle_relative_to_camera(dice_node : pyrender.Node, scene: pyrender.Scene, dice_top_face_index : int):
    """Gets the rotation of a dice node around its vertical aspect with respect to the camera, given the top face index.
    Note that for each upward-pointing face is defined another face as the forward-pointing face given by zero y-rotation."""
    dice_to_camera_transform = _get_dice_to_camera_transform(scene.main_camera_node, dice_node, scene)
    dice_pose = scene.get_pose(dice_node)

    local_forward_axis = DiceConfig.get_local_face_forward(dice_top_face_index)[:, np.newaxis]
    dice_scene_forward = transform_points_3d(local_forward_axis, dice_pose, True)
    dice_scene_forward[1] = 0
    dice_scene_forward /= np.linalg.norm(dice_scene_forward)

    dice_to_camera_translation = dice_to_camera_transform[0:3,3]
    direction_camera_from_dice = -dice_to_camera_translation[:, np.newaxis]
    direction_camera_from_dice[1] = 0
    direction_camera_from_dice /= np.linalg.norm(direction_camera_from_dice)

    #Note x axis takes place of 'y' in arctan while z axis takes place of 'x' since y-axis rotations rotates x into -z (equivalently, z into x)
    angle_forward = np.arctan2(dice_scene_forward[0], dice_scene_forward[2])
    angle_direction = np.arctan2(direction_camera_from_dice[0], direction_camera_from_dice[2])
    angle_difference = angle_forward - angle_direction
    angle_difference_wrapped = (angle_difference + np.pi) % (2.0 * np.pi) - np.pi
    return angle_difference_wrapped
