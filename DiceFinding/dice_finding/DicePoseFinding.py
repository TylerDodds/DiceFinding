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
"""Calculates best-fit dice poses from an image and estimated bounding boxes, classes, and y-rotations of dice and dots."""

import numpy as np
import tensorflow.compat.v2 as tf
import cv2
from pycpd import AffineRegistration, RigidRegistration
from collections import defaultdict
import copy

import DiceConfig
from CoordsHelper import Pose
from typing import Dict, List, Sequence
from DiceProjection import get_local_dots_facing_camera_from_eye_space_pose

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

distance_scale_px = 8.0 #Distance scale for calculating non-linear loss functions, similar to distance functions used in robust nonlinear regression, like https://scipy-cookbook.readthedocs.io/items/robust_regression.html
approx_distance_score_cutoff_fraction = 1.25 #Fraction of minimum Pose score (of approximate poses) that should be compared by full pose calculation
rounded_rectangle_radius = 0.35 #Radius of rounded rectangle for each die's bounding box. Dots at the corners outside the rounded radius are excluded, at they will belong to other die.
inlier_cutoff_px = 10

class PoseResult(object):
    """Represents the result of a pose fit, given in both pyrender and opencv coordinates.
    Also calculates comparison against found dot positions in the image."""
    def __init__(self, pose : Pose, additional_data : Dict, in_pyrender_coords : bool, distance_scale : float = distance_scale_px):
        if in_pyrender_coords:
            self.pose_pyrender = pose
            self.pose_cv = pose.get_converted_between_cv_and_pyrender_coords()
        else:
            self.pose_cv = pose
            self.pose_pyrender = pose.get_converted_between_cv_and_pyrender_coords()
        self.additional_data = additional_data
        self.distance_scale = distance_scale
        self.comparison_points_cv = None
        self.comparison_camera_matrix = None
        self.comparison_distortion_coefficients = None
        self.projected_points = None
        self.comparison_projected_distances = None
        self.comparison_soft_l1_distances = None
        self.comparison_cauchy_distances = None
        self.comparison_arctan_distances = None
        self.assignment_score_function = None
        self.assignment_scores = None
        self.comparison_indices = None
        self.projected_indices = None
        self.matched_scores = None
        self.matched_scores_rms = None

        self.calculate_inliers_within_bounding_box = False

    @property
    def has_comparison(self):
        """If comparison has been calculated."""
        return self.comparison_results is not None

    def calculate_comparison(self, dot_centers_cv : np.ndarray, camera_matrix : np.ndarray, distortion_coefficients : np.ndarray):
        """Calculates comparison against the given found dot center points, using the given camera matrix and distortion coefficients to perform projection."""
        self.comparison_points_cv = dot_centers_cv
        self.comparison_camera_matrix = camera_matrix
        self.comparison_distortion_coefficients = distortion_coefficients
        self._perform_comparison()

    def get_soft_l1_distance_scores(self, distances : np.ndarray):
        """Calculates soft-l1 distance scores for a set of distances."""
        return self.distance_scale * 2.0 * (np.sqrt(1.0 + (distances/self.distance_scale)**2) - 1.0)

    def get_cauchy_distance_scores(self, distances : np.ndarray):
        """Calculates Cauchy distance scores for a set of distances."""
        return self.distance_scale * np.log1p((distances/self.distance_scale)**2)

    def get_arctan_distance_scores(self, distances : np.ndarray):
        """Calculates arctan distance scores for a set of distances."""
        return self.distance_scale * np.arctan((distances/self.distance_scale)**2)

    def _perform_comparison(self):
        """Performs comparison against the found dot center points, using the given camera matrix and distortion coefficients to perform projection."""
        local_dots_visible_in_eye_space = get_local_dots_facing_camera_from_eye_space_pose(self.pose_pyrender)
        pose_points, pose_points_jacobian = cv2.projectPoints(local_dots_visible_in_eye_space, self.pose_cv.rotation_rodrigues, self.pose_cv.translation, self.comparison_camera_matrix, self.comparison_distortion_coefficients)
        self.projected_points = np.squeeze(pose_points, axis = 1)
        #For matching points: See https://stackoverflow.com/questions/41936760/how-to-find-a-unique-set-of-closest-pairs-of-points which suggests using https://en.wikipedia.org/wiki/Hungarian_algorithm on assignment problem. See https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
        self.comparison_projected_distances = cdist(self.comparison_points_cv, self.projected_points)#MxN matrix with M comparison_points and N projected points
        self.comparison_soft_l1_distances = self.get_soft_l1_distance_scores(self.comparison_projected_distances)
        self.comparison_cauchy_distances = self.get_cauchy_distance_scores(self.comparison_projected_distances)
        self.comparison_arctan_distances = self.get_arctan_distance_scores(self.comparison_projected_distances)
        self.assignment_score_function = lambda d : self.get_arctan_distance_scores(d)
        self.assignment_scores = self.assignment_score_function(self.comparison_projected_distances)
        
        self.comparison_indices, self.projected_indices = linear_sum_assignment(self.assignment_scores)#Returns (row_indices, column_indices)
        self.matched_scores = self.assignment_scores[self.comparison_indices, self.projected_indices]
        self.matched_scores_rms = np.sqrt(np.mean(np.square(self.matched_scores)))

    def calculate_inliers(self, dot_bounding_boxes_sizes_cv):
        """Calculates inliers of projected against found points, potentially including the found dot bounding box sizes in addition to their positions."""
        projected_points_ordered = self.projected_points[self.projected_indices, :]
        found_points_ordered = self.comparison_points_cv[self.comparison_indices, :]
        differences = projected_points_ordered - found_points_ordered
        if self.calculate_inliers_within_bounding_box:
            found_bb_sizes_ordered = dot_bounding_boxes_sizes_cv[self.comparison_indices, :]
            difference_bb_fractions = differences / found_bb_sizes_ordered
            abs_bb_fractions = np.abs(difference_bb_fractions)
            max_bb_fraction = np.max(abs_bb_fractions, axis = -1)
            inliers = max_bb_fraction < 1.0
        else:
            center_distances = np.linalg.norm(differences, axis = -1)
            inliers = center_distances < inlier_cutoff_px
        self.comparison_inlier_indices = tf.boolean_mask(self.comparison_indices, inliers)
        self.projected_inlier_indices = tf.boolean_mask(self.projected_indices, inliers)

def _delete_tf(tensor, idx, axis=0):
    """Deletes from a tensor along an axis at the given index."""
    n = tf.shape(tensor)[axis]
    t = tf.ones_like(idx, dtype=tf.bool)
    m = ~tf.scatter_nd(tf.expand_dims(idx, 1), t, [n])
    return tf.boolean_mask(tensor, m, axis=axis)

def _bounding_box_intersection(bounding_box_cv_1 : np.ndarray, bounding_box_cv_2 : np.ndarray):
    """Gets the intersection of two OpenCV-coordinate bounding boxes."""
    x_min = np.max(bounding_box_cv_1[0], bounding_box_cv_2[0])
    y_min = np.max(bounding_box_cv_1[1], bounding_box_cv_2[1])
    x_max = np.min(bounding_box_cv_1[2], bounding_box_cv_2[2])
    y_max = np.min(bounding_box_cv_1[3], bounding_box_cv_2[3])
    intersection = np.array([x_min, y_min, x_max, y_max])
    return intersection

def _bounding_box_area(bounding_box_cv : np.ndarray, clip_negative = True):
    """Gets the area of a bounding box."""
    width = bounding_box_cv[2] - bounding_box_cv[0]
    height = bounding_box_cv[3] - bounding_box_cv[1]
    if clip_negative:
        width = np.max(width, 0.0)
        height = np.max(height, 0.0)
    area = width * height
    return area

def _intersection_over_union(bounding_box_cv_1 : np.ndarray, bounding_box_cv_2 : np.ndarray):
    """Calculates the intersection over union for two bounding boxes in OpenCV coordinates."""
    intersection_bb = _bounding_box_intersection(bounding_box_cv_1, bounding_box_cv_2)
    area_intersection = _bounding_box_area(intersection_bb)
    area_1 = _bounding_box_area(bounding_box_cv_1)
    area_2 = _bounding_box_area(bounding_box_cv_2)
    iou = area_intersection / (area_1 + area_2 - area_intersection)
    return iou

def _get_dot_centers(dot_bounding_boxes):
    """Gets the center of Tensor bounding boxes."""
    axis_0 = tf.gather(dot_bounding_boxes, [0, 2], axis = -1)
    axis_1 = tf.gather(dot_bounding_boxes, [1, 3], axis = -1)
    axis_0_average = tf.reduce_mean(axis_0, axis = -1)
    axis_1_average = tf.reduce_mean(axis_1, axis = -1)
    centers = tf.stack([axis_0_average, axis_1_average], axis = -1)

    return centers

def _get_dot_sizes(dot_bounding_boxes):
    """Gets the size of Tensor bounding boxes."""
    axis_0 = dot_bounding_boxes[:, 2] - dot_bounding_boxes[:, 0]
    axis_1 = dot_bounding_boxes[:, 3] - dot_bounding_boxes[:, 1]
    sizes = tf.stack([axis_0, axis_1], axis = -1)
    return sizes

def _get_die_local_up_forward_right_axes(die_class : int):
    """Gets the local up, forward and right axes defined for a given die class (top face)."""
    die_local_face_normals = DiceConfig.get_local_face_normals()
    die_local_forward_axis = DiceConfig.get_local_face_forward(die_class)
    die_local_up_axis = die_local_face_normals[:, die_class - 1]
    die_local_right_axis = np.cross(die_local_up_axis, die_local_forward_axis)
    return die_local_up_axis, die_local_forward_axis, die_local_right_axis

def _get_approximate_die_pose(die_class : int, y_angle_deg : float, bounding_box_pose_result : PoseResult, x_axis_rotations_deg : Sequence[float], y_rot_offsets_deg : Sequence[float]) -> PoseResult:
    """Gets an approximate pose given the die class, rotation around vertical axes, and a rough pose result estimated from the bounding box.
    Checks the given set of rotation offsets rotations around x and y axes."""
    bb_pnp_res, bb_rotation, bb_translation = bounding_box_pose_result.pose_cv
    if bb_pnp_res:
        from scipy.spatial.transform import Rotation

        bb_translation_pyrender_coords = bb_translation * np.array([1, -1, -1])[:,np.newaxis]
        angle_of_translation_deg = np.rad2deg(np.arctan2(-bb_translation_pyrender_coords[0], -bb_translation_pyrender_coords[2]))
        y_angle_deg_with_position_offset = y_angle_deg + angle_of_translation_deg

        pose_results = []
        die_local_up_axis, die_local_forward_axis, die_local_right_axis = _get_die_local_up_forward_right_axes(die_class)
        for y_rot_offset_deg in y_rot_offsets_deg:
            y_angle_final = y_angle_deg_with_position_offset + y_rot_offset_deg
            angle_cos = np.cos(np.deg2rad(y_angle_final))
            angle_sin = np.sin(np.deg2rad(y_angle_final))
            die_local_to_scene_rotation = np.eye(3)
            die_local_to_scene_rotation[0, 0:3] = angle_cos * die_local_right_axis + angle_sin * die_local_forward_axis
            die_local_to_scene_rotation[1, 0:3] = die_local_up_axis 
            die_local_to_scene_rotation[2, 0:3] = - angle_sin * die_local_right_axis + angle_cos * die_local_forward_axis

            for x_axis_rotation_deg in x_axis_rotations_deg:
                x_axis_rotation = Rotation.from_euler('x', x_axis_rotation_deg, degrees = True)
                x_axis_rotation_matrix = x_axis_rotation.as_matrix()
                die_combined_rotation = x_axis_rotation_matrix @ die_local_to_scene_rotation

                die_rotation_rodrigues, die_rotation_rodrigues_jacobian = cv2.Rodrigues(die_combined_rotation)
                #NB Return in 'pyrender' coordinates
                approx_pose = Pose(True, die_rotation_rodrigues, bb_translation_pyrender_coords)
                pose_result = PoseResult(approx_pose, {}, in_pyrender_coords = True)
                pose_result.y_angle = y_angle_final
                pose_result.y_angle_rel = y_angle_deg + y_rot_offset_deg
                pose_result.x_angle = x_axis_rotation_deg
                pose_results.append(pose_result)
        return pose_results
    else:
        raise NotImplementedError("Cannot get approximate die pose if bounding box PNP was not found.")

def _match_points_with_point_cloud_registration(dot_centers_cv : np.ndarray, local_dots_to_project : np.ndarray, approximate_pose_result : PoseResult, camera_matrix : np.ndarray, distortion_coefficients : np.ndarray, matching_style = 'rigid_registration'):
    """Matches projected (from approximate pose result) with found dot center points, using Coherent Point Drift Algorithm registration between the 2D sets of points.
    Returns lists of projected and found indices for matching points."""
    #How to match (in 2D, in particular)? Affine transformations? etc? Match by (relative) dot-to-dot distances?
    #See, for example, https://en.wikipedia.org/wiki/Point_set_registration, http://greatpanic.com/pntmatch.html, or Affine Consistency Check of Features in KLT Tracker http://cecas.clemson.edu/~stb/klt/
    #Implementations can be found in https://pypi.org/project/pycpd/, ...
    #Here we'll try to use RigidRegistration or AffineRegistration as an alternative to matching points directly by distance function
    #While this type of approach can perform fairly well with only a rough approximate pose, we find that a multi-stage matching and pose estimation approach is more consistent.

    local_dots_projected, local_dots_projected_jacobian = cv2.projectPoints(local_dots_to_project, approximate_pose_result.pose_cv.rotation_rodrigues, approximate_pose_result.pose_cv.translation, camera_matrix, distortion_coefficients)
    local_dots_projected = np.squeeze(local_dots_projected)
    registration_dictionary = {'X': dot_centers_cv, 'Y': local_dots_projected}
    
    if matching_style == 'affine_registration':
        registration = AffineRegistration(**registration_dictionary)
    else:
        registration = RigidRegistration(**registration_dictionary)
        
    registration.register()
    registration.transform_point_cloud()
    dot_centers_transformed = registration.TY

    match_cutoff_score = 0.9

    #Registration.P has shape M, N where M is num projected, N num found (in dot_centers_cv)
    projected_indices_assigned, found_indices_assigned = linear_sum_assignment(1 - registration.P)#Returns (row_indices, column_indices)
    registration_match_score_mask = registration.P[projected_indices_assigned, found_indices_assigned] > match_cutoff_score
    projected_indices, found_indices = projected_indices_assigned[registration_match_score_mask], found_indices_assigned[registration_match_score_mask]

    additional_data = { 'dot_centers_transformed' : dot_centers_transformed, 'local_dots_projected' : local_dots_projected}
    return projected_indices, found_indices, additional_data

def _get_die_pose_from_projected(bounding_box_cv : np.ndarray, dot_centers_cv : np.ndarray, local_dots_to_project : np.ndarray, approximate_pose_result : PoseResult, camera_matrix : np.ndarray, distortion_coefficients : np.ndarray) -> PoseResult:
    """Determines a die's pose that matches projection of local dots to found dot_centers_cv to solve Perspective-n-Point problem. Uses an approximate pose result as initial guess."""
    #Note that we need a candidate pose to start performing matching, since are knows are model points (in 3D) and image-space dot locations (in 2D).
    #Therefore, we must either project the model points onto the image to 2D (given a candidate pose), or reproject the dot locations back to 3D (again, given a candidate pose).
    #Since the former might yield some model-space dots incorrectly omitted (since they face away from camera in candidate pose), the 3D matching problem gives more flexibility.
    #However, without knowing the dot size (image, and model space) there's no way to determine each dot's z-distance from the camera, so we'd need a projective matching/registration algorithm, which would supercede doing something like PNP.

    if approximate_pose_result.comparison_indices is not None and approximate_pose_result.projected_indices is not None:
        projected_indices = approximate_pose_result.projected_indices
        found_indices = approximate_pose_result.comparison_indices
        additional_data = {}
        #Note that poorer matches here will be handled with Ransac/outlier exclusion below.
    else:
        projected_indices, found_indices, additional_data = _match_points_with_point_cloud_registration(dot_centers_cv, local_dots_to_project, approximate_pose_result, camera_matrix, distortion_coefficients)

    local_dots_for_pnp_masked = local_dots_to_project[projected_indices, :]
    dot_centers_cv_masked = dot_centers_cv[found_indices, :]

    extrinsic_rvec = approximate_pose_result.pose_cv.rotation_rodrigues.copy()
    extrinsic_tvec = approximate_pose_result.pose_cv.translation.copy()

    num_dots_min = len(found_indices)
    inlier_distance = inlier_cutoff_px
    perform_iterative = False
    #NB It seems SolvePNP may not work for < 4 points, even in Iterative/useExtrinsicGuess case it claims to handle, and correspondingly the RANSAC version needs at least one more point to be meaningful.
    if num_dots_min >= 5:
        pnp_flags = cv2.SOLVEPNP_ITERATIVE
        matched_pnp = cv2.solvePnPRansac(local_dots_for_pnp_masked, dot_centers_cv_masked, camera_matrix, distortion_coefficients, reprojectionError = inlier_distance, rvec = extrinsic_rvec, tvec = extrinsic_tvec, useExtrinsicGuess = True, flags = pnp_flags)
        pose_cv = Pose.create_from_cv_results(matched_pnp)
        if pose_cv:
            pose_result = PoseResult(pose_cv, additional_data, in_pyrender_coords = False)
        else:
            perform_iterative = True
    elif num_dots_min == 4:
        four_dot_pnp_flags = cv2.SOLVEPNP_AP3P
        four_dot_pnp = cv2.solvePnP(local_dots_for_pnp_masked, dot_centers_cv_masked, camera_matrix, distortion_coefficients, flags = four_dot_pnp_flags)
        four_dot_pnp_pose_cv = Pose.create_from_cv_results(four_dot_pnp)
        if four_dot_pnp_pose_cv:
            pose_result = PoseResult(four_dot_pnp_pose_cv, additional_data, in_pyrender_coords = False)
            pose_result.calculate_comparison(dot_centers_cv_masked, camera_matrix, distortion_coefficients)
            pose_result_distances = pose_result.comparison_projected_distances[pose_result.comparison_indices, pose_result.projected_indices]
            all_distances_inliers = np.all(pose_result_distances < inlier_distance)
            perform_iterative = not all_distances_inliers
        else:
            perform_iterative = True
    else:
        perform_iterative = True

    if perform_iterative:
        iterative_pose_result = _get_die_pose_iterative(bounding_box_cv, dot_centers_cv, approximate_pose_result, camera_matrix, distortion_coefficients, get_reprojection_error_sum_assignment_arctan)
        pose_result = iterative_pose_result
    
    return pose_result

def _get_die_pose_from_visible_dots(bounding_box_cv : np.ndarray, dot_centers_cv : np.ndarray, approximate_pose_result : PoseResult, camera_matrix : np.ndarray, distortion_coefficients : np.ndarray) -> PoseResult:
    """Gets the die's pose from projection of visible dots matched to found dot_centers_cv, given initial approximate pose result."""
    local_dots_facing_pyrender = get_local_dots_facing_camera_from_eye_space_pose(approximate_pose_result.pose_pyrender)
    local_dots_to_project = local_dots_facing_pyrender

    return _get_die_pose_from_projected(bounding_box_cv, dot_centers_cv, local_dots_to_project, approximate_pose_result, camera_matrix, distortion_coefficients)

def _get_local_dots_projected(trial_pose_cv : Pose, camera_matrix : np.ndarray, distortion_coefficients : np.ndarray):
    """Gets the projected visible dot positions of a die given its pose and camera information."""
    local_dots_facing_pyrender = get_local_dots_facing_camera_from_eye_space_pose(trial_pose_cv.get_converted_between_cv_and_pyrender_coords())
    local_dots_to_project = local_dots_facing_pyrender
    local_dots_projected, local_dots_projected_jacobian = cv2.projectPoints(local_dots_to_project, trial_pose_cv.rotation_rodrigues, trial_pose_cv.translation, camera_matrix, distortion_coefficients)
    local_dots_projected = np.squeeze(local_dots_projected, axis = 1)
    return local_dots_projected

def get_reprojection_error_sum_assignment_arctan(bounding_box_cv : np.ndarray, dot_centers_cv : np.ndarray, trial_pose_cv : Pose, camera_matrix : np.ndarray, distortion_coefficients : np.ndarray):
    """Gets a reprojection error based on the sum of arctan distances between projected and found points, matched using linear sum assignment."""
    local_dots_projected = _get_local_dots_projected(trial_pose_cv, camera_matrix, distortion_coefficients)
    comparison_projected_distances = cdist(dot_centers_cv, local_dots_projected)#MxN matrix with M comparison_points and N projected points
    distance_scale = distance_scale_px
    arctan_scores = distance_scale * np.arctan((comparison_projected_distances/distance_scale)**2)
        
    comparison_indices, projected_indices = linear_sum_assignment(arctan_scores)#Returns (row_indices, column_indices)
    matched_scores = arctan_scores[comparison_indices, projected_indices]
    matched_scores_rms = np.sqrt(np.mean(np.square(matched_scores)))

    projected_dots_distance_outside_dice_bb = np.maximum(np.maximum(0, local_dots_projected - bounding_box_cv[2:4]), np.maximum(0, bounding_box_cv[0:2] - local_dots_projected))
    max_distance_outside_dice_bb = np.max(projected_dots_distance_outside_dice_bb)

    final_score = matched_scores_rms + max_distance_outside_dice_bb
    return final_score

def _get_die_pose_iterative(bounding_box_cv : np.ndarray, dot_centers_cv : np.ndarray, approximate_pose_result : PoseResult, camera_matrix : np.ndarray, distortion_coefficients : np.ndarray, reprojection_error_function) -> PoseResult:
    """Gets a die pose iteratively given a reprojection error function, initial pose estimate, camera information, bounding box and found dot positions dot_centers_cv."""
    def get_reprojection_error(trial_pose_rodrigues_translation_cv):
        trial_pose_rodrigues_cv = trial_pose_rodrigues_translation_cv[0:3]
        trial_pose_translation_cv = trial_pose_rodrigues_translation_cv[3:]
        trial_pose_cv = Pose(True, trial_pose_rodrigues_cv, trial_pose_translation_cv)
        reproj_error = reprojection_error_function(bounding_box_cv, dot_centers_cv, trial_pose_cv, camera_matrix, distortion_coefficients)
        return reproj_error

    from scipy.optimize import minimize
    initial_guess = approximate_pose_result.pose_cv.as_numpy_array()
    minimization_results = minimize(get_reprojection_error, initial_guess, method = 'Nelder-Mead')#NB Nelder-Mead may not be the most efficient method, but a gradient-free method seems best to handle this particular cost function.
    minimized_pose = Pose.create_from_numpy_array(minimization_results.x)
    return PoseResult(minimized_pose, {}, in_pyrender_coords = False)

def _convert_tensorflow_points_to_opencv(tensor, transpose = False):
    """Converts a tensor of points to an OpenCV-coordinate numpy array."""
    tensor_points = tensor.numpy().astype(np.float32)
    if transpose:
        tensor_points = tensor_points.T
    for j in range(tensor_points.shape[1] // 2):
        i =j *2
        tensor_points[:, [i, i+1]] = tensor_points[:, [i + 1, i]]#Since TF returns y (row) coordinates first, we need to switch x and y for use with OpenCV
    return tensor_points

def _get_die_image_bounding_box_pose(bounding_box, camera_matrix : np.ndarray, distortion_coefficients : np.ndarray) -> PoseResult:
    """Get an approximate pose (particularly for translation) by solving PNP problem of the corners of an image-space bounding box."""
    box_size = tf.gather(bounding_box, [2, 3], axis = -1) - tf.gather(bounding_box, [0, 1], axis = -1)
    max_box_dimension = tf.math.reduce_max(tf.math.abs(box_size))
    dice_local_bb_min_max_abs = tf.math.abs(DiceConfig.get_local_bounding_box_min_max())
    dice_local_bb_extent = tf.math.reduce_max(dice_local_bb_min_max_abs)
    dice_local_bb_extent = tf.cast(dice_local_bb_extent, tf.float32)

    #NB Since distance between dots on face 2 is actually slightly greater than to neighbouring dots on other faces, we can't simply cluster based on dot-to-dot distance within each face.

    quad_scaling_factor = 1.2#Fitting a slightly larger quad than the dice extent will tend to give more accurate distance results when fitting a quad to the image bounding box.
    quad_with_dice_size = (np.array([[-1, -1, 0],[-1, 1, 0], [1, -1, 0], [1, 1, 0]]) * quad_scaling_factor * dice_local_bb_extent.numpy()).astype(np.float32)
    bounding_box_corners = tf.stack([tf.gather(bounding_box, [0, 1], axis = -1), tf.gather(bounding_box, [2, 1], axis = -1), tf.gather(bounding_box, [0, 3], axis = -1), tf.gather(bounding_box, [2, 3], axis = -1)], axis = -1)
    bounding_box_points = _convert_tensorflow_points_to_opencv(bounding_box_corners, transpose = True)
    quad_pnp_results = cv2.solvePnP(quad_with_dice_size, bounding_box_points, camera_matrix, distortion_coefficients)
    quad_pnp_pose_cv = Pose.create_from_cv_results(quad_pnp_results)
    quad_pnp_pose_results = PoseResult(quad_pnp_pose_cv, {}, in_pyrender_coords = False)
    return quad_pnp_pose_results

def _get_die_pose(bounding_box, die_class, die_y_angle, dot_centers_cv : np.ndarray, bounding_box_pose_result : PoseResult, approximate_up_vector_pyrender : np.ndarray, camera_matrix : np.ndarray, distortion_coefficients : np.ndarray) -> PoseResult:
    """Gets the die pose, given its bounding box, class, estimated rotation angle around vertical axis, found dot center points, approximate pose result and up vector, and camera information."""
    die_class_np = die_class.numpy()
    die_y_angle_deg_np = np.rad2deg(die_y_angle.numpy())
    #TODO Handle case of tilted camera (around its forward axis). Up vector could be approximated from plane estimate (if >= 3 dice are found). This would affect the order in which dice bounding box are processed (in order of which die are likely to be in front of others).
    x_rotation_from_up_vector_deg = np.rad2deg(np.arctan2(approximate_up_vector_pyrender[2], approximate_up_vector_pyrender[1]))
    potential_die_pose_approx_results = _get_approximate_die_pose(die_class_np, die_y_angle_deg_np, bounding_box_pose_result, x_axis_rotations_deg = np.array([0, 7, -7, 15, -15, 30, -30, 45, -45]) + x_rotation_from_up_vector_deg, y_rot_offsets_deg = [0, 10, -10, 20, -20, 45, -45, 60, -60])

    #NB OpenCV coords are x right, y down, z forward. pyrender's are x right, y up, z backwards. Both are right-handed but a 180-degree rotation around the x-axis different.
    bounding_box_cv = bounding_box.numpy()
    bounding_box_cv = bounding_box_cv[[1,0,3,2]]
    bounding_box_cv_size = bounding_box_cv[2:] - bounding_box_cv[0:2]

    for potential_results in potential_die_pose_approx_results:
        potential_results.calculate_comparison(dot_centers_cv, camera_matrix, distortion_coefficients)
    max_num_correlations = max([len(r.matched_scores) for r in potential_die_pose_approx_results])
    def get_distance_score_with_missing_residuals(pr : PoseResult, num_correlations):
        distance_scores = pr.matched_scores
        missing_residual_score = pr.assignment_score_function(np.max(bounding_box_cv_size) * 0.5)
        return np.sum(distance_scores) + (num_correlations - len(distance_scores)) * missing_residual_score
    approx_pose_results_per_projected_indices_set = defaultdict(list)
    for potential_results in potential_die_pose_approx_results:
        potential_results.distance_score_with_missing_residuals = get_distance_score_with_missing_residuals(potential_results, max_num_correlations)
        potential_results.projected_indices_set = frozenset(potential_results.projected_indices)
        approx_pose_results_per_projected_indices_set[potential_results.projected_indices_set].append(potential_results)
    best_approx_pose_result_per_projected_indices_set = { indices : min(corresponding_pose_results, key = lambda r : r.distance_score_with_missing_residuals) for (indices, corresponding_pose_results) in approx_pose_results_per_projected_indices_set.items() }
    best_approx_pose_result = min(best_approx_pose_result_per_projected_indices_set.values(), key = lambda r : r.distance_score_with_missing_residuals)
    distance_score_cutoff = best_approx_pose_result.distance_score_with_missing_residuals * approx_distance_score_cutoff_fraction
    visible_fit_pose_result_per_projected_indices_set = { indices : _get_die_pose_from_visible_dots(bounding_box_cv, dot_centers_cv, pose_result, camera_matrix, distortion_coefficients) for (indices, pose_result) in best_approx_pose_result_per_projected_indices_set.items() if pose_result.distance_score_with_missing_residuals < distance_score_cutoff}

    potential_visible_fit_pose_results = list(visible_fit_pose_result_per_projected_indices_set.values())
    if(len(potential_visible_fit_pose_results)) > 0:
        for pr in potential_visible_fit_pose_results:
            pr.calculate_comparison(dot_centers_cv, camera_matrix, distortion_coefficients)
        max_num_correlations_vis = max([len(r.matched_scores) for r in potential_visible_fit_pose_results])
        for pr in potential_visible_fit_pose_results:
            pr.assignment_inlier_cutoff = pr.assignment_score_function(inlier_cutoff_px)
            inlier_matched_scores = pr.matched_scores[pr.matched_scores < pr.assignment_inlier_cutoff]
            pr.inlier_matched_scores_rms = np.sqrt(np.mean(np.square(inlier_matched_scores)))
            pr.distance_score_with_missing_residuals = get_distance_score_with_missing_residuals(potential_results, max_num_correlations_vis)

        visible_fit_pose_results = min(potential_visible_fit_pose_results, key = lambda r : r.inlier_matched_scores_rms)
        visible_fit_pose_results.additional_data['approx_pose_result'] = best_approx_pose_result
    else:
        visible_fit_pose_results = copy.deepcopy(bounding_box_pose_result)
        visible_fit_pose_results.additional_data['approx_pose_result'] = best_approx_pose_result

    return visible_fit_pose_results

def _get_normal_up_vector(points_roughly_planar : np.ndarray) -> np.ndarray:
    """
    Estimates the normal upward-facing vector given a set of points roughly defining a horizontal plane
    points_roughly_planar is a 3xN matrix of points roughly defining a plane.
    The plane's upward-facing normal will be returned, or a vector most upwardly-pointing in the case of two points.
    """
    normal_up = None
    if points_roughly_planar.shape[1] < 2:
        normal_up = np.array([0, 1, 0])
    elif points_roughly_planar.shape[1] == 2:
        difference = points_roughly_planar[:, 1] - points_roughly_planar[:, 0]
        direction = difference / np.linalg.norm(difference)
        up = np.array([0, 1, 0])
        plane_other_direction = np.cross(direction, up)
        normal = np.cross(direction, plane_other_direction)
        normal_up = normal * np.sign(normal[1])
    else:
        mean = points_roughly_planar.mean(axis=1)
        point_differences = points_roughly_planar - mean[:,np.newaxis]
        covariance = np.cov(point_differences)
        svd = np.linalg.svd(covariance)
        normal = svd[0][:,-1]
        normal_up = normal * np.sign(normal[1])
    return normal_up

def _get_approximate_dice_up_vector(bounding_box_pose_results : List[PoseResult], in_pyrender_coords : bool) -> np.ndarray:
    """Gets an approximate up vector for the die, given approximate pose translations, assuming all die are lying flat on the same plane, thereby pointing upwards."""
    if len(bounding_box_pose_results) > 0:
        if in_pyrender_coords:
            dice_translations = np.hstack([pr.pose_pyrender.translation for pr in bounding_box_pose_results])
        else:
            dice_translations = np.hstack([pr.pose_cv.translation for pr in bounding_box_pose_results])
    else:
        dice_translations = np.zeros((0, 0))
    #3xN matrix of points at center of die
    approx_up = _get_normal_up_vector(dice_translations)
    return approx_up

def get_dice_pose_results(bounding_boxes, classes, scores, y_rotation_angles, camera_matrix : np.ndarray, distortion_coefficients : np.ndarray, score_threshold : float = 0.5):
    """Estimates pose results for all die, given estimates for bounding box, die (top face) classes, scores and threshold, rotation angles around vertical axes, and camera information."""
    scores_in_threshold = tf.math.greater(scores, score_threshold)
    classes_in_score = tf.boolean_mask(classes, scores_in_threshold)
    boxes_in_scores = tf.boolean_mask(bounding_boxes, scores_in_threshold)
    y_angles_in_scores = tf.boolean_mask(y_rotation_angles, scores_in_threshold)

    classes_are_dots = tf.equal(classes_in_score, 0)
    classes_are_dice = tf.logical_not(classes_are_dots)
    dice_bounding_boxes = tf.boolean_mask(boxes_in_scores, classes_are_dice)
    dice_y_angles = tf.boolean_mask(y_angles_in_scores, classes_are_dice)
    dice_classes = tf.boolean_mask(classes_in_score, classes_are_dice)
    dot_bounding_boxes = tf.boolean_mask(boxes_in_scores, classes_are_dots)

    dot_centers = _get_dot_centers(dot_bounding_boxes)
    dot_sizes = _get_dot_sizes(dot_bounding_boxes)

    #NB Largest box[2] is the box lower bound 
    dice_bb_lower_y = dice_bounding_boxes[:,2]
    dice_indices = tf.argsort(dice_bb_lower_y, axis = -1, direction='DESCENDING')

    def get_area(bb):
        return tf.math.maximum(bb[:, 3] - bb[:, 1], 0) * tf.math.maximum(bb[:, 2] - bb[:, 0], 0)

    dice_indices_np = dice_indices.numpy()
    bounding_box_pose_results = [_get_die_image_bounding_box_pose(dice_bounding_boxes[index, :], camera_matrix, distortion_coefficients) for index in dice_indices_np]
    approximate_dice_up_vector_pyrender = _get_approximate_dice_up_vector(bounding_box_pose_results, in_pyrender_coords=True)
    pose_results = []
    for index, bounding_box_pose_result in zip(dice_indices_np, bounding_box_pose_results):
        die_box = dice_bounding_boxes[index, :]
        die_y_angle = dice_y_angles[index]
        die_class = dice_classes[index]

        die_box_size = (-die_box[0:2] + die_box[2:4])
        dot_centers_fraction_of_die_box = (dot_centers - die_box[0:2]) / die_box_size
        dot_centers_rounded_rectangle_distance = tf.norm(tf.math.maximum(tf.math.abs(dot_centers_fraction_of_die_box - 0.5) - 0.5 + rounded_rectangle_radius,0.0), axis = -1) - rounded_rectangle_radius
        dots_are_in_rounded_rectangle = dot_centers_rounded_rectangle_distance < 0

        dot_bb_intersection_left = tf.math.maximum(dot_bounding_boxes[:, 1], die_box[1])
        dot_bb_intersection_right = tf.math.minimum(dot_bounding_boxes[:, 3], die_box[3])
        dot_bb_intersection_top = tf.math.maximum(dot_bounding_boxes[:, 0], die_box[0])
        dot_bb_intersection_bottom = tf.math.minimum(dot_bounding_boxes[:, 2], die_box[2])
        dot_bb_intersection = tf.stack([dot_bb_intersection_top, dot_bb_intersection_left, dot_bb_intersection_bottom, dot_bb_intersection_right], axis = 1)
        dot_bb_intersection_area = get_area(dot_bb_intersection)
        dot_bb_area = get_area(dot_bounding_boxes)
        dot_bb_intersection_over_area = dot_bb_intersection_area / dot_bb_area
        dots_have_sufficient_bb_intersection_over_area = tf.greater(dot_bb_intersection_over_area, 0.9)
        
        dots_are_in_box = tf.logical_and(dots_have_sufficient_bb_intersection_over_area, dots_are_in_rounded_rectangle)

        dot_centers_in_box = tf.boolean_mask(dot_centers, dots_are_in_box)
        dot_centers_cv = _convert_tensorflow_points_to_opencv(dot_centers_in_box)
        die_pose_result = _get_die_pose(die_box, die_class, die_y_angle, dot_centers_cv, bounding_box_pose_result, approximate_dice_up_vector_pyrender, camera_matrix, distortion_coefficients)
        die_pose_result.calculate_comparison(dot_centers_cv, camera_matrix, distortion_coefficients)
        die_pose_result.calculate_inliers(_convert_tensorflow_points_to_opencv(dot_sizes))
        pose_results.append(die_pose_result)

        indices_in_box = tf.where(dots_are_in_box)
        inlier_indices_in_box = tf.gather(indices_in_box, die_pose_result.comparison_inlier_indices)
        dot_centers = _delete_tf(dot_centers, inlier_indices_in_box)
        dot_sizes = _delete_tf(dot_sizes, inlier_indices_in_box)
        dot_bounding_boxes = _delete_tf(dot_bounding_boxes, inlier_indices_in_box)

    return pose_results
