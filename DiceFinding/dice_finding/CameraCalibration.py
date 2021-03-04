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
"""Functions to determine opencv camera intrinsic matrix from pyrender Camera."""
import numpy as np

import pyrender
from trimesh.transformations import quaternion_about_axis, quaternion_matrix, translation_matrix

import cv2

def get_symmetric_frustum_opencv_camera_matrix(camera : pyrender.Camera, image_width : int, image_height : int):
    """Gets an OpoenCV intrinsic matrix given a pyrender Camera and image size, assuming a symmetric frustum."""
    camera_projection_matrix = camera.get_projection_matrix(image_width, image_height)
    fx = -image_width * camera_projection_matrix[0, 0] / 2
    fy = -image_height * camera_projection_matrix[1, 1] / 2
    cx = image_width  / 2
    cy = image_height  / 2
    #Since z-axis of OpenCV coordinates is reverse of that of pyrender's, we flip signs of fx, fy accordingly
    fx *= -1
    fy *= -1
    camera_matrix = np.eye(3)
    camera_matrix[0,0] = fx
    camera_matrix[1,1] = fy
    camera_matrix[0,2] = cx
    camera_matrix[1,2] = cy
    return camera_matrix.astype(np.float32)

def calibrate_camera(camera : pyrender.Camera, image_width : int, image_height : int):
    """Sanity check to determine intrinstic matrix and distortion coefficients from a pyrender Camera by performing calibration on images of a known set of points."""
    approximate_from_image = False

    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0.0, 0.0, 0.0])

    all_object_points = []
    all_image_points = []

    camera_pose = translation_matrix([0, 0, 0])
    scene.add(camera, pose=camera_pose)

    camera_projection_matrix = camera.get_projection_matrix(image_width, image_height)

    quad_points = np.array([[-1,1,0], [-0.75, 1, 0], [-0.5, -1, 0], [-0.25, -1, 0], [0, 0, 0], [0.25, 0.5, 0], [0.5, 0.75, 0]]).astype(np.float32)
    quad_colors = np.ones_like(quad_points)
    quad_mesh = pyrender.Mesh.from_points(quad_points, colors=quad_colors)
    
    all_offsets = [[0, 0, -2.0], [0, 0, -3.0], [0, 0, -4.0], [0.5, 0.5, -4.0], [-0.5, 0.5, -3.0], [0.23, -0.33, -4.0],
                    [0, -0.32, -3.0], [0, 0.2, -3.0], [0, 0.2, -4.0], [0.5, 0.6, -4.0], [-0.5, 0.6, -3.0], [0.23, -0.43, -4.0],
                    [0, 0.32, -3.0], [0, -0.2, -3.0], [0.1, -0.2, -4.0], [-0.5, -0.4, -4.0], [0.5, -0.6, -3.0], [-0.23, -0.33, -4.0]]

    all_rvecs = [[0, 0, -0.1], [0, 0, -0.2], [0, 0.1, -0.2], [0.1, 0.1, -0.2], [-0.1, 0.1, 0.1], [0.2, -0.1, -0.1]] * 3

    quad_poses = []

    for offsets, rvecs in zip(all_offsets, all_rvecs):

        quad_pose = translation_matrix(offsets)
        quad_rot_matrix, _ = cv2.Rodrigues(np.array(rvecs))
        quad_pose[0:3, 0:3] = quad_rot_matrix
        quad_poses.append(quad_pose)

    all_transformed_object_points = []

    for quad_pose in quad_poses:

        if approximate_from_image:
            quad_node = scene.add(quad_mesh, pose = quad_pose)
            renderer = pyrender.OffscreenRenderer(image_width, image_height)
            color_bg, depth_bg = renderer.render(scene)
            scene.remove_node(quad_node)

            gray = cv2.cvtColor(color_bg, cv2.COLOR_BGR2GRAY)
            threshold, binarized_image = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binarized_image,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contour_means = np.array([np.mean(np.squeeze(c, axis = 1), axis = 0) for c in contours]).astype(np.float32)
            contour_means = contour_means[np.argsort(contour_means[:, 0])]#Order contours in x to match order of quad_points
            all_image_points.append(contour_means)

        else:
            quad_points_homogeneous = np.pad(quad_points, ((0,0), (0,1)), constant_values = 1).T
            quad_points_homogeneous_transformed = quad_pose @ quad_points_homogeneous
            projected_points_h = camera_projection_matrix @ quad_points_homogeneous_transformed
            projected_points_h_normalized = projected_points_h / projected_points_h[-1, :]
            projected_points_ndc = projected_points_h_normalized[0:2, :]
            projected_points_screen_xy = 0.5 * (projected_points_ndc + 1) * np.array([image_width, image_height])[:, np.newaxis]
            projected_points_screen = projected_points_screen_xy.T.astype(np.float32)
            all_image_points.append(projected_points_screen)
            all_transformed_object_points.append(quad_points_homogeneous_transformed[0:3, :].T)

        all_object_points.append(quad_points)


    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(all_object_points, all_image_points, (image_width, image_height), None, None, flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)

    #To handle z-axis handednes
    camera_matrix[0,0] *= -1
    camera_matrix[1,1] *= -1

    projected_transformed_h = [camera_matrix @ top.T for top in all_transformed_object_points]
    projected_transformed = [(pt[0:2, :] / pt[2, :]).T for pt in projected_transformed_h]
    cv_projected_transformed = [np.squeeze(cv2.projectPoints(top.astype(np.float32), np.zeros(3), np.zeros(3), camera_matrix, dist_coefs)[0]) for top in all_transformed_object_points]
    for img_pts, proj_pts, cv_proj_pts in zip(all_image_points, projected_transformed, cv_projected_transformed):
        eps = 1e-3
        assert np.max(np.abs(img_pts - proj_pts)) < eps, "Image and projected points are not close"
        assert np.max(np.abs(img_pts - cv_proj_pts)) < eps, "Image and CV projected points are not close"

    return camera_matrix, dist_coefs