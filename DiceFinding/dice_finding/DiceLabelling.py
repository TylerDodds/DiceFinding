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
"""Generates rendered images of dice and writes images and labels into TFRecord file."""

import numpy as np

import tensorflow as tf
import pyrender
from trimesh.transformations import quaternion_about_axis, quaternion_matrix, translation_matrix

import DiceScene
import DicePositioning
import DiceProjection
import DiceMask
import CameraCalibration

from object_detection.utils import dataset_util

class SceneCreator(object):
    """Creates basic pyrender scenes (without dice)."""
    def get_empty_scene(self):
        """Creates an empty scene with a camera and spot light."""
        scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2], bg_color=[0.1, 0.1, 0.1])

        camera = pyrender.PerspectiveCamera(yfov = np.pi / 3.0, aspectRatio= 1)
        camera_pose = translation_matrix([0, 3, 4]) @ quaternion_matrix(quaternion_about_axis(np.deg2rad(-45.0), [1, 0, 0]))
        scene.add(camera, pose=camera_pose)

        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0)
        scene.add(light, pose=camera_pose)

        return scene

class LabelledImageGenerator(object):
    """Generates labelled images as TFExamples."""
    def __init__(self, scene : pyrender.Scene, image_width : int, image_height : int, target_image_width : int, target_image_height : int, dice_trimesh_location : str, dice_spread_radius : float):
        self.scene = scene
        self.image_width = image_width
        self.image_height = image_height
        self.target_image_width = target_image_width
        self.target_image_height = target_image_height
        self.scene_loader = DiceScene.SceneSetupLoader(dice_trimesh_location)
        self.dice_spread_radius = dice_spread_radius
        self.class_string_dictionary = { 0 : "dot", 1 : "dice_1", 2 : "dice_2", 3 : "dice_3", 4 : "dice_4", 5 : "dice_5", 6 : "dice_6"}

    def get_dice_boxes_indices_and_dot_boxes(self, number_of_dice : int, creation_seed : int):
        """Gets an image and corresponding dice bounding boxes, top face indices, vertical rotations, dot bounding boxes, and corresponding camera intrinsic matrix."""
        scene_setup = DicePositioning.get_scene_setup(self.dice_spread_radius, number_of_dice, creation_seed)
        self.scene_loader.load_dice_nodes(scene_setup)
        color_final, depth_final, labels_mask = DiceMask.get_image_depth_and_mask(self.scene, self.scene_loader, self.image_width, self.image_height, keep_nodes_in_scene = True)
        dice_bounding_boxes = DiceMask.get_mask_image_bounding_boxes(labels_mask, scene_setup.get_number_of_dice())
        dice_top_face_indices = [DiceProjection.get_scene_space_up_face_index(dice_node, self.scene) for dice_node in self.scene_loader.dice_nodes]
        dice_y_rotation_angle_relative_to_camera = [DiceProjection.get_y_rotation_angle_relative_to_camera(dice_node, self.scene, dice_top_face_index) for dice_node, dice_top_face_index in zip(self.scene_loader.dice_nodes, dice_top_face_indices)]
        dot_boxes_list = [DiceProjection.get_image_space_dot_bounds(self.scene.main_camera_node, dice_node, self.scene, labels_mask, dice_index, unsunken = True) for dice_index, dice_node in enumerate(self.scene_loader.dice_nodes)]
        camera_matrix = CameraCalibration.get_symmetric_frustum_opencv_camera_matrix(self.scene.main_camera_node.camera, self.target_image_width, self.target_image_height)
        self.scene_loader.remove_nodes_from_scene(self.scene)
        self.scene_loader.clear_dice_nodes()
        return color_final, dice_bounding_boxes, dice_top_face_indices, dice_y_rotation_angle_relative_to_camera, dot_boxes_list, camera_matrix

    def create_tf_example(self, number_of_dice : int, creation_seed : int):
        """Creates a labelled TFExample with the given number of dice at the corresponding creation seed."""
        image, dice_bounding_boxes, dice_top_face_indices, dice_y_rotations, dot_boxes_list, camera_matrix = self.get_dice_boxes_indices_and_dot_boxes(number_of_dice, creation_seed)
        camera_intrinsic = camera_matrix.ravel()
        camera_distortion = np.zeros(5)

        dice_y_rotations = [0.000001 if dice_rot is 0 else dice_rot for dice_rot in dice_y_rotations]#"Zero" rotation is a special label for 'no rotation data defined'

        all_dot_boxes = [box for dot_boxes_per_dice in dot_boxes_list for box in dot_boxes_per_dice]
        all_dot_classes = [0] * len(all_dot_boxes) if len(all_dot_boxes) > 0 else []
        all_dot_y_rotations = [0] * len(all_dot_boxes) if len(all_dot_boxes) > 0 else []

        all_y_rotations = dice_y_rotations + all_dot_y_rotations
        
        all_boxes = dice_bounding_boxes + all_dot_boxes
        xmins = [box[0] / self.image_width for box in all_boxes]
        ymins = [box[1] / self.image_height for box in all_boxes]
        xmaxs = [(box[0] + box[2]) / self.image_width for box in all_boxes]
        ymaxs = [(box[1] + box[3]) / self.image_height for box in all_boxes]

        all_classes_unshifted = dice_top_face_indices + all_dot_classes
        all_class_strings = [str.encode(self.class_string_dictionary[class_id]) for class_id in all_classes_unshifted]
        all_class_ids = [id + 1 for id in all_classes_unshifted]#Since label maps for TF object detection start at 1, it seems

        image_resized = tf.image.resize_with_pad(image, self.target_image_width, self.target_image_height, method = tf.image.ResizeMethod.BILINEAR, antialias=False)
        image_resized_normalized = image_resized / 255.0
        image_resized_converted = tf.image.convert_image_dtype(image_resized_normalized, dtype=tf.uint16, saturate=False)
        encoded_image_data = tf.image.encode_png(image_resized_converted.numpy())
        encoded_image_data_np = encoded_image_data.numpy()
        image_format = b'png'
        image_name = str.encode("image_{}_{}".format(number_of_dice, creation_seed))
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(self.image_height),
            'image/width': dataset_util.int64_feature(self.image_width),
            'image/filename': dataset_util.bytes_feature(image_name),
            'image/source_id': dataset_util.bytes_feature(image_name),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data_np),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(all_class_strings),
            'image/object/class/label': dataset_util.int64_list_feature(all_class_ids),
            'image/object/rotation/y_angle': dataset_util.float_list_feature(all_y_rotations),
            'image/camera/intrinsic': dataset_util.float_list_feature(camera_intrinsic),
            'image/camera/distortion': dataset_util.float_list_feature(camera_distortion),
        }))
        return tf_example

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Dice Labelling.')
    parser.add_argument('--trimesh_location', type = str, default = None)
    args, unknown = parser.parse_known_args()

    scene_creator = SceneCreator()
    scene = scene_creator.get_empty_scene()
    width = 2048
    height = 2048
    target_width = 512
    target_height = 512
    dice_trimesh_location = args.trimesh_location
    dice_spread_radius = 2.0
    labeller = LabelledImageGenerator(scene, width, height, target_width, target_height, dice_trimesh_location, dice_spread_radius)

    writer = tf.io.TFRecordWriter("tf_config/dice_test.record")

    min_num_dice = 1
    max_num_dice = 6
    num_dice_offset_at_index = []
    for i in range(min_num_dice, max_num_dice + 1):
        for j in range(i, 0, -1):
            num_dice_offset_at_index.append(j)

    num_examples = 1000
    current_offset = 100
    import random
    for example_num in range(num_examples):
        print(example_num)
        num_dice = num_dice_offset_at_index[current_offset % len(num_dice_offset_at_index)]
        example = labeller.create_tf_example(num_dice, current_offset)
        writer.write(example.SerializeToString())
        current_offset += num_dice
    writer.close()
