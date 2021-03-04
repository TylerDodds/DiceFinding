# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# Modifications Copyright (C) 2021 Dice Finding Authors.
# Altered to handle visualization and annotations from inference, and parse other potential command-line flags.
"""Perform inference and visualization on evaluation TFRecord."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

show_images = True
show_approx = False
get_poses = True
write_annotations = True

label_offset = 0
min_score = 0.5
config_file = "tf_config/centernet_pipeline.config"

import matplotlib; matplotlib.use('TkAgg')  # pylint: disable=multiple-statements
print("Matplotlib backend: ", matplotlib.get_backend())
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from absl import logging
import functools
import os
import sys
import pprint
import tensorflow.compat.v2 as tf
import numpy as np

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection import inputs



def run(callbacks=None):
    configs = config_util.get_configs_from_pipeline_file(config_file)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    #tf.config.experimental_run_functions_eagerly(True) 

    checkpoint_dir = app.FLAGS.checkpoint_dir
    checkpoint_path = tf.train.latest_checkpoint(app.FLAGS.model_dir if not len(checkpoint_dir) > 0 else checkpoint_dir, latest_filename=None)
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(checkpoint_path).expect_partial()

    def get_model_detection_function(model):
        """Get a tf.function for detection."""

        @tf.function
        def detect_fn(image):
            """Detect objects in image."""

            image, shapes = model.preprocess(image)
            prediction_dict = model.predict(image, shapes)
            detections = model.postprocess(prediction_dict, shapes)
            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detect_fn

    detect_fn = get_model_detection_function(detection_model)

    eval_config = configs['eval_config']
    eval_input_configs = configs['eval_input_configs']

    eval_inputs = []
    for eval_input_config in eval_input_configs:
        next_eval_input = inputs.eval_input(
            eval_config=eval_config,
            eval_input_config=eval_input_config,
            model_config=model_config,
            model=detection_model)
        eval_inputs.append((eval_input_config.name, next_eval_input))

    index = 0
    annotation_min_score = 0.5
    annotations_center = []
    annotations_box = []
    empty_json = "{}"

    for eval_name, eval_input in eval_inputs:
        for images_dict, labels in eval_input:
            orig_image = images_dict['original_image']
            images = tf.cast(orig_image, tf.float32)

            detections, prediction_outputs, shapes = detect_fn(images)
            images_shape = shapes.numpy()[0:2]
            image_shape_multiplier = tf.cast(tf.tile(shapes[0:2], [2]), tf.float32)

            def shape_boxes(boxes):
                boxes_px = boxes * image_shape_multiplier
                return boxes_px

            def get_boxes(numpy = True):
                boxes = detections['detection_boxes'][i]
                shaped = shape_boxes(boxes)
                if numpy:
                    shaped = shaped.numpy()
                return shaped

            for i in range(images.shape[0]):
                index += 1
                boxes = get_boxes(numpy = False)
                classes = detections['detection_classes'][i] + label_offset
                scores = detections['detection_scores'][i]
                y_angles = detections['detection_y_rotation_angles'][i]
                source_id = images_dict['source_id'][i]
                cv_camera_matrix = images_dict['camera_intrinsic'][i].numpy().reshape(3,3)
                cv_distortion_coefficients = images_dict['camera_distortion'][i].numpy()
                if get_poses:
                    from DicePoseFinding import get_dice_pose_results
                    dice_pose_results = get_dice_pose_results(boxes, classes, scores, y_angles, cv_camera_matrix, cv_distortion_coefficients)
                else:
                    dice_pose_results = []
                boxes = boxes.numpy()
                classes = classes.numpy()
                scores = scores.numpy()
                source_id = source_id.numpy()
                if show_images:
                    import matplotlib.patches as patches
                    image = images[i]
                    image_np = image.numpy() / 255.0
                    rotation_angles = detections['detection_y_rotation_angles'][i].numpy()
                    matplotlib.use('TkAgg')
                    plt.figure(figsize=(12,16))
                    plt.imshow(image_np)
                    ax = plt.gca()
                    for dice_pose_result in dice_pose_results:
                        approx_pose_result = dice_pose_result.additional_data['approx_pose_result']
                        from TransformUtil import transform_points_3d
                        import DiceConfig, DiceProjection
                        import cv2
                        local_dots_visible_in_eye_space = DiceProjection.get_local_dots_facing_camera_from_eye_space_pose(dice_pose_result.pose_pyrender)
                        #NB We should use cv-space pose when using cv2.projectPoints
                        if local_dots_visible_in_eye_space.size > 0:
                            pose_points, pose_points_jacobian = cv2.projectPoints(local_dots_visible_in_eye_space, dice_pose_result.pose_cv.rotation_rodrigues, dice_pose_result.pose_cv.translation, cv_camera_matrix, cv_distortion_coefficients)
                            pose_points = np.squeeze(pose_points)
                            ax.scatter(pose_points[:, 0], pose_points[:, 1], s = 4)
                            comp_pts = dice_pose_result.comparison_points_cv[dice_pose_result.comparison_indices]
                            proj_pts = dice_pose_result.projected_points[dice_pose_result.projected_indices]
                            ax.plot(np.vstack([comp_pts[:, 0],proj_pts[:, 0]]),np.vstack([comp_pts[:, 1],proj_pts[:, 1]]), 'g-')
                        local_dots_visible_in_eye_space_approx = DiceProjection.get_local_dots_facing_camera_from_eye_space_pose(approx_pose_result.pose_pyrender)
                        if show_approx and local_dots_visible_in_eye_space_approx.size > 0:
                            pose_points, pose_points_jacobian = cv2.projectPoints(local_dots_visible_in_eye_space_approx, approx_pose_result.pose_cv.rotation_rodrigues, approx_pose_result.pose_cv.translation, cv_camera_matrix, cv_distortion_coefficients)
                            pose_points = np.squeeze(pose_points)
                            ax.scatter(pose_points[:, 0], pose_points[:, 1], s = 4)
                            comp_pts = approx_pose_result.comparison_points_cv[approx_pose_result.comparison_indices]
                            proj_pts = approx_pose_result.projected_points[approx_pose_result.projected_indices]
                            ax.plot(np.vstack([comp_pts[:, 0],proj_pts[:, 0]]),np.vstack([comp_pts[:, 1],proj_pts[:, 1]]), 'y-')
                        try:
                            local_dots_projected = dice_pose_result.additional_data['local_dots_projected']
                            dot_centers_transformed = dice_pose_result.additional_data['dot_centers_transformed']
                            #ax.scatter(local_dots_projected[:, 0], local_dots_projected[:, 1], marker='+')
                            #ax.scatter(dot_centers_transformed[:, 0], dot_centers_transformed[:, 1], marker='s')
                        except KeyError:
                            pass
                    for j in range(boxes.shape[0]):
                        box = tuple(boxes[j].tolist())
                        score = scores[j]
                        class_id = classes[j]
                        rot_angle = (rotation_angles[j] % (2 * np.pi)) * 180. / np.pi
                        if score > min_score:
                            rect = patches.Rectangle((box[1],box[0]),box[3] - box[1],box[2] - box[0],linewidth=1,edgecolor='r',facecolor='none')
                            ax.add_patch(rect)
                            if class_id > 0:
                                plt.text(box[1], box[0], '<{:.2f}>:{}:{:.2f}\n@ {:.0f} {:.0f}'.format(score, class_id, rot_angle, box[1], box[0]), {'color': 'r'})
                            else:
                                #plt.text(box[1], box[0], ' {:.2f}:D'.format(score), {'color': 'r'})
                                pass
                    label_rotation_angles = labels['y_rotation_angle'][i].numpy()
                    label_boxes = labels['groundtruth_boxes'][i].numpy()
                    label_classes = labels['groundtruth_classes'][i].numpy()
                    label_classes_id = np.argmax(label_classes, axis = -1)
                    label_box_px = shape_boxes(label_boxes).numpy()
                    num_gt_boxes = labels['num_groundtruth_boxes'].numpy()[0]
                    for j in range(num_gt_boxes):
                        box = tuple(label_box_px[j].tolist())
                        rect = patches.Rectangle((box[1],box[0]),box[3] - box[1],box[2] - box[0],linewidth=1,linestyle=':',edgecolor='g',facecolor='none')
                        ax.add_patch(rect)
                        class_id = label_classes_id[j]
                        rot_angle = (label_rotation_angles[j] % (2 * np.pi)) * 180. / np.pi
                        if class_id > 0:
                            plt.text(box[3], box[0], '{}:{:.2f}'.format(class_id, rot_angle), {'color': 'g'})
                        else:
                            #plt.text(box[3], box[0], 'D', {'color': 'g'})
                            pass
                    plt.show()
                for j in range(boxes.shape[0]):
                    box = boxes[j].tolist()
                    score = scores[j]
                    class_id = classes[j]
                    if score > annotation_min_score:
                        x_min = box[1]
                        y_min = box[0]
                        width = box[3] - box[1]
                        height = box[2] - box[0]
                        x_mid = (box[1] + box[3]) / 2
                        y_mid = (box[0] + box[2]) / 2
                        region_attributes_string = "{\"type\":" + str(class_id) + "}"
                        region_shape_string_box = "{\"name\":\"rect\",\"x\":" + str(x_min) + ",\"y\":" + str(y_min) + ",\"width\":" + str(width) + ",\"height\":" + str(height) + "}"
                        annotations_box.append([source_id, empty_json, empty_json, empty_json, j, region_shape_string_box, region_attributes_string])
                        region_shape_string_center = "{\"name\":\"point\",\"cx\":" + str(x_mid) + ",\"cy\":" + str(y_mid) + "}"
                        annotations_center.append([source_id, empty_json, empty_json, empty_json, j, region_shape_string_center, region_attributes_string])
                print(index)

    if write_annotations:
        import csv
        with open('output_annotation_box.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in annotations_box:
                writer.writerow(row)
        with open('output_annotation_point.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in annotations_center:
                writer.writerow(row)

    print("DONE")

def main(argv):
    del argv  # Unused.

    run()

if __name__ == '__main__':
    assert tf.version.VERSION.startswith('2.')
    tf.config.set_soft_device_placement(True)
    app.flags.DEFINE_string('pipeline_config_path', '', 'Pipeline config path')
    app.flags.DEFINE_string('model_dir', '', 'Model dir')
    app.flags.DEFINE_string('trimesh_location', '', 'Trimesh location')
    app.flags.DEFINE_string('checkpoint_dir', '', 'Checkpoint directory')
    app.run(main)