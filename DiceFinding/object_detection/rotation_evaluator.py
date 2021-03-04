from object_detection.box_coders import mean_stddev_box_coder
from object_detection.core import box_list
from object_detection.core import region_similarity_calculator
from object_detection.core import standard_fields
from object_detection.core import target_assigner
from object_detection.matchers import argmax_matcher
from object_detection.metrics import calibration_metrics
from object_detection.utils import object_detection_evaluation

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
import numpy as np
import six

from object_detection import additional_fields

debug_get_matching_boxes = False

class RotationEvaluator():
    def __init__(self, iou_thresholds = [0.5, 0.75, 0.9], score_threshold = 0.5, display_histogram = False):
        self._iou_thresholds = iou_thresholds
        self._assigners = [self.create_target_assigner(iou_threshold) for iou_threshold in iou_thresholds]
        self._iou_thresholds_and_assigners = [(iou, assigner) for iou, assigner in zip(self._iou_thresholds, self._assigners)]
        self._score_threshold = 0.5
        self._display_histogram = display_histogram
        self._expected_keys = set([
            standard_fields.InputDataFields.key,
            standard_fields.InputDataFields.groundtruth_boxes,
            standard_fields.InputDataFields.groundtruth_classes,
            standard_fields.InputDataFields.num_groundtruth_boxes,
            additional_fields.GroundtruthResultFields.y_rotation_angles,
            standard_fields.DetectionResultFields.detection_boxes,
            standard_fields.DetectionResultFields.detection_scores,
            standard_fields.DetectionResultFields.detection_classes,
            additional_fields.DetectionResultFields.y_rotation_angles
            ])
        self._num_angle_bins = 180
        self._histogram_range = tf.constant([0.0, 180.0])
        self.total_num_angle_matches = {}
        self.total_angle_diff_sum_squared = {}
        self.angle_histograms = {}
        for iou_threshold in self._iou_thresholds:
            self.total_num_angle_matches[iou_threshold] = 0
            self.total_angle_diff_sum_squared[iou_threshold] = 0.0
            self.angle_histograms[iou_threshold] = tf.zeros(self._num_angle_bins, dtype = tf.int32)

    def create_target_assigner(self, iou_threshold):
        similarity_calc = region_similarity_calculator.IouSimilarity()
        matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=iou_threshold, unmatched_threshold=iou_threshold)
        box_coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
        assigner = target_assigner.TargetAssigner(similarity_calc, matcher, box_coder)
        return assigner
            
    def add_single_image_info(self, image_id, eval_dict):
        groundtruth_boxes = eval_dict[standard_fields.InputDataFields.groundtruth_boxes]
        groundtruth_classes = eval_dict[standard_fields.InputDataFields.groundtruth_classes]
        detection_boxes = eval_dict[standard_fields.DetectionResultFields.detection_boxes]
        detection_scores = eval_dict[standard_fields.DetectionResultFields.detection_scores]
        detection_classes = eval_dict[standard_fields.DetectionResultFields.detection_classes]

        groundtruth_has_rotation = groundtruth_classes > 1
        groundtruth_boxes_with_rotation = groundtruth_boxes[groundtruth_has_rotation]

        #ensure classes are both not 'dot' class, so they have a meaningful rotation value
        detection_within_score = detection_scores > self._score_threshold
        detection_class_has_rotation = detection_classes > 1
        detection_has_rotation_and_score = tf.logical_and(detection_within_score, detection_class_has_rotation)
        detection_boxes_within_score = detection_boxes[detection_has_rotation_and_score]
        detection_classes_within_score = detection_classes[detection_has_rotation_and_score]

        gt_boxlist = box_list.BoxList(tf.convert_to_tensor(groundtruth_boxes_with_rotation))
        det_boxlist = box_list.BoxList(tf.convert_to_tensor(detection_boxes_within_score))

        detection_y_rotation_angles = eval_dict[additional_fields.DetectionResultFields.y_rotation_angles]
        groundtruth_y_rotation_angles = eval_dict[additional_fields.GroundtruthResultFields.y_rotation_angles]
        detection_y_rotation_angles_within_score = detection_y_rotation_angles[detection_has_rotation_and_score]

        for iou_threshold, assigner in self._iou_thresholds_and_assigners:
            cls_targets, cls_weights, reg_targets, reg_weights, match = assigner.assign(det_boxlist, gt_boxlist)

            fg_detections = match >= 0
            fg_detection_boxes = detection_boxes_within_score[fg_detections, :]
            fg_matches = match[fg_detections]

            fg_matches_argsort = tf.argsort(fg_matches)
            fg_matches_sorted = tf.gather(fg_matches, fg_matches_argsort)

            gt_match_indices, fg_match_sorted_indices_with_repeats, fg_match_sorted_indices_counts = tf.unique_with_counts(fg_matches_sorted)
            fg_match_sorted_indices_no_repeats = tf.cumsum(tf.pad(fg_match_sorted_indices_counts,[[1,0]]))[:-1]

            fg_match_indices_no_repeats = tf.gather(fg_matches_argsort, fg_match_sorted_indices_no_repeats)

            def get_matches_and_angle_difference(fg_match_idx_tensor, gt_match_idx_tensor):
                if debug_get_matching_boxes:
                    gt_matching_detection_boxes = tf.gather(groundtruth_boxes_with_rotation, gt_match_idx_tensor, axis = 0)
                    fg_matching_detection_boxes = tf.gather(fg_detection_boxes, fg_match_idx_tensor, axis = 0)
                    pass

                fg_matching_detection_y_rot_angles = tf.gather(detection_y_rotation_angles_within_score, fg_match_idx_tensor, axis = 0)

                groundtruth_y_rotation_angles_matches = tf.gather(groundtruth_y_rotation_angles, gt_match_idx_tensor, axis = 0)
                groundtruth_has_y_rot = tf.math.logical_not(tf.math.equal(groundtruth_y_rotation_angles_matches, 0))
                groundtruth_existant_y_rot_angle = groundtruth_y_rotation_angles_matches[groundtruth_has_y_rot]

                detection_existant_y_rot_angle = fg_matching_detection_y_rot_angles[groundtruth_has_y_rot]

                angle_diff = detection_existant_y_rot_angle - groundtruth_existant_y_rot_angle
                angle_diff_unwrapped = tf.math.atan2(tf.math.sin(angle_diff), tf.math.cos(angle_diff))
                angle_diff_abs = tf.math.abs(angle_diff_unwrapped)

                n_angle_matches = len(angle_diff)

                return n_angle_matches, angle_diff_abs

            num_angle_matches, abs_angle_differences = get_matches_and_angle_difference(fg_match_indices_no_repeats, gt_match_indices)
            angle_diff_sum_square = tf.reduce_sum(tf.math.square(abs_angle_differences * 180 / np.pi))
            match_angle_diff_histogram = tf.histogram_fixed_width(abs_angle_differences * 180 / np.pi, self._histogram_range, nbins=self._num_angle_bins, dtype=tf.dtypes.int32)

            self.total_num_angle_matches[iou_threshold] += num_angle_matches
            self.total_angle_diff_sum_squared[iou_threshold] += angle_diff_sum_square
            self.angle_histograms[iou_threshold] += match_angle_diff_histogram

    def add_eval_dict(self, eval_dict):
        # remove unexpected fields
        eval_dict_filtered = dict()
        for key, value in eval_dict.items():
            if key in self._expected_keys:
                eval_dict_filtered[key] = value

        eval_dict_keys = list(eval_dict_filtered.keys())
        
        def update_op(image_id, *eval_dict_batched_as_list):
            if np.isscalar(image_id):
                single_example_dict = dict(zip(eval_dict_keys, eval_dict_batched_as_list))
                self.add_single_image_info(image_id, single_example_dict)
            else:
                for unzipped_tuple in zip(*eval_dict_batched_as_list):
                    single_example_dict = dict(zip(eval_dict_keys, unzipped_tuple))
                    image_id = single_example_dict[standard_fields.InputDataFields.key]
                    self.add_single_image_info(image_id, single_example_dict)

        args = [eval_dict_filtered[standard_fields.InputDataFields.key]]
        args.extend(six.itervalues(eval_dict_filtered))
        return tf.py_func(update_op, args, [])

    def evaluate(self):
        y_rot_category_metrics = {}
        for iou_threshold in self._iou_thresholds:

            total_angle_diff_sum_squared = self.total_angle_diff_sum_squared[iou_threshold]
            total_num_angle_matches = self.total_num_angle_matches[iou_threshold]
            total_angle_diff_histogram = self.angle_histograms[iou_threshold]
            
            angle_rms = tf.math.sqrt(total_angle_diff_sum_squared / total_num_angle_matches)
            angle_diff_histogram_normalized = total_angle_diff_histogram / total_num_angle_matches
            y_rot_category_metrics['YRotation_RMS_{}'.format(iou_threshold)] = angle_rms

            if self._display_histogram:
                import matplotlib; matplotlib.use('TkAgg')
                import matplotlib.pyplot as plt
                plt.figure()
                plt.bar(np.arange(self._num_angle_bins), angle_diff_histogram_normalized)
                plt.show()
        return y_rot_category_metrics
