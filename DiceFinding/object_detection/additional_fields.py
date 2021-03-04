# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

class InputDataFields(object):
  y_rotation_angle = 'y_rotation_angle'
  camera_intrinsic = 'camera_intrinsic'
  camera_distortion = 'camera_distortion'

class DetectionResultFields(object):
  y_rotation_angles = 'detection_y_rotation_angles'

class GroundtruthResultFields(object):
  y_rotation_angles = 'groundtruth_y_rotation_angles'

def do_add_rotation_evaluator(detection_model):
    try:
        do_add = detection_model.has_y_rotation
    except AttributeError:
        do_add = False
    return do_add