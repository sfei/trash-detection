#!/usr/bin/env python

"""export_tf_serving_model.py: Exports files for hosting on a TensorFlow serving server."""

__copyright__ = """
    Copyright (C) 2020 San Francisco Estuary Institute (SFEI)
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

# Based off of Daniel Stang's TrafficLightClassifier:
# https://gist.github.com/WuStangDan/f9cb0c4cda925dd3bd892fbf52f9e3e6#file-traffic_light_classifier-py

# Found in Medium post:
# https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-5-saving-and-deploying-a-model-8d51f56dbcf1

import sys
import os

# Add TensorFlow model API to python library path, including slim subdirectory
sys.path.append(r'[Path to Object Detection API]\research')
sys.path.append(r'[Path to Object Detection API]\research\slim')

# Import TensorFlow and other scientific libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# Import TensorFlow scripts
from visualization_utils import visualize_boxes_and_labels_on_image_array
from object_detection.utils import label_map_util
from object_detection.utils.config_util import create_pipeline_proto_from_configs
from object_detection.utils.config_util import get_configs_from_pipeline_file
from object_detection import exporter

# Input data paths
TEST_IMAGES_PATH = 'Path to cropped images'
MODEL_PATH = 'Path to model files'
NUM_CLASSES = 3

# TF Serving Model Export Directory
EXPORT_DIR = 'Path to output directory'
MODEL_VERSION = '1.0'

class TrashClassifier():
    """Class for loading and saving trash models.
    """
    def __init__(self, model_path):
        """Initialization, load model into memory.

        Args:
            model_path (string): System path to model files.
        """
        # laod graph
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            # Works up to here.
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.detection_graph)

        # Build the signature_def_map.
        classification_inputs = tf.compat.v1.saved_model.utils.build_tensor_info(
            self.image_tensor)
        classification_outputs_classes = tf.compat.v1.saved_model.utils.build_tensor_info(
            self.d_classes)
        classification_outputs_scores = tf.compat.v1.saved_model.utils.build_tensor_info(
            self.d_scores)

        classification_signature = (
            tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    tf.compat.v1.saved_model.signature_constants.CLASSIFY_INPUTS:
                        classification_inputs
                },
                outputs={
                    tf.compat.v1.saved_model.signature_constants
                    .CLASSIFY_OUTPUT_CLASSES:
                        classification_outputs_classes,
                    tf.compat.v1.saved_model.signature_constants
                    .CLASSIFY_OUTPUT_SCORES:
                        classification_outputs_scores
                },
                method_name=tf.compat.v1.saved_model.signature_constants
                .CLASSIFY_METHOD_NAME))

        tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
        tensor_info_y = tf.compat.v1.saved_model.utils.build_tensor_info(y)

        # prediction signature
        self.prediction_signature = (
            tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                inputs={'images':  self.image_tensor},
                outputs={'scores': self.d_scores},
                method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

    def export_serving_model(self,export_dir,model_version):
        """Saves TensorFlow serving model to file

        Args:
            export_dir (string): Path to location to save serving model.
            model_version (int): Model version number.
        """
        export_path = os.path.join(export_dir,'model_{}'.format(model_version))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            self.sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images':
                    self.prediction_signature,
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    classification_signature,
            },
            main_op=tf.tables_initializer())
        builder.save()

def main():
    """Loads model inot TrashClassifier instance then saves TF serving model to file.
    """

    # Configuration for model to be exported
    config_pathname = os.path.join(MODEL_PATH,'model','pipeline.config')

    # Input checkpoint for the model to be exported
    # Path to the directory which consists of the saved model on disk (see above)
    trained_model_dir = os.path.join(MODEL_PATH,'model')

    # Create proto from model confguration
    configs = get_configs_from_pipeline_file(config_pathname)
    pipeline_proto = create_pipeline_proto_from_configs(configs=configs)

    # Read .ckpt and .meta files from model directory
    checkpoint = tf.train.get_checkpoint_state(trained_model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # Model Version
    model_version_id = 1

    # Output Directory
    output_directory = "PATH TO OUTPUT DIRECTORY"

    # Export model for serving
    exporter.export_inference_graph(
        input_type='image_tensor',
        pipeline_config=pipeline_proto,
        trained_checkpoint_prefix=input_checkpoint,
        output_directory=output_directory
    )

if __name__ == '__main__':
    main()