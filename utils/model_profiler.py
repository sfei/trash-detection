#!/usr/bin/env python

"""model_profiler.py: Profile model loading and inferencing time."""

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
import sys
import os
import datetime
import numpy as np
import pandas as pd
from PIL import Image

# Add Object detection API to Python system path list.
sys.path.append(r'[OBJECT DETECTION API PATH]\research')
sys.path.append(r'[OBJECT DETECTION API PATH]\research\slim')

# Load tensorflow and relevant utilities.
import tensorflow as tf
from visualization_utils import visualize_boxes_and_labels_on_image_array
from object_detection.utils import label_map_util

MODEL_PATH = 'PATH TO MODEL FILES'
IMG_PATH = 'PATH TO IMAGE'
IMG_OUT_PATH = 'PATH TO STORE IMAGE WITH BOUDING BOX INFO'
NUM_CLASSES = 3

class TrashClassifier():
    """Class for loading and saving trash models.
    """
    def __init__(self, model_path):
        """Initialization, load model into memory.

        Args:
            model_path (string): System path to model files.
        """
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

    def get_classification(self, img):
        """Runs infrencing over an image.

        Args:
            img (numpy array): Numpy array containing pixel information

        Returns:
            tuple: Tuple containing bouding box information, classification scores, classifications, and 
        """
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        return boxes, scores, classes, num

def load_image_into_numpy_array(image):
    """Loads an image file into numpy array containing pixel values.

    Args:
        image (Image): Image loaded via pillow

    Returns:
        np.array: Array containing image pixel information.
    """
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def gen_classified_image(trash_classifier, category_index, img_in_path, img_out_path, min_score_thresh):
    """Runs trash classification inferencing over an image, draws bouding boxes over detections set to a 
       pre-defined threshold, then returns some useful information.

    Args:
        trash_classifier (TrashClassifier): Instance of the TrashClassifier class.
        category_index (dict): Dictionary containing crosswalk between category label and number.
        img_in_path (string): System path to image to run inferencing over.
        img_out_path (string): System path for saving image containing bounding boxes.
        min_score_thresh (float): Score threshold used for determining when to draw bounding boxes over positives.

    Returns:
        [type]: [description]
    """
    # Open image then convert to numpy array
    img = Image.open(img_in_path)
    img_np = load_image_into_numpy_array(img)

    # Get bounding boxes
    (boxes, scores, classes, num) = trash_classifier.get_classification(img_np)

    # Draw bounding boxes on image
    img_out_np = visualize_boxes_and_labels_on_image_array(
        image = img_np,
        boxes = np.squeeze(boxes),
        classes = np.squeeze(classes),
        scores = np.squeeze(scores),
        category_index = category_index,
        use_normalized_coordinates = True,
        skip_labels = False,
        agnostic_mode = False,
        min_score_thresh=min_score_thresh
    )

    # Save image with bounding boxes
    img_out = Image.fromarray(img_out_np)
    img_out.save(img_out_path)

    # Count scores over minimum score threshold
    score_vals = scores[0]
    detection_count = len(score_vals[score_vals > min_score_thresh])

    return (detection_count, num[0])

def main():
    ts1 = datetime.datetime.now()
    # Minimum threshold
    min_score_thresh = 0.5

    # Obtain label map
    label_map = label_map_util.load_labelmap(os.path.join(MODEL_PATH,'pascal_label_map.pbtxt'))
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    model_path = os.path.join(MODEL_PATH,'fine_tuned_model','frozen_inference_graph.pb')
    trash_classifier = TrashClassifier(model_path)
    (detection_count, num) = gen_classified_image(
            trash_classifier, 
            category_index, 
            IMG_PATH, 
            IMG_OUT_PATH,
            min_score_thresh
        )
    ts2 = datetime.datetime.now()

    print(str(ts2-ts1))

if __name__ == "__main__":
    main()