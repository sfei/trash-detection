# Model Evaluation Script Overview

- export_inference_graph.py: Obtained from [tensorflow object detection library](https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py). Pulled around August 2018. Used to export the inference graph for running inference computations. Requires hardcoding object detection API path within script.
- test_checkpoint.py: Adapted from various online examples, essentially wraps export-inference_graph function.
- trash_classifier.py: Adapted from various online examples. Reads in test image, exported model, then calls visualization_utils.visualize_boxes_and_labels_on_image_array to draw bounding boxes.
- visualization_utils.py: Obtained from [tensorflow object detection library](https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py). Pulled around August 2018, used to draw bounding boxes on inferenced images. Requires hardcoding object detection API path within script.
- export_tf_serving_model.py: Outputs checkpoint files in the TF serving model format.
