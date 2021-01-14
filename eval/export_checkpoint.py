"""export_checkpoint.py: Exports files for hosting on a TensorFlow serving server."""

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

# Append python path
import sys
import os
import datetime
import shutil
import glob
import configparser
import export_inference_graph
from pathlib import Path

CONFIG_FILE_NAME = Path("../config.ini")

def load_config():
    """Load configuration file containing important path information.

    Raises:
        FileNotFoundError: If the config file isn't found the script will fail.

    Returns:
        config: Instance of python config object.
    """
    if os.path.isfile(CONFIG_FILE_NAME):
        config = configparser.RawConfigParser()
        config.readfp(open(CONFIG_FILE_NAME))
        return config
    else:
        raise FileNotFoundError(
            errno.ENOENT, 
            os.strerror(errno.ENOENT), 
            CONFIG_FILE_NAME
        )

# Import configuration file
try:
    config = load_config()
except FileNotFoundError as e:
    sys.exit(e)

# Load object detection library
os_root = Path(config.get('os-roots',os.name))
object_detection_lib_path = os_root / config.get('tensorflow','library')
sys.path.append(object_detection_lib_path)
sys.path.append(object_detection_lib_path / 'slim')

import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

# Pull variables from the configuration file.
PROJECT_NAME = config.get('model-export','project_name')
MODEL_BUILD_DIR = os_root / config.get('model-export','input_path')
OUTPUT_PATH = os_root / config.get('model-export','output_path')
MODEL_DIR = config.get('model-export','model_dir')
FINE_TUNED_MODEL_DIR = config.get('model-export','fine_tuned_model')

def export_inference_graph(
        input_type,
        pipeline_config_path,
        trained_checkpoint_prefix,
        output_directory,
        config_override = '',
        input_shape = None,
        write_inference_graph = False
    ):
    """Wrapper for TensorFlow inference graph export script.

    Args:
        input_type (string): Input type string
        pipeline_config_path (string): File system path to pipeline.config file.
        trained_checkpoint_prefix (string): File system path for storing trained checkpoints.
        output_directory (string): File system path for storing the model.
        config_override (str, optional): Override config path. Defaults to ''.
        input_shape (numeric, optional): Input dimensions, see export_tf_serving_model.py for options. Defaults to None.
        write_inference_graph (bool, optional): Flag to specify if the inference graph should be written to file. Defaults to False.
    """

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge(config_override, pipeline_config)
    if input_shape:
        input_shape = [
            int(dim) if dim != '-1' else None
            for dim in input_shape.split(',')
        ]
    else:
        input_shape = None

    exporter.export_inference_graph(input_type, pipeline_config,
                                  trained_checkpoint_prefix,
                                  output_directory, input_shape,
                                  write_inference_graph)

def main():
    """Creates folder for storing exported model checkpoint.
    """
    # Create base folder internal folder structure
    folder_name = '{}-{}'.format(PROJECT_NAME,datetime.datetime.now().strftime('%Y-%m-%d_%H-%S'))
    folder_name = os.path.join(OUTPUT_PATH,folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        os.makedirs(os.path.join(folder_name,MODEL_DIR))
        os.makedirs(os.path.join(folder_name,FINE_TUNED_MODEL_DIR))

    # Copy latest model checkpoint
    trained_model_dir = os.path.join(MODEL_BUILD_DIR,MODEL_DIR)
    model_files = list(filter(lambda x: 'model.ckpt' in x,os.listdir(trained_model_dir)))
    model_files.sort()
    model_ckpt = os.path.splitext(model_files[-1])[0]
    ckpt_files = glob.glob(os.path.join(trained_model_dir,model_ckpt+'.*'))
    ckpt_files.append(os.path.join(trained_model_dir,'checkpoint'))
    ckpt_files.append(os.path.join(trained_model_dir,'graph.pbtxt'))
    ckpt_files.append(os.path.join(MODEL_BUILD_DIR,'pipeline.config'))
    for ckpt_file in ckpt_files:
        shutil.copy(ckpt_file,os.path.join(folder_name,MODEL_DIR))

    # Copy label map
    shutil.copy(os.path.join(MODEL_BUILD_DIR,'pascal_label_map.pbtxt'),folder_name)

    export_inference_graph(
        input_type = 'image_tensor',
        pipeline_config_path = os.path.join(folder_name,MODEL_DIR,'pipeline.config'),
        trained_checkpoint_prefix = os.path.join(folder_name,MODEL_DIR,model_ckpt),
        output_directory = os.path.join(folder_name,FINE_TUNED_MODEL_DIR)
    )

if __name__ == '__main__':
    main()