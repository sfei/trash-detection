#!/usr/bin/env python

"""gen_tfrecords.py: Generate TensorFlow records for model training and evaluation."""

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

import os
import io
import sys
import pandas as pd
import numpy as np
import glob
import random
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
from collections import namedtuple, OrderedDict

# Add TensorFlow model API to python library path, including slim subdirectory
sys.path.append(r'[Path to Object Detection API]\research')
sys.path.append(r'[Path to Object Detection API]\research\slim')

# Load TensorFlow library and object detection utils
import tensorflow as tf
from object_detection.utils import dataset_util


# Data paths
DATA_PATH = 'Path to data'
CROPPED_IMAGES_PATH = 'Path to cropped images'
CROPPED_IMAGE_DIR = 'Path to cropped images'

# Array for image paths
IMAGE_PATHS = []

def class_text_to_int(row_label):
    """Returns numeric id for class label.

    Args:
        row_label (string): 'plastic', 'not-plastic', or 'unknown'

    Returns:
        ing: Numeric label id.
    """
    if row_label == 'plastic':
        return 1
    elif row_label == 'not-plastic':
        return 2
    elif row_label == 'unknown':
        return 3
    else:
        None

def image_res(filename):
    """Returns string containing file size info.

    Args:
        filename (string): System path to image file.

    Returns:
        string: Formatted string for terminal printing.
    """
    im = Image.open(filename)
    return "The resolution of the image is {} x {}".format(im.size[0],im.size[1])

def output_jpeg_dimensions(jpeg_folder_path):
    """Iterate over folder containing images and print resolutions and counts.

    Args:
        jpeg_folder_path (string): System path to directory containing images.
    """
    responses = {}
    jpeg_files = list(filter(lambda x: '.jpg' in x.lower(), os.listdir(jpeg_folder_path)))
    for jpeg_file in jpeg_files:
        jpeg_file_path = os.path.join(jpeg_folder_path, jpeg_file)
        response = image_res(jpeg_file_path)
        if response in responses.keys():
            responses[response] += 1
        else:
            responses[response] = 1

    for response in responses.keys():
        print("{}: {}".format(response,responses[response]))

def group_df_by_column(df, group):
    """Groups annotations by column.

    Args:
        df (pd.dataframe): Pandas data frame containing annotation information.
        group (string): Column to group entries by.

    Returns:
        list: Grouped records.
    """
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def write_imageset(df, output_path):
    """Write dataframe to CSV

    Args:
        df (pd.dataframe): Pandas data frame for writing to csv.
        output_path (string): System path for CSV file location.
    """
    df.to_csv(output_path, index=None)

def create_tf_example(group, path):
    """Return a TensorFlow example for inclusion in TF record.

    Args:
        group (list): Grouped records, grouped by filename.
        path (string): System path to data.

    Returns:
        tf.train.Example: TensorFlow Example instance.
    """
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def crop_image_into_boxes(img_path,output_dir):
    """Find height/width dimensions that are a multiple of the original
    height and width under max_dim (set to 800).

    Args:
        img_path (string): System path to image file.
        output_dir (string): System path for storing cropped images.
    """
    img = Image.open(img_path)
    width, height = img.size

    # Find x such that width/(2^x) < 800
    max_dim = 800

    width_divisor = 1
    while (width / width_divisor) > max_dim:
        width_divisor = width_divisor * 2

    height_divisor = 1
    while (height / height_divisor) > max_dim:
        height_divisor = height_divisor * 2


    h_step = int(width / width_divisor)
    v_step = int(height / height_divisor)
    h_steps = range(0,width,h_step)
    v_steps = range(0,height,v_step)

    # images = []

    for j in v_steps:
        for i in h_steps:
            sub_img_file_name = os.path.basename(img_path)
            sub_img_file_name = sub_img_file_name.replace(".jpg","_{}-{}_{}-{}.jpg".format(i,j,i+h_step,j+v_step))
            sub_img_path = os.path.join(output_dir,sub_img_file_name)
            sub_img = img.crop(
                (
                    i,
                    j,
                    i+h_step,
                    j+v_step
                )
            )
            sub_img.save(sub_img_path)

def get_image_paths(row):
    """Find unique filenames

    Args:
        row (list): Annotation record.
    """
    if row['filename'] not in IMAGE_PATHS:
        IMAGE_PATHS.append(row['filename'])

def crop_images_into_boxes(df,output_path):
    """Find filenames for images with annotations and crop.

    Args:
        df (pd.dataframe): Annotation data contained in pandas dataframe.
        output_path (string): System path for storing cropped images.
    """
    df.apply(lambda row: get_image_paths(row), axis=1)
    for image_path in IMAGE_PATHS:
        crop_image_into_boxes(image_path, output_path)


def write_crops(row):
    """Load an image, crop using pre-defined bounds, save to system folder.

    Args:
        row (list): Annotation record.
    """
    im = Image.open(row['filename'])
    bounds = (row['xmin'],row['ymin'],row['xmax'],row['ymax'])
    im_crop = im.crop(bounds)

    # Swap root
    (root,filename) = os.path.split(row['filename'])
    new_filename = os.path.join(os.path.join(DATA_PATH,'cropped_images'),filename)

    # Construct new filename
    (root, ext) = os.path.splitext(new_filename)
    filename_out = "{}-{}-{}-{}-{}{}".format(root,row['xmin'],row['ymin'],row['xmax'],row['ymax'],ext)
    
    # Save
    im_crop.save(filename_out)

def make_crops(xml_df):
    """Apply crop function row by row.

    Args:
        xml_df (pd.dataframe): Pandas data frame containing annotation data.
    """
    xml_df.apply(write_crops, axis=1)

def xml_to_df(annotations_path):
    """Convert PASCAL VOC XML file to record in a dataframe.

    Args:
        annotations_path (string): Folder containing PASCAL VOC XML files.

    Returns:
        pd.dataframe: Dataframe containing annotation data.
    """
    xml_list = []
    xml_files = list(filter(lambda x: '.xml' in x, os.listdir(annotations_path)))
    for xml_file in xml_files:
        xml_file_path = os.path.join(annotations_path,xml_file)
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('path').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    
    return xml_df  

def get_box_steps(width,height):
    """Obtain crop dimension steps which are a multiple of original dimensions, 
    under max_dim which is set to 800.

    Args:
        width (int): Image width.
        height (int): Image height.

    Returns:
        tuple: Height and width step information for cropped dimensions.
    """
    max_dim = 800
    width_divisor = 1
    while (width / width_divisor) > max_dim:
        width_divisor = width_divisor * 2

    height_divisor = 1
    while (height / height_divisor) > max_dim:
        height_divisor = height_divisor * 2


    h_step = int(width / width_divisor)
    v_step = int(height / height_divisor)
    h_steps = range(0,width,h_step)
    v_steps = range(0,height,v_step)

    return (h_steps, h_step, v_steps, v_step)

def get_cropped_filename(row,cropped_image_dir):
    """Obtain cropped image filename.

    Args:
        row (list): Annotation record.
        cropped_image_dir (string): System path for cropped images.

    Returns:
        string: Cropped image filename.
    """

    # Get image dimensions
    img_path = row['filename']
    img = Image.open(img_path)
    width, height = img.size

    # Obtain step information
    h_steps, h_step, v_steps, v_step = get_box_steps(width,height)

    # Get bbox
    xmin = int(row['xmin'])
    xmax = int(row['xmax'])
    ymin = int(row['ymin'])
    ymax = int(row['ymax'])

    sub_img_path = row['filename']

    for j in v_steps:
        for i in h_steps:
            # Keep if box is in range for one of the steps
            if i <= xmin < i+h_step and i < xmax <= i+h_step:
                if j <= ymin < j+v_step and j < ymax <= j+v_step:
                    # filename
                    sub_img_file_name = os.path.basename(row['filename'])
                    sub_img_file_name = sub_img_file_name.replace(".jpg","_{}-{}_{}-{}.jpg".format(i,j,i+h_step,j+v_step))
                    sub_img_path = os.path.join(cropped_image_dir,sub_img_file_name)

    return sub_img_path

def filter_for_crops(row):
    """Identify if an annotation should be used for experimentation,
    this allows us to ignore images that span multiple cropped images.

    Args:
        row (pd.dataframe): List containing annotation record.

    Returns:
        boolean: Boolean flag indicating if a record should be used.
    """

    # Get image dimensions
    img_path = row['filename']
    img = Image.open(img_path)
    width, height = img.size

    # Obtain step information
    h_steps, h_step, v_steps, v_step = get_box_steps(width,height)

    # Get bbox
    xmin = int(row['xmin'])
    xmax = int(row['xmax'])
    ymin = int(row['ymin'])
    ymax = int(row['ymax'])

    keep = False

    for j in v_steps:
        for i in h_steps:
            # Keep if box is in range for one of the steps
            if i <= xmin < i+h_step and i < xmax <= i+h_step:
                if j <= ymin < j+v_step and j < ymax <= j+v_step:
                    keep = True

    return keep

def update_bbox(df):
    """Shift annotation bounding box coordinates from the original image
    coordinates to cropped image coordinates.

    Args:
        df (pd.dataframe): Annotation data.

    Returns:
        pd.dataframe: Updated annotation data.
    """
    for index, row in df.iterrows():
        # Write row to df_copy

        # Get image dimensions
        img_path = row['filename']
        img = Image.open(img_path)
        width, height = img.size

        # Get cropped image dimensions
        # img_crop_path = row['filename2']
        img_crop = Image.open(row['filename2'])
        width_crop, height_crop = img_crop.size

        # Obtain step information
        h_steps, h_step, v_steps, v_step = get_box_steps(width,height)

        # Get bbox
        xmin = int(row['xmin'])
        xmax = int(row['xmax'])
        ymin = int(row['ymin'])
        ymax = int(row['ymax'])

        for j in v_steps:
            for i in h_steps:
                # Keep if box is in range for one of the steps
                if i <= xmin < i+h_step and i < xmax <= i+h_step:
                    if j <= ymin < j+v_step and j < ymax <= j+v_step:
                        xmin = xmin - i
                        xmax = xmax - i
                        ymin = ymin - j
                        ymax = ymax - j

        df.at[index,'xmin'] = xmin
        df.at[index,'xmax'] = xmax
        df.at[index,'ymin'] = ymin
        df.at[index,'ymax'] = ymax
        df.at[index,'width'] = width_crop
        df.at[index,'height'] = height_crop

    return df

def gen_crops():
    """Run make_crops function over PASCAL VOC XML data.
    """
    # Read in annotation file(s)
    xml_df = xml_to_df(os.path.join(DATA_PATH,'annotations'))

    # Create cropped images
    make_crops(xml_df)

def gen_dataset_csv():
    """Write PASCAL VOC annotation data to csv file.
    """
    # Read in annotation file(s)
    xml_df = xml_to_df(os.path.join(DATA_PATH,'annotations'))

    # Filter rows with samples along crop boundaries
    xml_df['keep'] = xml_df.apply(lambda row: filter_for_crops(row), axis=1)
    xml_df = xml_df[xml_df['keep'] == True]

    # Get cropped image filenames
    xml_df['filename2'] = xml_df.apply(lambda row: get_cropped_filename(row,CROPPED_IMAGE_DIR), axis=1)

    df = update_bbox(xml_df)

    df['filename'] = df['filename2']
    del df['filename2']

    # Write to csv
    df.to_csv(os.path.join(DATA_PATH,'data.csv'), index=None)

def gen_imagesets(df, split):
    """Construct training and testing sets.

    Args:
        df (pd.dataframe): Annotation data.
        split (dict): Train and test percentage.

    Returns:
        tuple: Train and test lists stored as a tuple.
    """
    # train_set = pd.DataFrame(xml_list, columns=column_name)
    train_sets = []
    test_sets = []
    for label in df['class'].unique():
        filtered_df = df.loc[df['class'] == label]
        np.random.seed(split['seed'])
        mask = np.random.rand(len(filtered_df)) < split['train']
        train_sets.append(filtered_df[mask])
        test_sets.append(filtered_df[~mask])

    train_set = pd.concat(train_sets)
    test_set = pd.concat(test_sets)

    return (train_set, test_set)

def gen_tfrecords():
    """Generate TensorFlow records for training and testing sets.
    """
    split = {
        'train': 0.8,
        'seed': 43
    }

    df = pd.read_csv(os.path.join(DATA_PATH,'data.csv')) 

    # get train and test sets
    (train_set, test_set) = gen_imagesets(df,split)

    # Write training data
    original_images_path = 'Path to cropped images'
    tf_record_path = 'Path to store TensorFlow records.'
    tf_record_output_file_path = os.path.join(tf_record_path,'train.record')
    writer = tf.python_io.TFRecordWriter(tf_record_output_file_path)
    grouped = group_df_by_column(train_set, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, original_images_path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecords: {}'.format(tf_record_output_file_path))

    # Write test data
    tf_record_output_file_path = os.path.join(tf_record_path,'test.record')
    writer = tf.python_io.TFRecordWriter(tf_record_output_file_path)
    grouped = group_df_by_column(test_set, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, original_images_path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecords: {}'.format(tf_record_output_file_path))

def main():
    # Use function below to generate csv file, best to complete
    # this before generating tf records.
    # gen_dataset_csv()

    gen_tfrecords()

if __name__ == '__main__':
    main()