#!/usr/bin/env python

"""data_statistics.py: Functions used for gathering information about the dataset."""

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
import platform
import pandas as pd
import helpers
import xml.etree.ElementTree as ET
from pathlib import Path

# Load system paths from configuration file
try:
    config = helpers.load_config()
except FileNotFoundError as e:
    sys.exit(e)

OS_ROOT = Path(config.get('os-roots',os.name))
ANNOTATIONS_PATH = OS_ROOT / config.get('data','annotations')
RECORDS_PATH = OS_ROOT / config.get('data','records')
TF_RECORDS_PATH = OS_ROOT / config.get('data','tf_records')

def print_annotation_paths(annotations_path):
    """Display paths to annotation file paths, used for annotations stored via PASCAL VOC XML files.

    Args:
        annotations_path (string): System path to PASCAL VOC XML files.
    """
    paths = []
    xml_files = list(filter(lambda x: '.xml' in x, os.listdir(annotations_path)))
    for xml_file in xml_files:
        # xml_file_path = os.path.join(annotations_path, xml_file)
        xml_file_path = annotations_path / xml_file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for path in root.iter('path'):
            (head, tail) = os.path.split(path.text)
            if head not in paths:
                paths.append(head)
    
    print("Unique Annotation Paths:")
    print("")
    for path in paths:
        print(path)


def get_record_stats(records_csv_file):
    """Display summary information for annotation data stored via CSV.

    Args:
        records_csv_file (string): System path to CSV file containing annotation data.
    """
    df = pd.read_csv(records_csv_file)
    total_counts  = df['filename'].count()
    class_counts = df.groupby('class').count()

    print("Total Records:")
    print(total_counts)
    print("")
    print("TensorFlow Record counts by class:")
    print(class_counts)

def get_annotation_stats(annotations_path):
    """Display summary information gathered from PASCAL VOC XML files.

    Args:
        annotations_path (string): Path to directory containing PASCAL VOC XML files.
    """
    counts = {
        'plastic': 0,
        'unknown': 0,
        'not-plastic': 0
    }
    xml_files = list(filter(lambda x: '.xml' in x, os.listdir(annotations_path)))
    for xml_file in xml_files:
        # xml_file_path = os.path.join(annotations_path, xml_file)
        xml_file_path = annotations_path / xml_file
        xml_file_counts = get_class_counts(xml_file_path)
        for key in counts.keys():
            counts[key] += xml_file_counts[key]
    
    print("TensorFlow Annotation counts by class:")
    total_counts = 0
    for key in counts.keys():
        total_counts += counts[key]
        print("{}: {}".format(key,counts[key]))
    print("total: {}".format(total_counts))

def get_class_counts(xml_file_path):
    """Obtain counts of classes stored in a given PASCAL VOC XML file.

    Args:
        xml_file_path (string): System file to PASCAL VOC XML file.

    Returns:
        dict: Dictionary containing counts for each class, hardcoded in this function.
    """
    counts = {
        'plastic': 0,
        'unknown': 0,
        'not-plastic': 0
    }
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    for name in root.iter('name'):
        counts[name.text] += 1
    return counts

def get_data_counts():
    """Calls summary print functions.
    """
    print("")
    get_record_stats(RECORDS_PATH)
    print("")
    get_annotation_stats(ANNOTATIONS_PATH)

def get_files_from_folder(path):
    """Populate a list containing full system paths for files contained within.

    Args:
        path (string): System path to folder containing files.

    Returns:
        list: List of full paths for files within folder.
    """
    file_list = []
    for subdir, dirs, files in os.walk(path):
        for f in files:
            file_list.append( Path(path) / Path(subdir) / f )
    return file_list

def main():
    """Primary process for running any of the above functions via terminal.
    """

    print(get_files_from_folder(TF_RECORDS_PATH))

if __name__ == '__main__':
    main()