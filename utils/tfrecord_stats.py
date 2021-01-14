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
import helpers
import platform
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Load configuration data
try:
    config = helpers.load_config()
except FileNotFoundError as e:
    sys.exit(e)

OS_ROOT = Path(config.get('os-roots',os.name))
TF_RECORDS_PATH = OS_ROOT / config.get('data','tf_records')

def print_tf_records(tf_records_path):
    """Print records to the terminal.

    Args:
        tf_records_path (string): System path to tensorflow records folder.
    """
    record_files = list(filter(lambda x: '.record' in x, os.listdir(tf_records_path)))
    cnt = 0
    for fn in tf_records_filenames:
        for record in tf.python_io.tf_record_iterator(fn):
            print(record)
            cnt += 1

def count_tf_records(tf_records_path):
    """Obtain number of records stored in tensorflow record file.

    Args:
        tf_records_path (string): System path to folder containing tensorflow records.

    Returns:
        dict: Python dictionary containing number of records stored in tensorflow record file.
    """
    record_counts = {}
    for subdir, dirs, files in os.walk(tf_records_path):
        for f in files:
            if '.record' == os.path.splitext(f)[1]:
                full_path = tf_records_path / subdir / f
                cnt = 0
                for record in tf.python_io.tf_record_iterator(str(full_path)):
                    cnt += 1
                record_counts[full_path] = cnt
    return record_counts

def print_tf_record_count_dict(record_counts):
    """Print record counts to terminal.

    Args:
        record_counts (dict): Dictionary containing record counts.
    """
    for full_path in record_counts.keys():
        print('{}: {}'.format(full_path.absolute(),record_counts[full_path]))

def print_tf_record_counts(path):
    """Print record counts to terminal.

    Args:
        path (string): System path to tensorflow records.
    """
    for subdir, dirs, files in os.walk(path):
        for dir in dirs:
            record_counts = count_tf_records( path / dir )
            print_tf_record_count_dict(record_counts)
            print()

def get_tf_record_stats(tf_records_path):
    """Print TensorFlow records to terminal.

    Args:
        tf_records_path (string): System path to folder containing TensorFlow records.
    """
    for subdir, dirs, files in os.walk(tf_records_path):
        for f in files:
            # print(f)
            if '.record' == os.path.splitext(f)[1]:
                full_path = tf_records_path / subdir / f
                # print(full_path)
                for record in tf.python_io.tf_record_iterator(str(full_path)):
                    print(tf.train.Example.FromString(record))

def main():
    print_tf_record_counts(TF_RECORDS_PATH)

if __name__ == '__main__':
    main()