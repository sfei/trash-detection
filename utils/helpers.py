#!/usr/bin/env python

"""helpers.py: Some general helper functions."""

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

import errno
import os
import configparser
from pathlib import Path

# Configuration file path.
CONFIG_FILE_NAME = Path("../config.ini")

def load_config():
    """Returns config instance.

    Raises:
        FileNotFoundError: Flag error if file isn't found.

    Returns:
        configparser.RawConfigParser: Config instance.
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

def timezone_offset(row, offset, field):
    """Find timezone offest from string.

    Args:
        row (list): Record from dataframe or array.
        offset (string): Timezone offset represented as string.
        field (string): Column name to check for offset.

    Returns:
        string: Timestamp string.
    """
    if not isinstance(row[field], float):
        if row[field][-5:] == offset:
            return row[field]
        else:
            return row[field] + offset
    else:
        return row[field]

def update_df_timezone(df, field, offset):
    """Update timezone in a dataframe.

    Args:
        df (pd.dataframe): dataframe containing data.
        field (string): Column name for field containing datetime info.
        offset (string): Timezone offset

    Returns:
        pd.dataframe: Dataframe with updated datetime data.
    """
    field_updated = '{}_updated'.format(field)
    df[field_updated] = df.apply(lambda row: timezone_offset(row,offset,field), axis=1)
    df[field] = df[field_updated]
    del df[field_updated]
    return df

def get_files_from_folder(path):
    """Helper for obtaining directory file listing including full paths.

    Args:
        path (string): System path to folder to get list.

    Returns:
        list: List of file paths.
    """
    file_list = []
    for subdir, dirs, files in os.walk(path):
        for f in files:
            file_list.append( Path(subdir) / f )
    return file_list