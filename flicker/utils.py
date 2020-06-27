# Copyright 2020 Flicker Contributors
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from builtins import range

import random
import numpy as np
import pandas as pd

CHARACTERS = "abcdefghijklmnopqrstuvwxyz0123456789"


def gensym(names=[], prefix='col_', suffix='', n_max_tries=100,
           n_random_chars=4):
    """ Generate a new, unique column name that is different from
        all existing columns.
    """
    names = set(names)

    # Try out without any randomness first
    candidate = prefix + suffix
    if candidate in names:
        for i in range(n_max_tries):
            stub = ''.join([random.choice(CHARACTERS)
                            for _ in range(n_random_chars)])
            candidate = '{}{}{}'.format(prefix, stub, suffix)
            if candidate not in names:
                return candidate
    else:
        return candidate
    msg = 'No unique name generated in {} tries with {} random characters'
    raise KeyError(msg.format(n_max_tries, n_random_chars))


def get_float_column_names(df):
    """
    Returns a list of the column names in a pandas DataFrame that have the
    dtype float (of any precision). Note that the pandas DataFrame cannot
    have duplicate column names.

    Parameters
    ----------
    df: pandas DataFrame
        There can be no duplicate column names in the dataframe

    Returns
    -------
        List[str]

    Examples
    --------
    >>> # Example 1
    >>> df = pd.DataFrame({
        'a': [np.nan, 1.3, np.nan],
        'b': [True, False, True],
        'c': ['spark', np.nan, None],
        'd': [1, 2, 3]
    })

    >>> df.dtypes
    a    float64
    b       bool
    c     object
    d      int64
    dtype: object

    >>> get_float_column_names(df)
    ['a']

    >>> # Example 2 - 'object' dtype is not considered float
    >>> df = pd.DataFrame({'a': [np.nan, 1.3, np.nan, None]}, dtype='object')
    >>> df.dtypes
    a    object
    dtype: object

    >>> get_float_column_names(df)
    []
    """
    if not isinstance(df, pd.DataFrame):
        msg = 'df of type="{}" is not a pandas DataFrame'
        raise TypeError(msg.format(str(type(df))))
    if len(set(df.columns)) != len(df.columns):
        msg = 'df contains duplicated column names which is not supported'
        raise ValueError(msg)
    return list(set(df.select_dtypes(include=[np.floating]).columns))


def get_non_float_column_names(df):
    """
    Returns a list of the column names in a pandas DataFrame that don't have
    the dtype float (of any precision). Note that the pandas DataFrame cannot
    have duplicate column names.

    Parameters
    ----------
    df: pandas DataFrame
        There can be no duplicate column names in the dataframe

    Returns
    -------
        List[str]

    Examples
    --------
    >>> # Example 1
    >>> df = pd.DataFrame({
        'a': [np.nan, 1.3, np.nan],
        'b': [True, False, True],
        'c': ['spark', np.nan, None],
        'd': [1, 2, 3]
    })

    >>> df.dtypes
    a    float64
    b       bool
    c     object
    d      int64
    dtype: object

    >>> get_non_float_column_names(df)
    ['c', 'b', 'd']

    >>> # Example 2 - 'object' dtype is not considered float
    >>> df = pd.DataFrame({'a': [np.nan, 1.3, np.nan, None]}, dtype='object')
    >>> df.dtypes
    a    object
    dtype: object

    >>> get_non_float_column_names(df)
    ['a']
    """
    if not isinstance(df, pd.DataFrame):
        msg = 'df of type="{}" is not a pandas DataFrame'
        raise TypeError(msg.format(str(type(df))))
    if len(set(df.columns)) != len(df.columns):
        msg = 'df contains duplicated column names which is not supported'
        raise ValueError(msg)
    return list(set(df.select_dtypes(exclude=[np.floating]).columns))
