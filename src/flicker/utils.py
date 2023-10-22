# Copyright 2023 Flicker Contributors
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
from __future__ import annotations
from typing import Iterable, Sized, Any

import numpy as np
from pyspark.sql import DataFrame


def is_nan_scalar(x: Any) -> bool:
    """ Check if the given value is a scalar NaN (Not a Number).

    Parameters
    ----------
    x: Any
        The value to be checked.

    Returns
    -------
    bool
        True if the value is a scalar NaN, False otherwise.

    Examples
    --------
    >>> is_nan_scalar(5)
    False

    >>> is_nan_scalar(float('nan'))
    True

    >>> is_nan_scalar('hello')
    False
    """
    result = False
    try:
        result = bool(np.isnan(x))
    except TypeError:
        pass
    return result


def get_length(iterable: Iterable | Sized) -> int:
    """ Get the length of an ``Iterable`` object.

    This method attempts to use the ``len()`` function. If ``len()`` is not available, this method attempts to count
    the number of items in ``iterable`` by iterating over the iterable. This iteration over ``iterable`` can be a
    problem for single-use ``Iterator``s.


    Parameters
    ----------
    iterable: Iterable | Sized
        The iterable object for which the length is to be determined

    Returns
    -------
    int
        The length of the iterable object

    Raises
    ------
    TypeError
        If the length of the iterable object cannot be determined.

    Examples
    --------
    >>> get_length([1, 2, 3, 4, 5])
    5

    >>> get_length('Hello, World!')
    13

    >>> get_length({'a': 1, 'b': 2, 'c': 3})
    3

    >>> get_length(range(4))
    4
    """
    try:
        length = len(iterable)
    except TypeError as e:
        if 'len()' in str(e):
            length = sum([1 for _ in iterable])
        else:
            raise e
    return length


def get_names_by_dtype(df: DataFrame, dtype: str) -> list[str]:
    """ Get the list of column names that match the specified data type.

    Parameters
    ----------
    df: pyspark.sql.DataFrame
        The input ``pyspark.sql.DataFrame``
    dtype: str
        The data type to filter the column names. Example: ``'bigint'``.

    Returns
    -------
    list[str]
        A list of column names that match the specified data type

    Examples
    --------
    >>> spark = SparkSession.builder.getOrCreate()
    >>> rows = [(1, 3.4, 1), (3, 4.5, 2)]
    >>> df = spark.createDataFrame(rows, schema=['col1', 'col2', 'col3'])
    >>> get_names_by_dtype(df, 'bigint')
    ['col1', 'col3']
    """
    return [
        name for name, dtype_ in df.dtypes
        if dtype_ == dtype
    ]
