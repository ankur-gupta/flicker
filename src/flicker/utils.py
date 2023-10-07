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
    result = False
    try:
        result = bool(np.isnan(x))
    except TypeError:
        pass
    return result


def get_length(iterable: Iterable | Sized) -> int:
    """
    Parameters
    ----------
    iterable : Iterable | Sized
        The iterable object for which the length is to be determined.

    Returns
    -------
    int
        The length of the iterable object.

    Raises
    ------
    TypeError
        If the length of the iterable object cannot be determined.

    Notes
    -----
    This method uses the `len()` function to determine the length of the iterable object. If the `len()` function
    throws a TypeError, the method attempts to calculate the length by iterating over the iterable and counting the
    elements. If any other TypeError occurs during this process, it is raised as an exception.

    Examples
    --------
    >>> lst = [1, 2, 3, 4, 5]
    >>> get_length(lst)
    5

    >>> s = 'Hello, World!'
    >>> get_length(s)
    13

    >>> d = {'a': 1, 'b': 2, 'c': 3}
    >>> get_length(d)
    3
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
    """ Get the list of column names with the specified data type.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The input DataFrame.

    dtype : str
        The data type to filter the column names.

    Returns
    -------
    list[str]
        A list of column names with the specified data type.

    """
    return [
        name for name, dtype_ in df.dtypes
        if dtype_ == dtype
    ]
