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
from contextlib import contextmanager
from .mkname import mkname
from .udf import len_udf
from .dataframe import FlickerDataFrame


@contextmanager
def delete_extra_columns(df: FlickerDataFrame):
    """ This context manager exists to provide a commonly needed functionality.
    Unlike pandas which lets you compute a temporary quantity in a
    separate Series or a numpy array, pyspark requires you to create a new
    column even for temporary quantities.

    This context manager makes sure that any new columns that you create
    within the context manager will get deleted (in-place) afterwards
    even if your code encounters an Exception. Any columns that you start
    with will not be deleted (unless of course you deliberately delete
    them yourself).

    Note that this context manager will not prevent you from overwriting
    any column (new or otherwise).

    Parameters
    ----------
    df : FlickerDataFrame
        The FlickerDataFrame object from which extra columns will be deleted.

    Yields
    ------
    names_to_keep : list
        A list of column names to keep after deleting extra columns.

    Raises
    ------
    TypeError
        If `df` is not an instance of FlickerDataFrame.

    Examples
    --------
    >>> spark = SparkSession.builder.getOrCreate()
    >>> from flicker import delete_extra_columns
    >>> df = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'])
    >>> df
    FlickerDataFrame[a: double, b: double]
    >>> with delete_extra_columns(df) as names_to_keep:
    ...     print(names_to_keep)
    ...     df['c'] = 1
    ...     print(df.names)
    ['a', 'b']
    ['a', 'b', 'c']
    >>> print(df.names)
    ['a', 'b']  # 'c' column is deleted automatically
    """
    if not isinstance(df, FlickerDataFrame):
        raise TypeError(f'df must be FlickerDataFrame; you provided {type(df)}')
    names_to_keep = list(df.names)
    yield names_to_keep
    for name in df.names:
        if name not in names_to_keep:
            del df[name]


def find_empty_columns(df: FlickerDataFrame, verbose: bool = True) -> list[str]:
    """ A very opinionated function that returns the names of 'empty' columns in a ``FlickerDataFrame``.

    A column is considered empty if all of its values are None or have length 0. Note that a column with all NaNs is
    not considered empty.

    Parameters
    ----------
    df: FlickerDataFrame
        The DataFrame object to check for empty columns
    verbose: bool, optional
        Flag indicating whether to print progress information while checking
        the columns. Default is True.

    Returns
    -------
    list[str]
        A list of names of empty columns found in the DataFrame

    Raises
    ------
    TypeError
        If the provided ``df`` parameter is not of type ``FlickerDataFrame``

    Examples
    --------
    >>> import numpy as np
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = FlickerDataFrame.from_shape(spark, 3, 2, names=['col1', 'col2'], fill='rowseq')
    >>> df['col3'] = None
    >>> df['col4'] = np.nan
    >>> empty_cols = find_empty_columns(df)
    >>> print(empty_cols)
    ['col3']
    """
    if not isinstance(df, FlickerDataFrame):
        raise TypeError(f'df must be FlickerDataFrame; you provided {type(df)}')

    empty_names = []  # list to store the names of empty columns
    with delete_extra_columns(df) as names:
        for i, name in enumerate(names):
            if verbose:
                print(f'Checking {i + 1}/{len(names)}: {name}')

            # Generate a new unique name. Note that we must use the latest
            # list of names (and not `names`) here.
            len_name = mkname(df.names, prefix=f'len_{name}', suffix='')

            # 1. Modify df in-place by using a 'mutable' function
            # 2. len_udf returns a length 1 for 'scalar'/'atomic' objects
            #    that don't have a __len__ attribute.
            df[len_name] = len_udf(df[name]._column)
            if df[df[len_name] > 0].nrows == 0:
                if verbose:
                    print(f'{name} was found to be empty')
                empty_names = empty_names + [name]
    return empty_names
