# Copyright 2020 Ankur Gupta
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

import copy

import six
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Column
from pyspark.sql.functions import lit, isnan

from flicker.compat import Iterable

PYSPARK_FLOAT_DTYPES = {'double', 'float'}


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


class FlickerDataFrame(object):
    """
    A thin wrapper over pyspark.sql.DataFrame that provides a pandas-like
    API over pyspark DataFrames without compromising performance.
    """

    def __init__(self, df):
        """
        Default constructor for FlickerDataFrame. Returns a FlickerDataFrame
        using a pyspark DataFrame as input.


        Parameters
        ----------
        df: pyspark.sql.DataFrame

        Returns
        -------
            FlickerDataFrame

        See Also
        --------
        FlickerDataFrame.from_pandas
        FlickerDataFrame.from_dict
        FlickerDataFrame.from_records
        FlickerDataFrame.from_rows
        FlickerDataFrame.from_items
        FlickerDataFrame.from_columns
        FlickerDataFrame.from_shape

        Examples
        --------
        >>> spark_df = spark.createDataFrame([(x, x) for x in range(5)],
                                             'a INT, b INT')
        >>> df = FlickerDataFrame(spark_df)
        >>> df
        FlickerDataFrame[a: int, b: int]

        >>> df()  # show first 5 rows
           a  b
        0  0  0
        1  1  1
        2  2  2
        3  3  3
        4  4  4
        """
        if not isinstance(df, pyspark.sql.DataFrame):
            msg = 'df of type "{}" is not a pyspark DataFrame object'
            raise TypeError(msg.format(str(type(df))))
        if len(set(df.columns)) != len(df.columns):
            msg = 'df contains duplicated column names which is not supported'
            raise ValueError(msg)
        self._df = df
        self._nrows = None
        self._ncols = None

    def _reset(self, df):
        """ This function makes objects of this class mutable. """
        if not isinstance(df, pyspark.sql.DataFrame):
            msg = 'type(df)="{}" is not a pyspark DataFrame object'
            raise TypeError(msg.format(msg.format(str(type(df)))))
        self._df = df
        self._nrows = None
        self._ncols = None

    @classmethod
    def from_pandas(cls, spark, df,
                    convert_nan_to_null_in_non_float=True,
                    convert_nan_to_null_in_float=False,
                    copy_df=False):
        """
        Construct a FlickerDataFrame from pandas.DataFrame. Note that this
        method may modify the input pandas DataFrame; please make a copy of
        the pandas DataFrame before calling this function if you want to
        prevent that.

        Parameters
        ----------
        spark: pyspark.sql.SparkSession
            SparkSession object. This can be manually created by something
            like: `spark = SparkSession.builder.appName('app').getOrCreate()`
        df: pandas.DataFrame
            Note that this dataframe may be modified in place; make a copy
            first if you don't want that.
        convert_nan_to_null_in_non_float: bool
            If True (recommended), we convert np.nan (which has the
            type 'float') into None in the non-float columns. Unlike a pandas
            DataFrame, a pyspark DataFrame does not allow a np.nan in a
            non-float/non-double column. Note that any 'object' type column
            in `df` will be considered as a non-float column.
        convert_nan_to_null_in_float: bool
            If True, we convert np.nan (which has the type 'float') into None
            in all float/double columns. A pyspark DataFrame allows both
            np.nan and None to exist in a (nullable) float or double column.
        copy_df: bool
            If True, we deepcopy `df` if before modifying it.
            This prevents original `df` value from modifying. If False,
            we don't make a copy of `df`. `copy_df=True` is good for small
            datasets but `copy_df=False` (default) should be useful for large
            `df`.

        Returns
        -------
            FlickerDataFrame

        Examples
        --------
        >>> pdf = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [1.2, None, np.nan, 4.5],
            'c': ['spark', None, 'flicker', np.nan]
        })

        >>> pdf
           a    b        c
        0  1  1.2    spark
        1  2  NaN     None
        2  3  NaN  flicker
        3  4  4.5      NaN

        >>> pdf.dtypes
        a      int64
        b    float64
        c     object
        dtype: object

        >>> # Example 1
        >>> df = FlickerDataFrame.from_pandas(spark, pdf, copy_df=True)
        >>> df
        FlickerDataFrame[a: bigint, b: double, c: string]

        >>> df.show()
        +---+---+-------+
        |  a|  b|      c|
        +---+---+-------+
        |  1|1.2|  spark|
        |  2|NaN|   null|
        |  3|NaN|flicker|
        |  4|4.5|   null|
        +---+---+-------+

        >>> # Example 2
        >>> df = FlickerDataFrame.from_pandas(
                spark, pdf, convert_nan_to_null_in_non_float=False,
                copy_df=True)
        # Will fail expectedly because string-type column 'c' has a np.nan
        # which is not allowed in a pyspark DataFrame (unless we convert np.nan
        # to None).

        >>> # Example 3  (complicated example that can be skipped)
        >>> pdf_double_as_object = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [1.2, None, np.nan, 4.5]
        }, dtype=object)
        >>> pdf_double_as_object
           a     b
        0  1   1.2
        1  2  None
        2  3   NaN
        3  4   4.5

        >>> pdf_double_as_object.dtypes
        a    object
        b    object
        dtype: object

        >>> df = FlickerDataFrame.from_pandas(
                spark, pdf_double_as_object,
                convert_nan_to_null_in_non_float=False,
                convert_nan_to_null_in_float=False, copy_df=True)
        >>> df.show()
        +---+----+
        |  a|   b|
        +---+----+
        |  1| 1.2|
        |  2|null|
        |  3| NaN|
        |  4| 4.5|
        +---+----+

        Note that we use `df.show()` instead of `df()` here because converting
        a pyspark DataFrame to a pandas DataFrame would convert all nulls to
        np.nan in float columns.
        """
        if not isinstance(spark, SparkSession):
            msg = 'spark of type "{}" is not a SparkSession object'
            raise TypeError(msg.format(str(type(spark))))
        if not isinstance(df, pd.DataFrame):
            msg = 'df of type "{}" is not a pandas DataFrame object'
            raise TypeError(msg.format(str(type(df))))

        # Do this check early and within pandas to make rest of the code
        # easier.
        if len(set(df.columns)) != len(df.columns):
            msg = 'df contains duplicated column names which is not supported'
            raise ValueError(msg)

        # Even though .isnull() finds the np.nan's, assigning to None
        # does not convert them to None(s) in float columns. But this
        # works for other types of columns!
        # An np.nan in a non-float column would make spark.createDataFrame()
        # call fail. This is why convert_nan_to_null_in_non_float=True,
        # by default.
        if convert_nan_to_null_in_non_float:
            non_float_column_names = get_non_float_column_names(df)
            if (len(non_float_column_names) > 0) and copy_df:
                # We may modify df. Make a copy if needed.
                df = copy.deepcopy(df)
            for name in non_float_column_names:
                # If we simply perform, df.loc[df[name].isnull(), name] = None
                # we accidentally convert bool -> double even when we don't
                # have any np.nan or None(s).
                nulls_logical = df[name].isnull()
                if any(nulls_logical):
                    df.loc[nulls_logical, name] = None

        # A pandas float column automatically converts None to np.nan.
        # But, if a pandas column is of type 'object', None's remain as-is.
        # Instead of trying to find them all, we'll let pyspark do this for us.
        # pyspark will convert even 'object' type pandas columns to a 'double'
        # type pyspark column.
        df_spark = spark.createDataFrame(df)
        if convert_nan_to_null_in_float:
            # Typically, pandas will convert None -> np.nan by itself but
            # we run it for double columns to be extra sure. A pyspark
            # dataframe won't contain np.nan in a non-double column anyway.
            float_column_names = [name for name, dtype in df_spark.dtypes
                                  if dtype in PYSPARK_FLOAT_DTYPES]
            df_spark = df_spark.replace(np.nan, None, float_column_names)
        return cls(df_spark)

    @classmethod
    def from_rows(cls, spark, rows, columns=None,
                  convert_nan_to_null_in_non_float=True,
                  convert_nan_to_null_in_float=False):
        """
        Construct a FlickerDataFrame from a list of rows. Each row can be
        a dict or a tuple or something similar. This function first uses
        pandas.DataFrame.from_records(rows, columns=columns) to create a
        pandas DataFrame and then converts it to a FlickerDataFrame.

        Parameters
        ----------
        spark: pyspark.sql.SparkSession
            SparkSession object. This can be manually created by something
            like: `spark = SparkSession.builder.appName('app').getOrCreate()`
        rows: List[Any]
            A list of rows. For example, list of dicts.
        columns: List[str]
            Column names to use. If `rows` does not have names
            associated with them, this argument provides names for the
            columns. Otherwise this argument indicates the order of the columns
            in the result. See pandas.DataFrame.from_records for more
            information.
        convert_nan_to_null_in_non_float: bool
            If True (recommended), we convert np.nan (which has the
            type 'float') into None in the non-float columns. Unlike a pandas
            DataFrame, a pyspark DataFrame does not allow a np.nan in a
            non-float/non-double column. Note that any 'object' type column
            in `df` will be considered as a non-float column.
        convert_nan_to_null_in_float: bool
            If True, we convert np.nan (which has the type 'float') into None
            in all float/double columns. A pyspark DataFrame allows both
            np.nan and None to exist in a (nullable) float or double column.

        Returns
        -------
            FlickerDataFrame

        See Also
        --------
        FlickerDataFrame.from_records: alias of FlickerDataFrame.from_rows
        FlickerDataFrame.from_items: alias of FlickerDataFrame.from_rows
        FlickerDataFrame.from_pandas
        FlickerDataFrame.from_columns

        Examples
        --------
        >>> # Example 1
        >>> df = FlickerDataFrame.from_rows(spark, [(1, 'a'), (2, 'b')])
        >>> df
        FlickerDataFrame[0: bigint, 1: string]

        >>> df()
           0  1
        0  1  a
        1  2  b

        >>> # Example 2
        >>> df = FlickerDataFrame.from_rows(spark, [(1, 'a'), (2, 'b')],
                                            columns=['a', 'b'])
        >>> df
        FlickerDataFrame[a: bigint, b: string]

        >>> df()
           a  b
        0  1  a
        1  2  b

        >>> # Example 3
        >>> df = FlickerDataFrame.from_rows(spark, [{'a': 1, 'b': 4.5},
                                                    {'a': 2, 'b': 6.7}])
        >>> df
        FlickerDataFrame[a: bigint, b: double]

        >>> df()
           a    b
        0  1  4.5
        1  2  6.7
        """
        df = pd.DataFrame.from_records(rows, columns=columns)
        return cls.from_pandas(
            spark, df,
            convert_nan_to_null_in_non_float=convert_nan_to_null_in_non_float,
            convert_nan_to_null_in_float=convert_nan_to_null_in_float
        )

    from_records = from_rows
    from_items = from_rows

    @classmethod
    def from_dict(cls, spark, data, convert_nan_to_null_in_non_float=True,
                  convert_nan_to_null_in_float=False):
        """
        Construct a FlickerDataFrame from a dict. Each key in the dict must be
        a string and is used as the column name. The values are used as
        column contents. This function first uses
        pandas.DataFrame.from_dict(data) to create a
        pandas DataFrame and then converts it to a FlickerDataFrame.

        Parameters
        ----------
        spark: pyspark.sql.SparkSession
            SparkSession object. This can be manually created by something
            like: `spark = SparkSession.builder.appName('app').getOrCreate()`
        data: dict
            Of the form {column-name : column-contents}
        convert_nan_to_null_in_non_float: bool
            If True (recommended), we convert np.nan (which has the
            type 'float') into None in the non-float columns. Unlike a pandas
            DataFrame, a pyspark DataFrame does not allow a np.nan in a
            non-float/non-double column. Note that any 'object' type column
            in `df` will be considered as a non-float column.
        convert_nan_to_null_in_float: bool
            If True, we convert np.nan (which has the type 'float') into None
            in all float/double columns. A pyspark DataFrame allows both
            np.nan and None to exist in a (nullable) float or double column.

        Returns
        -------
            FlickerDataFrame

        See Also
        --------
        FlickerDataFrame.from_pandas
        FlickerDataFrame.from_columns

        Examples
        --------
        >>> df = FlickerDataFrame.from_dict(spark, {
            'a': [1, 2, 3, 4],
            'b': [1.2, None, np.nan, 4.5]
        })
        >>> df
        FlickerDataFrame[a: bigint, b: double]

        >>> df()
           a    b
        0  1  1.2
        1  2  NaN
        2  3  NaN
        3  4  4.5
        """
        # Note that converting to a pandas DataFrame would process and convert
        # some None(s) to NaNs even before we call the cls.from_pandas().
        df = pd.DataFrame.from_dict(data)
        return cls.from_pandas(
            spark, df,
            convert_nan_to_null_in_non_float=convert_nan_to_null_in_non_float,
            convert_nan_to_null_in_float=convert_nan_to_null_in_float
        )

    @classmethod
    def from_columns(cls, spark, data, columns=None,
                     convert_nan_to_null_in_non_float=True,
                     convert_nan_to_null_in_float=False):
        """
        Construct a FlickerDataFrame from a list of columns and optionally
        a list of column names. This function converts the input into a
        dict and then uses FlickerDataFrame.from_dict().

        Parameters
        ----------
        spark: pyspark.sql.SparkSession
            SparkSession object. This can be manually created by something
            like: `spark = SparkSession.builder.appName('app').getOrCreate()`
        data: List[Iterable]
            List of column contents.
        columns: List[str]
            Column names to use. If empty, column names are set to '0', '1',
            '2', ... str(len(data) - 1). Duplicate column names are not
            allowed.
        convert_nan_to_null_in_non_float: bool
            If True (recommended), we convert np.nan (which has the
            type 'float') into None in the non-float columns. Unlike a pandas
            DataFrame, a pyspark DataFrame does not allow a np.nan in a
            non-float/non-double column. Note that any 'object' type column
            in `df` will be considered as a non-float column.
        convert_nan_to_null_in_float: bool
            If True, we convert np.nan (which has the type 'float') into None
            in all float/double columns. A pyspark DataFrame allows both
            np.nan and None to exist in a (nullable) float or double column.

        Returns
        -------
            FlickerDataFrame

        See Also
        --------
        FlickerDataFrame.from_pandas
        FlickerDataFrame.from_dict

        Examples
        --------
        >>> data = [[1, 2, 3],
                    ['spark', 'flicker', 'pandas'],
                    [2.3, np.nan, 4.5]]
        >>> df = FlickerDataFrame.from_columns(spark, data,
                                               columns=['a', 'b', 'c'])
        >>> df
        FlickerDataFrame[a: bigint, b: string, c: double]

        >>> df()
           a        b    c
        0  1    spark  2.3
        1  2  flicker  NaN
        2  3   pandas  4.5
        """
        if columns is None:
            columns = [str(i) for i in list(range(len(data)))]
        if len(data) != len(columns):
            msg = 'len(data)={} and len(columns)={} do not match'
            raise ValueError(msg.format(len(data), len(columns)))
        data_dict = {name: value for name, value in zip(columns, data)}
        return cls.from_dict(
            spark, data_dict,
            convert_nan_to_null_in_non_float=convert_nan_to_null_in_non_float,
            convert_nan_to_null_in_float=convert_nan_to_null_in_float
        )

    @classmethod
    def from_shape(cls, spark, nrows, ncols, columns=None,
                   fill='zero'):
        """
        Construct a dataframe from only a shape and fill it with zeros or
        random floats. Note that this function first creates a pandas
        DataFrame and then converts it into a PySpark DataFrame. Don't use
        this function to create a dataframe that is too big to fit in memory.

        Parameters
        ----------
        spark: pyspark.sql.SparkSession
            SparkSession object. This can be manually created by something
            like: `spark = SparkSession.builder.appName('app').getOrCreate()`
        nrows: int
            Number of rows
        ncols: int
            Number of columns
        columns: List[str]
            List of column names. Duplicate column names are not allowed. If
            empty, column names are set to '0', '1', '2', ... str(ncols - 1).
        fill: str
            One of 'zero', 'rand', 'randn'.
            'zero': elements in the dataframe are zeros
            'rand': elements in the dataframe are uniformly distributed
            'randn': elements in the dataframe are normally distributed

        Returns
        -------
        FlickerDataFrame

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 10, 3,
                                             columns=['a', 'b', 'c'])
        >>> df
        FlickerDataFrame[a: double, b: double, c: double]

        >>> df()
            a    b    c
        0  0.0  0.0  0.0
        1  0.0  0.0  0.0
        2  0.0  0.0  0.0
        3  0.0  0.0  0.0
        4  0.0  0.0  0.0
        """
        if not isinstance(spark, SparkSession):
            msg = 'spark of type "{}" is not a SparkSession object'
            raise TypeError(msg.format(str(type(spark))))
        if fill == 'zero':
            data = list(np.zeros((nrows, ncols)))
        elif fill == 'rand':
            data = list(np.random.rand(nrows, ncols))
        elif fill == 'randn':
            data = list(np.random.randn(nrows, ncols))
        else:
            msg = 'fill="{}" is not supported'
            raise ValueError(msg.format(str(fill)))
        df = pd.DataFrame.from_records(data, columns=columns)
        return cls.from_pandas(spark, df)

    def __repr__(self):
        dtypes_str_list = ['{}: {}'.format(name, dtype)
                           for name, dtype in self._df.dtypes]
        dtypes_str = ', '.join(dtypes_str_list)
        return '{}[{}]'.format(self.__class__.__name__, dtypes_str)

    def __str__(self):
        return repr(self)

    # Added by me to augment pythonic functionality
    @property
    def nrows(self):
        """
        Returns the number of rows.

        Note that when called for the first time, this will require running
        .count() on the underlying pyspark DataFrame, which can take a long
        time.

        Returns
        -------
            int

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 10, 3, ['a', 'b', 'c'])
        >>> df
        FlickerDataFrame[a: double, b: double, c: double]

        >>> df.nrows
        10
        """
        if self._nrows is None:
            self._nrows = self._df.count()
        return self._nrows

    @property
    def ncols(self):
        """
        Returns the number of columns

        This should return the answer really fast.

        Returns
        -------
            int

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 10, 3, ['a', 'b', 'c'])
        >>> df
        FlickerDataFrame[a: double, b: double, c: double]

        >>> df.ncols
        3
        """
        if self._ncols is None:
            self._ncols = len(self._df.columns)
        return self._ncols

    def __contains__(self, name):
        return name in self._df.columns

    def __setitem__(self, name, value):
        """ This function makes objects of this class mutable. """
        if isinstance(value, Column):
            pass
        elif (value is None) or \
                isinstance(value, (int, float, six.string_types)):
            value = lit(value)
        else:
            msg = 'value of type "{}" is not supported'
            raise TypeError(msg.format(str(type(value))))

        if not isinstance(value, Column):
            msg = 'value cannot be converted to pyspark Column'
            raise ValueError(msg)
        self._reset(self._df.withColumn(name, value))

    def __delitem__(self, name):
        """ Delete a column from the dataframe in-place. This function makes
        objects of this class mutable.

        Parameters
        ----------
        name: str
            A column name that exists in the dataframe

        Returns
        -------
            None

        See Also
        --------
        FlickerDataFrame.drop

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 10, 3, ['a', 'b', 'c'])
        >>> df(2)
             a    b    c
        0  0.0  0.0  0.0
        1  0.0  0.0  0.0

        >>> del df['a']
        >>> df(2)
             b    c
        0  0.0  0.0
        1  0.0  0.0
        """
        if not isinstance(name, six.string_types):
            msg = 'type(name)="{}" is not a string'
            raise TypeError(msg.format(str(type(name))))
        if name not in self._df.columns:
            msg = 'column "{}" not found'
            raise KeyError(msg.format(name))
        self._reset(self._df.drop(name))

    def __call__(self, nrows=5, item=None):
        """Calling a FlickerDataFrame converts the underlying pyspark
        DataFrame into a pandas DataFrame and returns the result. This function
        is a shorthand for .select(item).limit(nrows).toPandas(). See
        Examples.

        Note that converting to pandas DataFrame may convert null/None to
        np.nan in float columns.

        Parameters
        ----------
        nrows: int or None
          Number of rows to select when before converting to pandas DataFrame.
          When nrows=None, we select all rows. Be careful when using
          nrows=None, because the pandas DataFrame may be too big to be stored
          in memory.
        item: str or list of str
          A single column name or a list of column names to select before
          converting to a pandas DataFrame.

        Returns
        -------
            pandas.DataFrame

        See Also
        --------
        FlickerDataFrame.head

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 10, 3, ['a', 'b', 'c'])
        >>> df
        FlickerDataFrame[a: double, b: double, c: double]
        >>> df()
             a    b    c
        0  0.0  0.0  0.0
        1  0.0  0.0  0.0
        2  0.0  0.0  0.0
        3  0.0  0.0  0.0
        4  0.0  0.0  0.0
        >>> df(2)
             a    b    c
        0  0.0  0.0  0.0
        1  0.0  0.0  0.0
        >>> df[['a']](nrows=2)
             a
        0  0.0
        1  0.0
        >>> df[['a', 'b']](nrows=2)
             a    b
        0  0.0  0.0
        1  0.0  0.0

        >>> df = FlickerDataFrame.from_shape(
                spark, 10, 3, columns=['a', 'b', 'c'], fill='randn')
        >>> df
        FlickerDataFrame[a: double, b: double, c: double]
        >>> df[df['a'] > 0](2)
                a         b         c
        0  0.975916 -0.977497 -0.033339
        1  1.471300 -0.744226 -1.810772
        """

        if item is None:
            item = list(self._df.columns)
        if isinstance(item, six.string_types):
            item = [item]
        if nrows is None:
            out = self._df[item].toPandas()
        else:
            out = self._df[item].limit(nrows).toPandas()
        return out

    # Added by me to augment Pandas-like API
    def rename(self, mapper_or_list):
        """
        Return a new FlickerDataFrame with column names renamed.
        This does not edit the dataframe in place.

        Parameters
        ----------
        mapper_or_list: List[str] or dict
            If a List[str], the len(mapper_or_list) must have the same
            number of elements as the number of columns in the dataframe.
            Duplicate column names are not allowed.
            If a dict, mapper_or_list must be of the form {old-name: new-name}.
            FlickerDataFrame cannot have duplicate column names.

        Returns
        -------
            FlickerDataFrame

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 10, 3,
                                             columns=['a', 'b', 'c'])

        >>> # Example 1
        >>> new_df = df.rename(['p', 'q', 'r'])
        >>> new_df(2)
             p    q    r
        0  0.0  0.0  0.0
        1  0.0  0.0  0.0

        >>> # Example 2
        >>> new_df = df.rename({'a': 'p'})
        >>> new_df(2)
             p    b    c
        0  0.0  0.0  0.0
        1  0.0  0.0  0.0
        """
        if isinstance(mapper_or_list, list):
            names = list(mapper_or_list)
            if len(names) != len(self._df.columns):
                msg = ('mapper_or_list is a list but len(mapper_or_list)={}'
                       ' does not match ncols={}')
                raise ValueError(msg.format(len(names), len(self._df.columns)))
            out = self._df.toDF(*names)
        elif isinstance(mapper_or_list, dict):
            mapper = dict(mapper_or_list)
            for old in mapper.keys():
                if old not in self._df.columns:
                    msg = 'column "{}" not found'
                    raise KeyError(msg.format(old))

            # Mild in-sufficient check to prevent duplicate names
            new_names = list(mapper.values())
            if len(set(new_names)) != len(new_names):
                msg = ('new names in mapper contains duplicate names'
                       ' which are not allowed')
                raise ValueError(msg)

            out = self._df
            for old, new in mapper.items():
                out = out.withColumnRenamed(old, new)
        else:
            msg = 'mapper_or_list of type "{}" is neither a list nor a dict'
            raise TypeError(msg.format(str(type(mapper_or_list))))
        return self.__class__(out)

    def value_counts(self, names, normalize=False, sort=True, ascending=False,
                     drop_null=False, nrows=None):
        """
        Returns a FlickerDataFrame that contains a frequency table of the
        selected column.

        Parameters
        ----------
        names: str or list of str
            Name of a column in the dataframe or a list of column names
        normalize: bool
            If True, the frequency counts are normalized by the number of
            rows in the entire dataframe (even when nrows is set to a non-None
            value).
        sort: bool
            If True (default), the output is sorted according to the
            `ascending` argument.
        ascending: bool
            Used only when `sort=True`. If True, the output is sorted in
            increasing order of frequency  counts. If False (default), the
            output is sorted in decreasing order of frequency counts.
        drop_null: bool
            If True, we discard all null values in the pyspark DataFrame.
        nrows: int or None:
            If None, all rows of the dataframe are used. If an int, we use
            that many rows to compute the output. Note that if nrows is more
            than the actual number of rows in the dataframe, all rows are used.

        Returns
        -------
            FlickerDataFrame

        Examples
        --------
        >>> data = {
            'a': [1, 1, 2, 3, 4, 1],
            'b': ['spark', 'flicker', None, None, 'spark', 'pandas']
        }
        >>> df = FlickerDataFrame.from_dict(spark, data)
        >>> df()
           a        b
        0  1    spark
        1  1  flicker
        2  2     None
        3  3     None
        4  4    spark

        >>> df.value_counts('a')(None)
           a  count
        0  1      3
        1  3      1
        2  2      1
        3  4      1

        >>> df.value_counts('b')(None)
                 b  count
        0     None      2
        1    spark      2
        2   pandas      1
        3  flicker      1

        >>> df.value_counts('b', normalize=True)(None)
                 b     count
        0     None  0.333333
        1    spark  0.333333
        2   pandas  0.166667
        3  flicker  0.166667

        >>> df.value_counts('b', drop_null=True)(None)
                 b  count
        0    spark      2
        1   pandas      1
        2  flicker      1
        """
        if isinstance(names, six.string_types):
            names = [names]
        for name in names:
            # FIXME: check that name is of type str here as well
            if name not in self._df.columns:
                msg = 'column "{}" not found'
                raise KeyError(msg.format(name))
        if 'count' in self._df.columns:
            msg = ('column "count" already exists in dataframe; '
                   'please rename it before calling value_counts()')
            raise ValueError(msg)

        out = self._df
        if drop_null:
            out = out[out[names].isNotNull()]
        if nrows is not None:
            out = out.limit(nrows)

        # .count() is lazy when called on  pyspark.sql.group.GroupedData
        # creates a column called "count"
        out = out.groupBy(names).count()
        if sort:
            out = out.orderBy('count', ascending=ascending)
        if normalize:
            # always normalize by number of rows in dataframe, not the value
            # used to limit. Don't count on out which has been filtered and
            # limited.
            den = float(self._df.count())
            out = out.withColumn('count', out['count'] / den)
        return self.__class__(out)

    def count_value(self, name, value):
        """ Count the number of times a value appears in a column.
            CAVEAT: spark ignores data types when doing this. See
            https://www.perfectlyrandom.org/bites/2020/03/19/spark-ignores-data-types-when-filtering-on-equality
            for more details. This is an issue with spark (and even SQL) not
            with flicker.
        """
        if name not in self._df.columns:
            msg = 'column "{}" not found'
            raise KeyError(msg.format(name))
        return self._df[self._df[name].isin([value])].count()

    def sort_values(self, by=None, ascending=True):
        if by is None:
            by = list(self._df.columns)
        if not isinstance(by, list):
            msg = 'type(by="{}" is not a list'
            raise TypeError(msg.format(str(type(by))))
        for name in by:
            if name not in self._df.columns:
                msg = 'column "{}" not found'
                raise KeyError(msg.format(name))
        return self.__class__(self._df.orderBy(*by, ascending=ascending))

    def isnan(self, name):
        if name not in self._df.columns:
            msg = 'column "{}" not found'
            raise KeyError(msg.format(name))
        return isnan(self._df[name])

    def isnull(self, name):
        if name not in self._df.columns:
            msg = 'column "{}" not found'
            raise KeyError(msg.format(name))
        return self._df[name].isNull()

    def max(self, name, ignore_nan=True):
        # Note that there is no need to ignore null(s) because pyspark
        # already does that for you.
        # Note that this is one of those places where NaN behaves differently
        # than null.
        if name not in self._df.columns:
            msg = 'column "{}" not found'
            raise KeyError(msg.format(name))
        df = self._df
        if ignore_nan:
            df = df[~isnan(df[name])]

        # Based on https://stackoverflow.com/questions/33224740/best-way-to-get-the-max-value-in-a-spark-dataframe-column
        return df.agg({name: 'max'}).collect()[0][0]

    def min(self, name, ignore_nan=True):
        # Note that there is no need to ignore null(s) because pyspark
        # already does that for you.
        # Note that this is one of those places where NaN behaves differently
        # than null.
        if name not in self._df.columns:
            msg = 'column "{}" not found'
            raise KeyError(msg.format(name))
        df = self._df
        if ignore_nan:
            df = df[~isnan(df[name])]

        # Based on https://stackoverflow.com/questions/33224740/best-way-to-get-the-max-value-in-a-spark-dataframe-column
        return df.agg({name: 'min'}).collect()[0][0]

    def rows_with_max(self, name, ignore_nan=True):
        # Note the plural. This is because a large number of rows can have
        # the max value.
        # Note that this is one of those places where NaN behaves differently
        # than null.
        max_value = self.max(name=name, ignore_nan=ignore_nan)
        out = self._df[self._df[name].isin([max_value])]
        return self.__class__(out)

    def rows_with_min(self, name, ignore_nan=True):
        # Note the plural. This is because a large number of rows can have
        # the min value.
        # Note that this is one of those places where NaN behaves differently
        # than null.
        min_value = self.min(name=name, ignore_nan=ignore_nan)
        out = self._df[self._df[name].isin([min_value])]
        return self.__class__(out)

    # Modified to provide a Pandas-like API
    @property
    def shape(self):
        """
        Returns a tuple (nrows, ncols).

        Note that when called for the first time, this will require running
        .count() on the underlying pyspark DataFrame, which can take a long
        time.

        Returns
        -------
            (int, int)

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 10, 3, ['a', 'b', 'c'])
        >>> df
        FlickerDataFrame[a: double, b: double, c: double]

        >>> df.shape
        (10, 3)
        """
        return self.nrows, self.ncols

    def head(self, nrows=5):
        """
        Return the first n rows as a Pandas DataFrame.

        Parameters
        ----------
        nrows: int or None
            Number of rows to use. If None, all rows are returned (which
            can be bigger than can fit in memory).

        Returns
        -------
            pandas.DataFrame

        See Also
        --------
        FlickerDataFrame.__call__
        FlickerDataFrame.head

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 10, 3, ['a', 'b', 'c'])
        >>> df.head(2)
             a    b    c
        0  0.0  0.0  0.0
        1  0.0  0.0  0.0
        """
        if nrows is None:
            out = self._df.toPandas()
        else:
            out = self._df.limit(nrows).toPandas()
        return out

    def tail(self, nrows=5):
        """
        Not implemented. Calling this would give you an error. Use .head()
        instead.
        """
        msg = '.tail() is not well-defined. Use .head() instead'
        raise NotImplementedError(msg)

    def first(self):
        """
        Return the first row as a Pandas DataFrame.

        Returns
        -------
            pandas.DataFrame

        See Also
        --------
        FlickerDataFrame.head
        FlickerDataFrame.__call__

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 10, 3, ['a', 'b', 'c'])
        >>> df.first()
             a    b    c
        0  0.0  0.0  0.0
        """
        return self.head(1)

    def last(self):
        """
        Not implemented. Calling this would give you an error. Use .first()
        instead.
        """
        msg = '.last() is not well-defined. Use .first() instead'
        raise NotImplementedError(msg)

    def show(self, nrows=5):
        """
        Print the first n rows of the dataframe. This does not return
        a string. This function simply calls pyspark.sql.DataFrame.show().

        Parameters
        ----------
        nrows: int
            Number of rows to print.

        Returns
        -------
            None

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 10, 3, ['a', 'b', 'c'])
        >>> df.show()
        +---+---+---+
        |  a|  b|  c|
        +---+---+---+
        |0.0|0.0|0.0|
        |0.0|0.0|0.0|
        |0.0|0.0|0.0|
        |0.0|0.0|0.0|
        |0.0|0.0|0.0|
        +---+---+---+
        """
        return self._df.show(nrows)

    def take(self, nrows=5):
        """
        Returns the first n rows as a list of Row objects. This function
        simply calls pyspark.sql.DataFrame.take().

        Parameters
        ----------
        nrows: int
            Number of rows to return

        Returns
        -------
            List[Row]

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 10, 3, ['a', 'b', 'c'])
        >>> df.take()
        [Row(a=0.0, b=0.0, c=0.0),
         Row(a=0.0, b=0.0, c=0.0),
         Row(a=0.0, b=0.0, c=0.0),
         Row(a=0.0, b=0.0, c=0.0),
         Row(a=0.0, b=0.0, c=0.0)]
        """
        return self._df.take(nrows)

    def describe(self, names=None):
        """
        Returns a pandas DataFrame containing basic statistics for numeric
        and string columns. Note that other dtypes don't appear in the
        output of this function.

        Parameters
        ----------
        names: List[str] or str or None
            If None, output contains all string and numeric columns.
            If str, it should be a valid column name.
            If List[str], each element should be a valid column name.
            Duplicate column names are not allowed.

        Returns
        -------
            pandas DataFrame

        Examples
        --------
        >>> data = [(1, 'spark', 2.4, {}), (2, 'flicker', np.nan, {'key': 1})]
        >>> column_names = ['a', 'b', 'c', 'd']
        >>> df = FlickerDataFrame.from_rows(spark, rows=data,
                                            columns=column_names)
        >>> df()
           a        b    c           d
        0  1    spark  2.4          {}
        1  2  flicker  NaN  {'key': 1}

        >>> df.describe()  # Note how column 'd' doesn't appear
                        a        b    c
        summary
        count    2.000000        2    2
        mean     1.500000     None  NaN
        stddev   0.707107     None  NaN
        min      1.000000  flicker  2.4
        max      2.000000    spark  NaN

        >>> df.describe(['a'])
                        a
        summary
        count    2.000000
        mean     1.500000
        stddev   0.707107
        min      1.000000
        max      2.000000

        >>> df.describe(['a', 'c'])
                    c         a
        summary
        count      2  2.000000
        mean     NaN  1.500000
        stddev   NaN  0.707107
        min      2.4  1.000000
        max      NaN  2.000000
        """
        if names is None:
            names = list(self._df.columns)
        elif isinstance(names, six.string_types):
            names = [names]
        else:
            names = list(set(names))
        for name in names:
            if name not in self._df.columns:
                msg = 'column "{}" not found'
                raise KeyError(msg.format(name))

        # This is always going to be a small dataframe of shape
        # (5, number of columns in self._df).
        out = self._df.describe(names).toPandas()

        # If self._df has a column "summary", then we'll have duplicates.
        # First column is always the true "summary" column. Make it an
        # index and remove it.
        out.index = out.iloc[:, 0]
        out = out.iloc[:, 1:]

        # Convert from string to numeric, if possible. Output of
        # pyspark.sql.DataFrame.describe() can contain string columns.
        # It appears that min/max of strings get returned.
        # Note that 'NaN' may sometimes be stored as a string.
        for column in out.columns:
            try:
                out[column] = pd.to_numeric(out[column])
            except ValueError:
                # This often happens for strings.
                pass
        return out

    def drop(self, names=[]):
        """
        Returns a new FlickerDataFrame object with the specified list of
        column names dropped. This function does not modify the dataframe
        in-place. If an empty list is provided, this is a no-op. Similar to
        pyspark.sql.DataFrame.drop, no error is raised if a non-existent
        column is passed.

        Note that this FlickerDataFrame.drop differs slightly from
        pyspark.sql.DataFrame.drop in how it treats input arguments.
        Flicker's `drop` accepts one argument that is a (possibly empty) list
        of column names while pyspark's `drop` accepts a variable number of
        arguments.

        Parameters
        ----------
        names: str or list of str
            A list of column names that need to be dropped. This list can be
            empty in which case no column is dropped. This list can contain a
            column name that does not exist in the dataframe. Duplicate values
            are also allowed.

        Returns
        -------
            FlickerDataFrame

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 10, 3, ['a', 'b', 'c'])
        >>> df
        FlickerDataFrame[a: double, b: double, c: double]

        >>> df.drop(['a'])
        FlickerDataFrame[b: double, c: double]

        >>> df.drop(['a', 'a'])
        FlickerDataFrame[b: double, c: double]

        >>> df.drop(['non-existent-column-name'])
        FlickerDataFrame[a: double, b: double, c: double]

        >>> df.drop(['a', 'non-existent-column-name'])
        FlickerDataFrame[b: double, c: double]

        >>> df.drop(['a', 'a', 'non-existent-column-name'])
        FlickerDataFrame[b: double, c: double]

        >>> df.drop()
        FlickerDataFrame[a: double, b: double, c: double]

        >>> df.drop([])
        FlickerDataFrame[a: double, b: double, c: double]
        """
        if isinstance(names, six.string_types):
            names = [names]
        for name in names:
            if not isinstance(name, six.string_types):
                msg = ('column names must be of type str but you provided '
                       'type = {}')
                msg = msg.format(str(type(name)))
                raise TypeError(msg)
            if name not in self._df.columns:
                msg = ('column name "{}" not found in the dataframe')
                msg = msg.format(name)
                raise KeyError(msg)
        return self.__class__(self._df.drop(*names))

    def join(self, other, on=None, how='inner',
             lsuffix=None, rsuffix=None, lprefix=None, rprefix=None):
        # Note that any None column value is not matched on. It's ignored.

        def _validate_column_name(name, names, dataframe):
            if not isinstance(name, six.string_types):
                msg = ('column names must be of type str but you provided '
                       'type = {}')
                msg = msg.format(str(type(name)))
                raise TypeError(msg)
            if name not in names:
                msg = ('column name "{}" not found in the {} dataframe')
                msg = msg.format(name, dataframe)
                raise KeyError(msg)

        def _validate_prefix_suffix(value, name):
            if (value is not None) and \
                    (not isinstance(value, six.string_types)):
                msg = ('{} must be None or of type str, you provided '
                       'type({})={}')
                msg = msg.format(name, name, str(type(value)))
                raise TypeError(msg)

        def _add_prefix_suffix(prefix, name, suffix):
            if prefix is None:
                prefix = ''
            if suffix is None:
                suffix = ''
            return prefix + name + suffix

        # Check other
        if not isinstance(other, FlickerDataFrame):
            msg = ('other must be of type FlickerDataFrame but you provided '
                   'type(FlickerDataFrame)={}')
            raise TypeError(msg.format(str(type(other))))

        # Check types of prefix and suffix arguments
        _validate_prefix_suffix(lsuffix, 'lsuffix')
        _validate_prefix_suffix(rsuffix, 'rsuffix')
        _validate_prefix_suffix(lprefix, 'lprefix')
        _validate_prefix_suffix(rprefix, 'rprefix')

        # Check `on`
        # Convert from string to list so we can handle it in a more
        # streamlined fashion.
        if isinstance(on, six.string_types):
            on = [on]

        # Convert list into a dict for streamlined handling
        if isinstance(on, (list, tuple)):
            on = {name: name for name in on}

        # Since we allow renaming of columns, we need to rename the keys in
        # `on` as well.
        if on is None:
            # We pass it into the pyspark.sql.DataFrame.join()
            pass
        elif isinstance(on, dict):
            # `on` must me a dict of the form
            # {'col_name_in_left': 'col_name_in_right'}
            for lname, rname in on.items():
                _validate_column_name(lname, self.columns, 'left (self)')
                _validate_column_name(rname, other.columns, 'right (other)')

            # Create and replace `on` with a new dict that has updated names
            on = {
                _add_prefix_suffix(lprefix, lname, lsuffix):
                    _add_prefix_suffix(rprefix, rname, rsuffix)
                for lname, rname in on.items()
            }
        else:
            msg = ('`on` must be a string, list of strings, or a dict; you '
                   'provided type(on)={}')
            raise TypeError(msg.format(str(type(on))))

        # Rename left (aka self)
        lnames_mapper = {
            name: _add_prefix_suffix(lprefix, name, lsuffix)
            for name in self.columns
        }
        left = self.rename(lnames_mapper)

        # Rename right (aka other)
        rnames_mapper = {
            name: _add_prefix_suffix(rprefix, name, rsuffix)
            for name in other.columns
        }
        right = other.rename(rnames_mapper)

        # If `on` is a dict, convert it into a condition because that's the
        # most general when you have different column names in left and right.
        if isinstance(on, dict):
            # Note that `on` can be an empty dict
            conditions = [left._df[lname] == right._df[rname]
                          for lname, rname in on.items()]
            if len(conditions) == 0:
                on = None
            else:
                on = conditions[0]
                for condition in conditions[1:]:
                    on = on & condition

        # Join the (possibly) renamed columns
        out = left._df.join(other=right._df, on=on, how=how)
        if len(set(out.columns)) != len(out.columns):
            msg = ('joined dataframe contains duplicated column names '
                   'which is not supported. Did you forget to provide '
                   'lprefix, rprefix, lsuffix, or rsuffix?')
            raise ValueError(msg)
        if isinstance(out, pyspark.sql.DataFrame):
            out = self.__class__(out)
        return out

    # Pass through functions
    @property
    def dtypes(self):
        """
        Returns list of (column name, dtype). This is simply a wrapper over
        pyspark.sql.DataFrame.dtypes.

        Returns
        -------
            List[(str, str)]

        Examples
        --------
        >>> df = FlickerDataFrame.from_dict(spark, {
            'a': [True, False, True],
            'b': [3.4, 6.7, 9.0],
            'c': ['spark', 'pandas', 'flicker']
        })

        >>> df
        FlickerDataFrame[a: double, b: double, c: string]

        >>> df()
               a    b        c
        0   True  3.4    spark
        1  False  6.7   pandas
        2   True  9.0  flicker

        >>> df.dtypes
        [('a', 'boolean'), ('b', 'double'), ('c', 'string')]

        >>> string_cols = [col for col, dtype in df.dtypes
                           if dtype == 'string']
        >>> string_cols
        ['c']
        """
        return self._df.dtypes

    @property
    def dtypes_only(self):
        return [dtype for _, dtype in self._df.dtypes]

    @property
    def names(self):
        """
        Returns list of all column names.

        Returns
        -------
            List[str]

        See Also
        --------
        FlickerDataFrame.columns: alias of FlickerDataFrame.names

        Examples
        --------
        >>> df = FlickerDataFrame.from_dict(spark, {
            'a': [True, False, True],
            'b': [3.4, 6.7, 9.0],
            'c': ['spark', 'pandas', 'flicker']
        })

        >>> df
        FlickerDataFrame[a: double, b: double, c: string]

        >>> df()
               a    b        c
        0   True  3.4    spark
        1  False  6.7   pandas
        2   True  9.0  flicker

        >>> df.namess
        ['a', 'b', 'c']

        >>> df.columns
        ['a', 'b', 'c']
        """
        return self._df.columns

    columns = names

    def __getattr__(self, name):
        return self._df.__getattr__(name)

    def __getitem__(self, item):
        out = self._df.__getitem__(item)
        if isinstance(out, pyspark.sql.DataFrame):
            out = self.__class__(out)
        return out

    def _ipython_key_completions_(self):
        """ Provide list of auto-completions for __getitem__ (not attributes)
            that is completed by df["c"+tab. Note that attribute completion
            is separate that happens automatically even when __dir__() is not
            explicitly defined.

            See https://ipython.readthedocs.io/en/stable/config/integrating.html

            This function enables auto-completion in both jupyter notebook and
            ipython terminal.
        """
        return list(self._df.columns)

    def limit(self, nrows=None):
        """
        Returns a new FlickerDataFrame with only the number of rows specified.

        Parameters
        ----------
        nrows: int or None
            If None, all rows are returned. All rows are returned if
            nrows is more than the number of rows in the dataframe.
            nrows=0 is a valid input which returns in a zero-row dataframe
            that still retains the column names.

        Returns
        -------
            FlickerDataFrame

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 10, 3, ['a', 'b', 'c'])
        >>> df
        FlickerDataFrame[a: double, b: double, c: double]

        >>> df.nrows
        10

        >>> df.limit(5)
        FlickerDataFrame[a: double, b: double, c: double]

        >>> df.limit(5).nrows
        5

        >>> df.limit().nrows
        10

        >>> df.limit() is df  # a new FlickerDataFrame is returned
        False

        >>> df.limit(0).nrows
        0

        >>> df.limit(0)()
        Empty DataFrame
        Columns: [a, b, c]
        Index: []

        >>> df.limit(100).nrows  # dataframe only has 10 rows
        10
        """
        if nrows is None:
            # Return a new FlickerDataFrame so id is different
            return self.__class__(self._df)
        else:
            return self.__class__(self._df.limit(nrows))

    # noinspection PyPep8Naming
    def toPandas(self):
        """
        Returns the contents as a pandas DataFrame. Note that resulting
        pandas DataFrame may be too big to store in memory. This function
        is simply a pass-through to pyspark.sql.DataFrame.toPandas.

        Returns
        -------
            pandas.DataFrame

        See Also
        --------
        FlickerDataFrame.to_pandas: alias of FlickerDataFrame.toPandas
        FlickerDataFrame.collect: if you want a list of Row objects

        Examples
        --------
        >>> df = FlickerDataFrame.from_dict(spark, {
            'a': [True, False, True],
            'b': [3.4, 6.7, 9.0],
            'c': ['spark', 'pandas', 'flicker']
        })

        >>> df
        FlickerDataFrame[a: double, b: double, c: string]

        >>> df(None)
               a    b        c
        0   True  3.4    spark
        1  False  6.7   pandas
        2   True  9.0  flicker

        >>> df.toPandas()
               a    b        c
        0   True  3.4    spark
        1  False  6.7   pandas
        2   True  9.0  flicker
        """
        return self._df.toPandas()

    to_pandas = toPandas

    def collect(self):
        """
        Returns all the rows as a list of Row objects. Note that this
        the returned list may be too big to store in memory. This function
        is simply a pass-through to pyspark.sql.DataFrame.collect.

        Returns
        -------
            List[Row]

        See Also
        --------
        FlickerDataFrame.toPandas: if you want a pandas DataFrame instead

        Examples
        --------
        >>> df = FlickerDataFrame.from_dict(spark, {
            'a': [True, False, True],
            'b': [3.4, 6.7, 9.0],
            'c': ['spark', 'pandas', 'flicker']
        })

        >>> df
        FlickerDataFrame[a: double, b: double, c: string]

        >>> df()
               a    b        c
        0   True  3.4    spark
        1  False  6.7   pandas
        2   True  9.0  flicker

        >>> df.collect()
        [Row(a=True, b=3.4, c='spark'),
         Row(a=False, b=6.7, c='pandas'),
         Row(a=True, b=9.0, c='flicker')]
        """
        return self._df.collect()

    def count(self):
        """
        Returns the number of rows. Use .nrows instead. This function exists
        only for convenience. It is a pass-through to
        pyspark.sql.DataFrame.count.

        Returns
        -------
            int

        See Also
        --------
        FlickerDataFrame.nrows: recommended over FlickerDataFrame.count
        FlickerDataFrame.shape

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 10, 3, ['a', 'b', 'c'])
        >>> df.count()
        10
        """
        return self._df.count()

    def distinct(self):
        """
        Returns a new FlickerDataFrame that contains only the distinct rows
        of the dataframe.

        Returns
        -------
            FlickerDataFrame

        Examples
        --------
        >>> df = FlickerDataFrame.from_dict(spark, {
            'a': [1, 2, 1],
            'b': [3.14, 2.89, 3.14],
            'c': ['spark', 'pandas', 'spark']
        })
        >>> df
        FlickerDataFrame[a: bigint, b: double, c: string]

        >>> df()
            a     b       c
        0  1  3.14   spark
        1  2  2.89  pandas
        2  1  3.14   spark

        >>> df.distinct()
        FlickerDataFrame[a: bigint, b: double, c: string]

        >>> df.distinct()()
           a     b       c
        0  1  3.14   spark
        1  2  2.89  pandas
        """
        out = self._df.distinct()
        if isinstance(out, pyspark.sql.DataFrame):
            out = self.__class__(out)
        return out
