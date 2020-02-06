from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from builtins import range

import six
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Column
from pyspark.sql.functions import lit, isnan


class FlickerDataFrame(object):
    def __init__(self, df):
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
        if not isinstance(df, pyspark.sql.DataFrame):
            msg = 'type(df)="{}" is not a pyspark DataFrame object'
            raise TypeError(msg.format(msg.format(str(type(df)))))
        self._df = df
        self._nrows = None
        self._ncols = None

    @classmethod
    def from_pandas(cls, spark, df,
                    convert_nan_to_null_in_non_double=True,
                    convert_nan_to_null_in_double=False):
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
        # call fail. This is why convert_nan_to_null_in_non_double=True,
        # by default.
        if convert_nan_to_null_in_non_double:
            # Note this will modify the pandas dataframe in-place.
            for name in df.columns:
                # If we simply perform, df.loc[df[name].isnull(), name] = None
                # we accidentally convert bool -> double even when we don't
                # have any np.nan or None(s).
                nulls_logical = df[name].isnull()
                if any(nulls_logical):
                    df.loc[nulls_logical, name] = None

        # We do this in pyspark because it's much easier than to do it
        # in pandas (which would've required a float -> object dtype
        # conversion).
        df_spark = spark.createDataFrame(df)
        if convert_nan_to_null_in_double:
            # We run it for double columns to be extra sure. A pyspark
            # dataframe won't contain np.nan in a non-double column anyway.
            double_columns = [
                name
                for name, dtype in df_spark.dtypes
                if dtype == 'double'
            ]
            df_spark = df_spark.replace(np.nan, None, double_columns)
        return cls(df_spark)

    @classmethod
    def from_rows(cls, spark, rows, columns=None,
                  convert_nan_to_null_in_non_double=True,
                  convert_nan_to_null_in_double=False):
        df = pd.DataFrame.from_records(rows, columns=columns)
        return cls.from_pandas(
            spark, df,
            convert_nan_to_null_in_non_double=convert_nan_to_null_in_non_double,
            convert_nan_to_null_in_double=convert_nan_to_null_in_double
        )

    from_records = from_rows
    from_items = from_rows

    @classmethod
    def from_dict(cls, spark, data, convert_nan_to_null_in_non_double=True,
                  convert_nan_to_null_in_double=False):
        df = pd.DataFrame.from_dict(data)
        return cls.from_pandas(
            spark, df,
            convert_nan_to_null_in_non_double=convert_nan_to_null_in_non_double,
            convert_nan_to_null_in_double=convert_nan_to_null_in_double
        )

    @classmethod
    def from_columns(cls, spark, data, columns=None,
                     convert_nan_to_null_in_non_double=True,
                     convert_nan_to_null_in_double=False):
        if columns is None:
            columns = [str(i) for i in list(range(len(data)))]
        if len(data) != len(columns):
            msg = 'len(data)={} and len(columns)={} do not match'
            raise ValueError(msg.format(len(data), len(columns)))
        data_dict = {name: value for name, value in zip(columns, data)}
        return cls.from_dict(
            spark, data_dict,
            convert_nan_to_null_in_non_double=convert_nan_to_null_in_non_double,
            convert_nan_to_null_in_double=convert_nan_to_null_in_double
        )

    @classmethod
    def from_shape(cls, spark, nrows, ncols, columns=None,
                   fill='zero'):
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

    def value_counts(self, name, normalize=False, sort=True, ascending=False,
                     drop_null=False, nrows=None):
        if name not in self._df.columns:
            msg = 'column "{}" not found'
            raise KeyError(msg.format(name))
        if 'count' in self._df.columns:
            msg = ('column "count" already exists in dataframe; '
                   'please rename it before calling value_counts()')
            raise ValueError(msg)

        out = self._df
        if drop_null:
            out = out[out[name].isNotNull()]
        if nrows is not None:
            out = out.limit(nrows)

        # .count() is lazy when called on  pyspark.sql.group.GroupedData
        # creates a column called "count"
        out = out.groupBy(name).count()
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
        if name not in self._df.columns:
            msg = 'column "{}" not found'
            raise KeyError(msg.format(name))
        # FIXME: the dtype is not respected here value = '1' or 1 returns the
        #  same Column object when name points to a sting-type column.
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
        return self._df.limit(nrows).toPandas()

    def tail(self, nrows=5):
        msg = '.tail() is not well-defined. Use .head() instead'
        raise NotImplementedError(msg)

    def first(self):
        return self.head(1)

    def last(self):
        msg = '.last() is not well-defined. Use .first() instead'
        raise NotImplementedError(msg)

    def show(self, nrows=5):
        return self._df.show(nrows)

    def take(self, nrows=5):
        return self._df.take(nrows)

    def describe(self, names=None):
        if names is None:
            names = list(self._df.columns)
        elif isinstance(names, six.string_types):
            names = [names]
        else:
            names = list(names)
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

        # Convert from string to numeric
        for column in out.columns:
            out[column] = pd.to_numeric(out[column])
        return out

    def drop(self, cols=[]):
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
        cols: list of str
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
        return self.__class__(self._df.drop(*cols))

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
    def columns(self):
        """
        Returns list of all column names.

        Returns
        -------
            List[str]

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

        >>> df.columns
        ['a', 'b', 'c']
        """
        return self._df.columns

    def __getattr__(self, name):
        return self._df.__getattr__(name)

    def __getitem__(self, item):
        out = self._df.__getitem__(item)
        if isinstance(out, pyspark.sql.DataFrame):
            out = self.__class__(out)
        return out

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
