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
from pyspark.sql.functions import lit


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
            for name in df.columns:
                df.loc[df[name].isnull(), name] = None

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
    def from_shape(cls, spark, nrows, ncols, columns=None):
        if not isinstance(spark, SparkSession):
            msg = 'spark of type "{}" is not a SparkSession object'
            raise TypeError(msg.format(str(type(spark))))
        zeros = list(np.zeros((nrows, ncols)))
        df = pd.DataFrame.from_records(zeros, columns=columns)
        return cls.from_pandas(spark, df)

    def __repr__(self):
        dtypes_str = ['{}: {}'.format(name, dtype)
                      for name, dtype in self._df.dtypes]
        return '{}{}'.format(self.__class__.__name__, dtypes_str)

    def __str__(self):
        return repr(self)

    # Added by me to augment pythonic functionality
    @property
    def nrows(self):
        if self._nrows is None:
            self._nrows = self._df.count()
        return self._nrows

    @property
    def ncols(self):
        if self._ncols is None:
            self._ncols = len(self._df.columns)
        return self._ncols

    def __contains__(self, name):
        return name in self._df.columns

    def __setitem__(self, name, value):
        if (value is None) or \
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
            raise KeyError(msg)
        self._reset(self._df.drop(name))

    def __call__(self, item=None, nrows=5):
        if item is None:
            item = list(self._df.columns)
        if isinstance(item, six.string_types):
            item = [item]
        return self._df[item].limit(nrows).toPandas()

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
            raise KeyError(msg)
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

    def sort_values(self, by=None, ascending=True):
        if by is None:
            by = list(self._df.columns)
        if not isinstance(by, list):
            msg = 'type(by="{}" is not a list'
            raise TypeError(msg.format(str(type(by))))
        for name in by:
            if name not in self._df.columns:
                msg = 'column "{}" not found'
                raise KeyError(msg)
        return self.__class__(self._df.orderBy(*by, ascending=ascending))

    # Modified to provide a Pandas-like API
    @property
    def shape(self):
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

    # Pass through functions
    @property
    def dtypes(self):
        return self._df.dtypes

    @property
    def columns(self):
        return self._df.columns

    def __getattr__(self, name):
        return self._df.__getattr__(name)

    def __getitem__(self, item):
        out = self._df.__getattr__(item)
        if isinstance(out, pyspark.sql.DataFrame):
            out = self.__class__(out)
        return out

    def limit(self, nrows):
        return self.__class__(self._df.limit(nrows))

    # noinspection PyPep8Naming
    def toPandas(self):
        return self._df.toPandas()

    def collect(self):
        return self._df.collect()

    def drop(self, *cols):
        return self.__class__(self._df.drop(*cols))
