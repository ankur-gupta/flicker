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
from typing import Iterable, Callable
from pyspark.sql import DataFrame, Column
from pyspark.sql.functions import isnan
import pandas as pd
from .variables import PYTHON_TO_SPARK_DTYPES, PYSPARK_FLOAT_DTYPES
from .summary import (get_column_mean, get_column_stddev, get_column_min, get_column_max, get_timestamp_column_stddev,
                      get_timestamp_column_mean, get_summary, get_columns_as_dict, get_boolean_column_mean,
                      get_boolean_column_stddev)


class FlickerColumn:
    _df: DataFrame
    _column: Column
    _dtype: str

    def __init__(self, df: DataFrame, column: Column):
        if not isinstance(df, DataFrame):
            raise TypeError(f'df must be of type pyspark.sql.DataFrame; you provided type(df)={type(df)}')
        if not isinstance(column, Column):
            raise TypeError(f'column must be of type pyspark.sql.Column; you provided type(column)={type(column)}')
        if len(df.columns) != len(set(df.columns)):
            # pyspark.sql.DataFrame:
            # 1. does NOT enforce uniqueness names
            # 2. enforces str-type for column names
            raise ValueError(f'df contains duplicate columns names which is not supported. '
                             f'Rename the columns to be unique.')
        self._df = df
        self._column = column
        self._dtype = self._df[[self._column]].dtypes[0][1]

    @property
    def dtype(self):
        return self._dtype

    def _ensure_boolean(self):
        if self._dtype != 'boolean':
            raise TypeError(f'Column dtype must be dtype=boolean; you provided dtype={self._dtype}')

    def _ensure_float(self):
        if self._dtype not in PYSPARK_FLOAT_DTYPES:
            raise TypeError(f'Column dtype must be one of {PYSPARK_FLOAT_DTYPES}; you provided dtype={self._dtype}')

    def __repr__(self):
        return 'Flicker' + repr(self._column)

    def __str__(self):
        return repr(self)

    def __bool__(self) -> FlickerColumn:
        # self._column.__bool__() fails with this error. We don't intercept this error.
        # ValueError: Cannot convert column into bool: please use '&' for 'and', '|' for 'or', '~' for 'not' when
        # building DataFrame boolean expressions.
        return self.__class__(self._df, self._column.__bool__())

    def __neg__(self) -> FlickerColumn:
        return self.__class__(self._df, self._column.__neg__())

    def __invert__(self) -> FlickerColumn:
        # PySpark sucks. For a pyspark.sql.DataFrame `df` and a 'double' column 'a':
        # `~df['a']` does not fail
        # `df['b'] = ~df['a']` fails
        # We'd rather fail early in Flicker.
        self._ensure_boolean()
        return self.__class__(self._df, self._column.__invert__())

    # Binary operations

    def __eq__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__eq__(other))

    def __ne__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__ne__(other))

    def __le__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__le__(other))

    def __lt__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__lt__(other))

    def __ge__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__ge__(other))

    def __gt__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__gt__(other))

    def __and__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__and__(other))

    def __rand__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__rand__(other))

    def __or__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__or__(other))

    def __ror__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__ror__(other))

    def __add__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__add__(other))

    def __radd__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__radd__(other))

    def __sub__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__sub__(other))

    def __rsub__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            # This is unreachable because python will always call __sub__ first and manually calling __rsub__
            # encounters an error in pyspark.sql.Column.__rsub__.
            other = other._column  # pragma: no cover
        return self.__class__(self._df, self._column.__rsub__(other))

    def __mul__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__mul__(other))

    def __rmul__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__rmul__(other))

    def __div__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__div__(other))

    def __rdiv__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            # This is unreachable because python does not honor __rdiv__ anymore and manually calling __rdiv__
            # encounters an error in pyspark.sql.Column.__rdiv__.
            other = other._column  # pragma: no cover
        return self.__class__(self._df, self._column.__rdiv__(other))

    def __truediv__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__truediv__(other))

    def __rtruediv__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            # This is unreachable because python will always call __truediv__ first and manually calling __rtruediv__
            # encounters an error in pyspark.sql.Column.__rtruediv__.
            other = other._column  # pragma: no cover
        return self.__class__(self._df, self._column.__rtruediv__(other))

    def __pow__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__pow__(other))

    def __rpow__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__rpow__(other))

    def __mod__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            other = other._column
        return self.__class__(self._df, self._column.__mod__(other))

    def __rmod__(self, other: FlickerColumn | Column) -> FlickerColumn:
        if isinstance(other, FlickerColumn):
            # This is unreachable because python will always call __mod__ first and manually calling __rmod__
            # encounters an error in pyspark.sql.Column.__rmod__.
            other = other._column  # pragma: no cover
        return self.__class__(self._df, self._column.__rmod__(other))

    def __call__(self, n: int | None = 5, use_pandas_dtypes: bool = False) -> pd.DataFrame:
        df = self._df[[self._column]]
        if n is None:
            n = df.count()
        if use_pandas_dtypes:
            return df.limit(n).toPandas()[df.columns[0]]
        else:
            data = get_columns_as_dict(df, n)
            return pd.DataFrame.from_dict(data, dtype=object)[df.columns[0]]

    def take(self, n: int | None = 5) -> list:
        if n is None:
            n = self._df.count()
        return [list(row.asDict(recursive=True).values())[0] for row in self._df[[self._column]].take(n)]

    def astype(self, type_: type | str) -> FlickerColumn:
        if isinstance(type_, type):
            if type_ not in PYTHON_TO_SPARK_DTYPES:
                raise ValueError(f'Unsupported value {type_}')
            type_ = PYTHON_TO_SPARK_DTYPES[type_]
        return self.__class__(self._df, self._column.astype(type_))

    def isin(self, values: Iterable) -> FlickerColumn:
        return self.__class__(self._df, self._column.isin(values))

    def is_nan(self) -> FlickerColumn:
        return self.__class__(self._df, isnan(self._column))

    def is_null(self) -> FlickerColumn:
        return self.__class__(self._df, self._column.isNull())

    def is_not_null(self) -> FlickerColumn:
        return self.__class__(self._df, self._column.isNotNull())

    def any(self) -> bool:
        self._ensure_boolean()
        return self._df[self._column.isin([True])].count() > 0

    def all(self, ignore_null: bool = False) -> bool:
        # When all values are null and:
        # 1. dtype='boolean', ignore_null=True => return True
        # 2. dtype='boolean', ignore_null=False => return False
        # 3. dtype='void', ignore_null=True => raises TypeError
        # 4. dtype='void', ignore_null=False => raises TypeError
        self._ensure_boolean()

        # Since the dtype is boolean, we can run a collect query because there will at most be 3 rows, one each for
        # {True, False, None}. But there could be fewer rows.
        elements = set([
            list(row.asDict().values())[0]
            for row in self._df[[self._column]].distinct().collect()
        ])
        if ignore_null:
            elements = elements.difference({None})
        return (not elements) or (elements == {True})

    def _get_non_nan_dataframe(self, ignore_nan: bool) -> DataFrame:
        # Nulls are automatically ignored
        if ignore_nan and (self._dtype in PYSPARK_FLOAT_DTYPES):
            df = self._df[~isnan(self._column)][[self._column]]
        else:
            df = self._df[[self._column]]
        return df

    def min(self, ignore_nan: bool = True):
        # This is not the best way because it fails often with different data types.
        # https://stackoverflow.com/questions/33224740/best-way-to-get-the-max-value-in-a-spark-dataframe-column
        # return df.agg({df.columns[0]: 'max'}).collect()[0][0]
        df = self._get_non_nan_dataframe(ignore_nan)
        return get_column_min(df, df.columns[0])

    def max(self, ignore_nan: bool = True):
        # This is not the best way because it fails often with different data types.
        # https://stackoverflow.com/questions/33224740/best-way-to-get-the-max-value-in-a-spark-dataframe-column
        # return df.agg({df.columns[0]: 'max'}).collect()[0][0]
        df = self._get_non_nan_dataframe(ignore_nan)
        return get_column_max(df, df.columns[0])

    def mean(self, ignore_nan: bool = True):
        df = self._get_non_nan_dataframe(ignore_nan)
        if self._dtype == 'timestamp':
            return get_timestamp_column_mean(df, df.columns[0])
        elif self._dtype == 'boolean':
            return get_boolean_column_mean(df, df.columns[0])
        else:
            return get_column_mean(df, df.columns[0])

    def stddev(self, ignore_nan: bool = True):
        df = self._get_non_nan_dataframe(ignore_nan)
        if self._dtype == 'timestamp':
            return get_timestamp_column_stddev(df, df.columns[0])
        elif self._dtype == 'boolean':
            return get_boolean_column_stddev(df, df.columns[0])
        else:
            return get_column_stddev(df, df.columns[0])

    def value_counts(self, sort: bool = True, ascending: bool = False, drop_null: bool = False, normalize: bool = False,
                     n: int | None = None) -> FlickerDataFrame:
        if drop_null:
            df = self._df[self._column.isNotNull()][[self._column]]
        else:
            df = self._df[[self._column]]
        if df.columns[0] == 'count':
            raise KeyError(f'value_counts() cannot be run on this {self.__class__.__name__} object because it already '
                           f'has a column named "counts" because spark creates a new column called "count". Rename the'
                           f' column to something else before calling value_counts().')
        if n is not None:
            df = df.limit(n)

        counts = df.groupBy(*df.columns).count()
        if sort:
            counts = counts.orderBy('count', ascending=ascending)
        if normalize:
            # We always normalize by total number of rows in the dataframe and not by the number of rows used to limit.
            den = float(self._df.count())
            counts = counts.withColumn('count', counts['count'] / den)
        return FlickerDataFrame(counts)

    def describe(self) -> pd.DataFrame:
        return get_summary(self._df[[self._column]]).iloc[:, 0]

    def apply(self, udf: Callable) -> FlickerColumn:
        column = udf(self._column)
        df = self._df[[self._column, column]]
        return FlickerColumn(df, column)


# Import here to avoid circular imports
# https://github.com/ankur-gupta/rain/tree/v1#circular-imports-or-dependencies
from .dataframe import FlickerDataFrame
