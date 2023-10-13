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
from typing import Iterable, Any
from collections import OrderedDict

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Row, Column
from pyspark.sql.functions import lit

from .utils import get_length, is_nan_scalar
from .summary import get_columns_as_dict, get_summary


class FlickerDataFrame:
    _df: DataFrame
    _nrows: int | None
    _ncols: int | None
    _dtypes: OrderedDict | None

    def __init__(self, df: DataFrame):
        if not isinstance(df, DataFrame):
            raise TypeError(f'df must be of type pyspark.sql.DataFrame; you provided type(df)={type(df)}')
        if len(df.columns) != len(set(df.columns)):
            # pyspark.sql.DataFrame:
            # 1. does NOT enforce uniqueness names
            # 2. enforces str-type for column names
            raise ValueError(f'df contains duplicate columns names which is not supported. '
                             f'Rename the columns to be unique.')
        self._mutate(df)

    def _mutate(self, df: DataFrame):
        self._df = df
        self._nrows, self._ncols = None, len(df.columns)
        self._dtypes = OrderedDict()
        for name, dtype in self._df.dtypes:
            self._dtypes[name] = dtype

    def _check_names(self, names: Iterable[str]):
        for name in names:
            if name not in self._df.columns:
                raise KeyError(f'No column named {name}')

    def __repr__(self):
        return 'Flicker' + repr(self._df)

    def __str__(self):
        return repr(self)

    def __contains__(self, name: str) -> bool:
        # pyspark.sql.DataFrame enforces str-type for column names:
        # PySparkTypeError: [NOT_COLUMN_OR_STR] Argument `col` should be a Column or str, got int.
        return name in self._df.columns

    def __setitem__(self, name: str, value: Column | FlickerColumn | Any):
        if isinstance(value, Column):
            self._mutate(self._df.withColumn(name, value))
        elif isinstance(value, FlickerColumn):
            self._mutate(self._df.withColumn(name, value._column))
        else:
            self._mutate(self._df.withColumn(name, lit(value)))

    def __getitem__(self, item: tuple | slice | str | list | Column | FlickerColumn):
        """
        Parameters
        ----------
        item : tuple | slice | str | list | Column | FlickerColumn
            The index value to retrieve from the FlickerDataFrame object.

        Returns
        -------
        FlickerColumn or FlickerDataFrame
            If the index value is a string, returns a FlickerColumn object containing the column specified by the string.
            If the index value is a Column object, returns a new FlickerDataFrame object with only the specified column.
            If the index value is a FlickerColumn object, returns a new FlickerDataFrame object with only the column of the
            FlickerColumn object.
            If the index value is a slice object, returns a new FlickerDataFrame object with the columns specified by the slice.
            If the index value is a tuple of two slices, returns a new FlickerDataFrame object with the columns specified by the
            second slice, limited by the stop value of the first slice.
            If the index value is an iterable, returns a new FlickerDataFrame object with the columns specified by the elements
            of the iterable.

        Raises
        ------
        KeyError
            If the index value is not a supported index type.
        """
        # FIXME: should I support df[3] ?
        if isinstance(item, str):
            return FlickerColumn(self._df, self._df[item])
        if isinstance(item, Column):
            return self.__class__(self._df[item])
        if isinstance(item, FlickerColumn):
            return self.__class__(self._df[item._column])
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self._df.columns))
            names = [self._df.columns[i] for i in range(start, stop, step)]
            return self[names]
        if isinstance(item, tuple) and (len(item) == 2) and isinstance(item[0], slice) and isinstance(item[1], slice):
            first, second = item
            if (first.start is not None) or (first.step is not None):
                raise KeyError(f'When using [first_slice, second_slice] indexing, first_slice can only be of the '
                               f'form `:` or `:n`')
            if first.stop is None:
                return self[second]
            else:
                return self.__class__(self._df.limit(first.stop))[second]
        if isinstance(item, Iterable):
            names_or_columns = []
            for element in item:
                if isinstance(element, (str, Column)):
                    names_or_columns.append(element)
                elif isinstance(element, FlickerColumn):
                    names_or_columns.append(element._column)
                else:
                    raise KeyError(f'Unsupported type in list-style indexing {element}={type(element)}')
            return self.__class__(self._df[names_or_columns])
        raise KeyError(f'{item} is not a supported index type')

    def __delitem__(self, name: str):
        self._check_names([name])
        self._mutate(self._df.drop(name))

    def _ipython_key_completions_(self):
        """
        Provide list of auto-completions for __getitem__ (not attributes)
        that is completed by df["c"+tab. Note that attribute completion
        is separate that happens automatically even when __dir__() is not
        explicitly defined.

        See https://ipython.readthedocs.io/en/stable/config/integrating.html

        This function enables auto-completion in both jupyter notebook and
        ipython terminal.
        """
        return self._df.columns

    def __call__(self, n: int | None = 5, use_pandas_dtypes: bool = False) -> pd.DataFrame:
        if n is None:
            n = self._df.count()
        if use_pandas_dtypes:
            return self._df.limit(n).toPandas()
        else:
            data = get_columns_as_dict(self._df, n)
            return pd.DataFrame.from_dict(data, dtype=object)[self._df.columns]

    def show(self, n: int | None = 5, truncate: bool = True, vertical: bool = False) -> None:
        self._df.show(n=n, truncate=truncate, vertical=vertical)

    @property
    def nrows(self) -> int:
        if self._nrows is None:
            self._nrows = self._df.count()
        return self._nrows

    @property
    def ncols(self) -> int:
        return self._ncols

    @property
    def shape(self) -> tuple[int, int]:
        return self.nrows, self.ncols

    @property
    def names(self) -> list[str]:
        return self._df.columns

    @property
    def dtypes(self) -> OrderedDict:
        return self._dtypes

    @classmethod
    def from_shape(cls, spark: SparkSession, nrows: int, ncols: int, names: list[str] | None = None,
                   fill='zero') -> FlickerDataFrame:
        """
        Parameters
        ----------
        spark : SparkSession
            The Spark session used for creating the DataFrame.
        nrows : int
            The number of rows in the DataFrame.
        ncols : int
            The number of columns in the DataFrame.
        names : list[str] | None, optional
            The names of the columns in the DataFrame. If not provided, column names will be generated as '0', '1', '2', etc.
        fill : str, optional
            The value used for filling the DataFrame. Default is 'zero'.
            Accepted values are: 'zero', 'one', 'rand', 'randn'.

        Returns
        -------
        FlickerDataFrame
            A new instance of the FlickerDataFrame class created from the given shape and parameters.
        """
        if names is None:
            names = [f'{i}' for i in range(ncols)]
        if fill == 'zero':
            data = np.zeros((nrows, ncols), dtype=int)
        elif fill == 'one':
            data = np.ones((nrows, ncols), dtype=int)
        elif fill == 'rand':
            data = np.random.rand(nrows, ncols)
        elif fill == 'randn':
            data = np.random.randn(nrows, ncols)
        else:
            raise ValueError(f'fill={fill} is not supported')
        return cls(spark.createDataFrame(data=data, schema=names))

    @classmethod
    def from_rows(cls, spark: SparkSession, rows: Iterable[Iterable],
                  names: list[str] | None = None) -> FlickerDataFrame:
        """
        Parameters
        ----------
        spark : SparkSession
            The SparkSession object used to create the DataFrame.
        rows : Iterable[Iterable]
            An iterable of iterables representing the rows of data.
        names : list[str] | None, optional
            A list of column names for the DataFrame. If not provided, column names will be generated as '0', '1', '2', etc.

        Returns
        -------
        FlickerDataFrame
            A FlickerDataFrame object created from the given rows and column names.

        Raises
        ------
        ValueError
            If the rows contain different number of columns.

        Examples
        --------
        # Example 1: Creating a FlickerDataFrame from rows
        >>> spark = SparkSession.builder.getOrCreate()
        >>> rows = [[1, 'Alice', 25], [2, 'Bob', 30], [3, 'Charlie', 35]]
        >>> names = ['id', 'name', 'age']
        >>> df = FlickerDataFrame.from_rows(spark, rows, names)
        >>> df.show()
        +---+-------+---+
        | id|   name|age|
        +---+-------+---+
        |  1|  Alice| 25|
        |  2|    Bob| 30|
        |  3|Charlie| 35|
        +---+-------+---+

        # Example 2: Creating a FlickerDataFrame without specifying column names
        >>> spark = SparkSession.builder.getOrCreate()
        >>> rows = [[1, 'Alice', 25], [2, 'Bob', 30], [3, 'Charlie', 35]]
        >>> df = FlickerDataFrame.from_rows(spark, rows)
        >>> df.show()
        +---+-------+---+
        |  0|      1|  2|
        +---+-------+---+
        |  1|  Alice| 25|
        |  2|    Bob| 30|
        |  3|Charlie| 35|
        +---+-------+---+
        """
        data = [
            [None if is_nan_scalar(element) else element for element in row]
            for row in rows
        ]
        if names is None:
            maybe_ncols = set([len(row) for row in data])  # we have converted to list[list]
            if len(maybe_ncols) > 1:
                raise ValueError(f'records contain different number of columns: {maybe_ncols}')
            ncols = maybe_ncols.pop()
            names = [f'{i}' for i in range(ncols)]
        return cls(spark.createDataFrame(data=data, schema=names))

    @classmethod
    def from_columns(cls, spark: SparkSession, columns: Iterable[Iterable],
                     names: list[str] | None = None) -> FlickerDataFrame:
        ncols = get_length(columns)
        maybe_nrows = set([get_length(column) for column in columns])
        if len(maybe_nrows) > 1:
            raise ValueError(f'columns contain different number of rows: {maybe_nrows}')
        nrows = maybe_nrows.pop()
        data = [[None] * ncols for _ in range(nrows)]  # Python containers: must use "for" outer loop
        for j, column in enumerate(columns):
            for i, element in enumerate(column):
                data[i][j] = None if is_nan_scalar(element) else element
        if names is None:
            names = [f'{i}' for i in range(ncols)]
        return cls(spark.createDataFrame(data=data, schema=names))

    @classmethod
    def from_records(cls, spark: SparkSession, records: Iterable[dict]) -> FlickerDataFrame:
        for record in records:
            for name, element in record.items():
                record[name] = None if is_nan_scalar(element) else element
        return cls(spark.createDataFrame(data=records))

    @classmethod
    def from_dict(cls, spark: SparkSession, data: dict) -> FlickerDataFrame:
        names = list(data.keys())
        ncols = len(names)
        maybe_nrows = set([len(column) for column in data.values()])
        if len(maybe_nrows) > 1:
            raise ValueError(f'columns contain different number of rows: {maybe_nrows}')
        nrows = maybe_nrows.pop()
        rows = [[None] * ncols for _ in range(nrows)]  # Python containers: must use "for" outer loop
        for j, column in enumerate(data.values()):
            for i, element in enumerate(column):
                rows[i][j] = None if is_nan_scalar(element) else element
        return cls(spark.createDataFrame(data=rows, schema=names))

    def to_dict(self, n: int | None = 5) -> dict:
        return get_columns_as_dict(self._df, n)

    @classmethod
    def from_pandas(cls, spark: SparkSession, df: pd.DataFrame) -> FlickerDataFrame:
        return cls(spark.createDataFrame(data=df))

    def to_pandas(self) -> pd.DataFrame:
        return self._df.toPandas()

    def head(self, n: int | None = 5) -> FlickerDataFrame:
        if n is None:
            df = self._df
        else:
            df = self._df.limit(n)
        return self.__class__(df)

    def take(self, n: int | None = 5, convert_to_dict: bool = True) -> list[dict | Row]:
        if n is None:
            n = self._df.count()
        if convert_to_dict:
            return [row.asDict(recursive=True) for row in self._df.take(n)]
        else:
            return self._df.take(n)

    def drop(self, names: list[str]) -> FlickerDataFrame:
        self._check_names(names)
        return self.__class__(self._df.drop(*names))

    def rename(self, from_to_mapper: dict[str, str]) -> FlickerDataFrame:
        self._check_names(from_to_mapper.keys())
        return self.__class__(self._df.withColumnsRenamed(from_to_mapper))

    def sort(self, names: list[str], ascending: bool = True) -> FlickerDataFrame:
        self._check_names(names)
        return self.__class__(self._df.sort(*names, ascending=ascending))

    def unique(self) -> FlickerDataFrame:
        return self.__class__(self._df.distinct())

    def describe(self) -> pd.DataFrame:
        return get_summary(self._df)

    def concat(self, other: FlickerDataFrame | DataFrame, ignore_names: bool = False) -> FlickerDataFrame:
        if isinstance(other, FlickerDataFrame):
            other = other._df
        if ignore_names:
            return self.__class__(self._df.union(other))
        else:
            if set(self._df.columns) != set(other.columns):
                raise KeyError(f'Dataframes have different sets of column names. Cannot concat when ignore_names=True')
            return self.__class__(self._df.union(other[self._df.columns]))

    def merge(self, right: FlickerDataFrame | DataFrame, on: Iterable[str], how: str = 'inner',
              lprefix: str = '', lsuffix: str = '_l', rprefix: str = '', rsuffix: str = '_r') -> FlickerDataFrame:
        # All names in `on` must exist in both left and right dataframes
        if isinstance(on, dict):
            raise TypeError(f'`.merge()` requires that `on` be Iterable[str] but `on` is a dict. '
                            f'Did you mean `.join()` instead?')
        if isinstance(on, str):
            raise TypeError(f'`.merge()` requires that `on` be Iterable[str] but `on` is a `str`, on={on}. '
                            f'Did you want to wrap `on` into a list?')
        on = list(set(on))
        if len(on) == 0:
            raise ValueError(f'`on` cannot be an empty Iterable[str].')
        for i, name in enumerate(on):
            if not isinstance(name, str):
                raise TypeError(f'`on` must be Iterable[str] but you provided type(on[{i}])={type(name)}')
        self._check_names(on)
        if isinstance(right, DataFrame):
            right = FlickerDataFrame(right)
        right._check_names(on)

        # Prefixes and suffixes are not needed for the columns specified in `on` because `on` columns will not
        # be duplicated. But, the `join` operation may result in duplicate columns. We only need to rename those
        # columns.
        left, right = self._df, right._df
        names = set(left.columns).difference(on).intersection(set(right.columns).difference(on))
        lnames = {name: f'{lprefix}{name}{lsuffix}' for name in names}
        rnames = {name: f'{rprefix}{name}{rsuffix}' for name in names}

        # Note that renaming will result in unique names only if values of lprefix, rprefix, lsuffix, rsuffix
        # are chosen properly. It's possible that an `lname` already exists in the dataframe.
        # There are lots of ways to be wrong. We cannot check for every single possibility.
        left, right = left.withColumnsRenamed(lnames), right.withColumnsRenamed(rnames)

        # Typically, these errors are not raised because the above lines fail within Spark. But, we'll still
        # keep these lines here just in case spark's behavior changes.
        if len(left.columns) != len(set(left.columns)):
            raise KeyError(f'After renaming, left dataframe contains duplicate column names: {left.columns}. '
                           f'Choose lprefix, lsuffix carefully.')  # pragma: no cover
        if len(right.columns) != len(set(right.columns)):
            raise KeyError(f'After renaming, right dataframe contains duplicate column names: {right.columns}. '
                           f'Choose rprefix, rsuffix carefully.')  # pragma: no cover

        # Perform join and note that column names cannot be duplicate in the joined result
        return self.__class__(left.join(right, on=on, how=how))

    def join(self, right: FlickerDataFrame | DataFrame,
             on: dict[str, str], how: str = 'inner',
             lprefix: str = '', lsuffix: str = '_l', rprefix: str = '', rsuffix: str = '_r') -> FlickerDataFrame:
        # on = map of {left_name: right_name}
        if not isinstance(on, dict):
            raise TypeError(f'`.join()` requires that `on` be dict[str, str] but `on` is a {type(on)}. '
                            f'Did you mean `.merge()` instead?')
        if len(on) == 0:
            raise ValueError(f'`on` cannot be an empty dictionary')
        for lname, rname in on.items():
            if not isinstance(lname, str):
                raise TypeError(f'`.join()` requires that `on` be dict[str, str] `on` has '
                                f'a key={lname} of {type(lname)}.')
            if not isinstance(rname, str):
                raise TypeError(f'`.join()` requires that `on` be dict[str, str] `on` has '
                                f'a value=on[{repr(lname)}]={rname} of {type(rname)}.')
        self._check_names(on.keys())
        if isinstance(right, DataFrame):
            right = FlickerDataFrame(right)
        right._check_names(on.values())

        # In this method, a column named 'col' that exists in both left and right dataframes will not be
        # merged into one 'col' column in the result. Instead, both left-'col' and right-'col' will be preserved.
        # This means that there is a lot of potential for duplicate names.
        left, right = self._df, right._df
        names = set(left.columns).intersection(set(right.columns))
        lnames = {name: f'{lprefix}{name}{lsuffix}' for name in names}
        rnames = {name: f'{rprefix}{name}{rsuffix}' for name in names}

        # Note that renaming will result in unique names only if values of lprefix, rprefix, lsuffix, rsuffix
        # are chosen properly. It's possible that an `lname` already exists in the dataframe.
        # There are lots of ways to be wrong. We cannot check for every single possibility.
        left, right = left.withColumnsRenamed(lnames), right.withColumnsRenamed(rnames)

        # Typically, these errors are not raised because the above lines fail within Spark. But, we'll still
        # keep these lines here just in case spark's behavior changes.
        if len(left.columns) != len(set(left.columns)):
            raise KeyError(f'After renaming, left dataframe contains duplicate column names: {left.columns}. '
                           f'Choose lprefix, lsuffix carefully.')  # pragma: no cover
        if len(right.columns) != len(set(right.columns)):
            raise KeyError(f'After renaming, right dataframe contains duplicate column names: {right.columns}. '
                           f'Choose rprefix, rsuffix carefully.')  # pragma: no cover

        # Note that since we already checked for duplicates above, it's impossible that `new_on` contains
        # duplicate lnames or duplicate rnames.
        new_on = {}
        for lname, rname in on.items():
            if lname in lnames:
                lname = lnames[lname]
            if rname in rnames:
                rname = rnames[rname]
            new_on[lname] = rname
        if len(new_on) < len(on):
            raise NotImplementedError(f'This is a bug. Please report it.')
        conditions = [left[lname] == right[rname] for lname, rname in new_on.items()]

        # Perform join and note that column names cannot be duplicate in the joined result
        return self.__class__(left.join(right, on=conditions, how=how))

    def groupby(self, names: list[str]) -> FlickerGroupedData:
        return FlickerGroupedData(self._df, self._df.groupBy(*names))


# Import here to avoid circular imports
# https://github.com/ankur-gupta/rain/tree/v1#circular-imports-or-dependencies
from .column import FlickerColumn
from .group import FlickerGroupedData
