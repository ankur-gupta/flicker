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
    """ ``FlickerDataFrame`` is a wrapper over ``pyspark.sql.DataFrame``. ``FlickerDataFrame`` provides a modern,
    clean, intuitive, pythonic, polars-like API over a ``pyspark`` backend.
    """
    _df: DataFrame
    _nrows: int | None
    _ncols: int | None
    _dtypes: OrderedDict | None

    def __init__(self, df: DataFrame):
        """ Construct a ``FlickerDataFrame`` from a ``pyspark.sql.DataFrame``. Construction will fail if the
        ``pyspark.sql.DataFrame`` contains duplicate column names.

        Parameters
        ----------
        df: ``pyspark.sql.DataFrame``
            The input ``pyspark.sql.DataFrame`` to initialize a ``FlickerDataFrame`` object

        Raises
        ------
        TypeError
            If the df parameter is not an instance of ``pyspark.sql.DataFrame``
        ValueError
            If the df parameter contains duplicate column names

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> rows = [('spark', 1), ('pandas', 3), ('polars', 2)]
        >>> spark_df = spark.createDataFrame(rows, schema=['package', 'rank'])
        >>> df = FlickerDataFrame(spark_df)
        >>> df()
          package rank
        0   spark    1
        1  pandas    3
        2  polars    2
        """
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

    def __getitem__(self,
                    item: tuple | slice | str | list | Column | FlickerColumn) -> FlickerColumn | FlickerDataFrame:
        """ Index into the dataframe in various ways
        Parameters
        ----------
        item: tuple | slice | str | list | Column | FlickerColumn
            The index value to retrieve from the FlickerDataFrame object

        Returns
        -------
        FlickerColumn | FlickerDataFrame
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
        """ Return a selection of ``pyspark.sql.DataFrame`` as a ``pandas.DataFrame``.

        Parameters
        ----------
        n : int | None, optional
            Number of rows to return. If not specified, defaults to 5.
            If df.nrows < n, only df.nrows are returned.
            If n=None, all rows are returned.
        use_pandas_dtypes : bool, optional
            If False (recommended and default), the resulting pandas.DataFrame will have all column dtypes as object.
            This option preserves NaNs and None(s) as-is.
            If True, the resulting pandas.DataFrame will have parsed dtypes. This option may be a little faster, but it
            allows pandas to convert None(s) in numeric columns to NaNs.

        Returns
        -------
        pandas.DataFrame
            pandas DataFrame

        Examples
        -------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> rows = [('spark', 1), ('pandas', 3), ('polars', 2)]
        >>> df = FlickerDataFrame.from_rows(spark, rows, names=['package', 'rank'])
        >>> df() # call the FlickerDataFrame to quickly see a snippet
          package rank
        0   spark    1
        1  pandas    3
        2  polars    2
        """
        if n is None:
            n = self._df.count()
        if use_pandas_dtypes:
            return self._df.limit(n).toPandas()
        else:
            data = get_columns_as_dict(self._df, n)
            return pd.DataFrame.from_dict(data, dtype=object)[self._df.columns]

    def show(self, n: int | None = 5, truncate: bool | int = True, vertical: bool = False) -> None:
        """Prints the first ``n`` rows to the console as a (possibly) giant string. This is a pass-through method
        to ``pyspark.sql.DataFrame.show()``.

        Parameters
        ----------
        n: int, optional
            Number of rows to show. Defaults to 5.
        truncate: bool or int, optional
            If True, strings longer than 20 chars are truncated.
            If ``truncate > 1``, strings longer than ``truncate`` are truncated to ``length=truncate`` and
            made right-aligned.
        vertical: bool, optional
            If True, print output rows vertically (one line per column value).

        Returns
        -------
        None

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = FlickerDataFrame.from_shape(spark, 3, 2, names=['col1', 'col2'], fill='zero')
        >>> df.show()
        +----+----+
        |col1|col2|
        +----+----+
        |   0|   0|
        |   0|   0|
        |   0|   0|
        +----+----+
       """
        self._df.show(n=n, truncate=truncate, vertical=vertical)

    @property
    def nrows(self) -> int:
        """ Returns the number of rows. This method may take a long time to count all the rows in the dataframe.
        Once the number of rows is computed, it is automatically cached until the dataframe is mutated.
        Cached number of rows is returned immediately without having to re-count all the rows.

        Returns
        -------
        int
            number of rows

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = FlickerDataFrame.from_shape(spark, 1000, 2, names=['col1', 'col2'], fill='zero')
        >>> df.nrows
        1000
        """
        if self._nrows is None:
            self._nrows = self._df.count()
        return self._nrows

    @property
    def ncols(self) -> int:
        """ Returns the number of columns. This method always returns immediately no matter the number of rows in the
        dataframe.

        Returns
        -------
        int
            number of columns

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = FlickerDataFrame.from_shape(spark, 3, 2, names=['col1', 'col2'], fill='zero')
        >>> df.ncols
        2
        """
        return self._ncols

    @property
    def shape(self) -> tuple[int, int]:
        """ Returns the shape of the FlickerDataFrame as (nrows, ncols)

        Returns
        -------
        tuple[int, int]
            shape as (nrows, ncols)

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = FlickerDataFrame.from_shape(spark, 3, 2, names=['col1', 'col2'], fill='zero')
        >>> df.shape
        (3, 2)
        """
        return self.nrows, self.ncols

    @property
    def names(self) -> list[str]:
        """ Returns a list of column names in the FlickerDataFrame

        Returns
        -------
        list[str]
            list of column names in order of occurrence

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = FlickerDataFrame.from_shape(spark, 3, 2, names=['col1', 'col2'], fill='zero')
        >>> df.names
        ['col1', 'col2']
        """
        return self._df.columns

    @property
    def dtypes(self) -> OrderedDict:
        """ Returns the column names and corresponding data types as an OrderedDict.
        The order of key-value pairs in the output is the same order as that of (left-to-right) columns in the
        dataframe.

        Returns
        -------
        OrderedDict
            Keys are column names and values are dtypes

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = FlickerDataFrame.from_shape(spark, 3, 2, names=['col1', 'col2'], fill='zero')
        >>> df.dtypes
        OrderedDict([('col1', 'bigint'), ('col2', 'bigint')])
        """
        return self._dtypes

    @classmethod
    def from_shape(cls, spark: SparkSession, nrows: int, ncols: int, names: list[str] | None = None,
                   fill='zero') -> FlickerDataFrame:
        """ Create a FlickerDataFrame from a given shape and fill. This method is useful for creating test data and
        experimentation.

        Parameters
        ----------
        spark: SparkSession
            The Spark session used for creating the DataFrame.
        nrows: int
            The number of rows in the DataFrame.
        ncols: int
            The number of columns in the DataFrame.
        names: list[str] | None, optional
            The names of the columns in the DataFrame. If not provided, column names will be generated as
            '0', '1', '2', ..., f'{ncols -1}'.
        fill: str, optional
            The value used for filling the DataFrame. Default is 'zero'.
            Accepted values are: 'zero', 'one', 'rand', 'randn', 'rowseq', 'colseq'

        Returns
        -------
        FlickerDataFrame
            A new instance of the FlickerDataFrame class created from the given shape and parameters.

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = FlickerDataFrame.from_shape(spark, 3, 2, names=['col1', 'col2'], fill='rowseq')
        >>> df()
          col1 col2
        0    0    1
        1    2    3
        2    4    5
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
        elif fill == 'rowseq':
            data = np.arange(nrows * ncols).reshape(nrows, ncols, order='C')
        elif fill == 'colseq':
            data = np.arange(nrows * ncols).reshape(nrows, ncols, order='F')
        else:
            raise ValueError(f'fill={fill} is not supported')
        return cls(spark.createDataFrame(data=data, schema=names))

    @classmethod
    def from_rows(cls, spark: SparkSession, rows: Iterable[Iterable],
                  names: list[str] | None = None, nan_to_none: bool = True) -> FlickerDataFrame:
        """ Create a FlickerDataFrame from rows.
        Parameters
        ----------
        spark: SparkSession
        rows: Iterable[Iterable]
            The rows of data to be converted into a DataFrame. For example, ``[('row1', 1), ('row2', 2)]``.
        names: list[str] | None, optional
            The column names of the DataFrame. If None, column names will be generated as
            '0', '1', '2', ..., f'{ncols -1}'.
        nan_to_none : bool, optional
            Flag indicating whether to convert all NaN values to None. Default and recommended value is True.

        Returns
        -------
        FlickerDataFrame

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> rows = [['a', 1, 2.0], ['b', 2, 4.0]]
        >>> names = ['col1', 'col2', 'col3']
        >>> df = FlickerDataFrame.from_rows(spark, rows, names)
        >>> df()
          col1 col2 col3
        0    a    1  2.0
        1    b    2  4.0

        Raises
        ------
        ValueError
            If the rows contain different number of columns
        """
        if nan_to_none:
            data = [
                [None if is_nan_scalar(element) else element for element in row]
                for row in rows
            ]
        else:
            data = [
                [element for element in row]
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
                     names: list[str] | None = None, nan_to_none: bool = True) -> FlickerDataFrame:
        """ Create a ``FlickerDataFrame`` from columns

        Parameters
        ----------
        spark: SparkSession
        columns: Iterable[Iterable]
            The columns to create the DataFrame from. Each column should be an iterable. For example:
            ``[('col1', 'a'), (1, 2), ('col3', 'b')]``
        names: list[str] | None, optional
            The column names of the DataFrame. If None, column names will be generated as
            '0', '1', '2', ..., f'{ncols -1}'.
        nan_to_none: bool, optional
            Flag indicating whether to convert all NaN values to None. Default and recommended value is True.

        Returns
        -------
        FlickerDataFrame

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> columns = [[1, 2, 3], ['a', 'b', 'c']]
        >>> names = ['col1', 'col2']
        >>> df = FlickerDataFrame.from_columns(spark, columns, names)
        >>> df()
          col1 col2
        0    1    a
        1    2    b
        2    3    c

        Raises
        ------
        ValueError
            If the columns contain different number of rows
        """
        ncols = get_length(columns)
        maybe_nrows = set([get_length(column) for column in columns])
        if len(maybe_nrows) > 1:
            raise ValueError(f'columns contain different number of rows: {maybe_nrows}')
        nrows = maybe_nrows.pop()
        data = [[None] * ncols for _ in range(nrows)]  # Python containers: must use "for" outer loop
        if nan_to_none:
            for j, column in enumerate(columns):
                for i, element in enumerate(column):
                    data[i][j] = None if is_nan_scalar(element) else element
        else:
            for j, column in enumerate(columns):
                for i, element in enumerate(column):
                    data[i][j] = element
        if names is None:
            names = [f'{i}' for i in range(ncols)]
        return cls(spark.createDataFrame(data=data, schema=names))

    @classmethod
    def from_records(cls, spark: SparkSession, records: Iterable[dict], nan_to_none: bool = True) -> FlickerDataFrame:
        """ Create a ``FlickerDataFrame`` from a list of dictionaries (similar to JSON lines format)

        Parameters
        ----------
        spark: SparkSession
        records: Iterable[dict]
            An iterable of dictionaries. Each dictionary represents a row (aka record).
        nan_to_none: bool, optional
            Flag indicating whether to convert all NaN values to None. Default and recommended value is True.

        Returns
        -------
        FlickerDataFrame

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> records = [{'col1': 1, 'col2': 1}, {'col1': 2, 'col2': 2}, {'col1': 3, 'col2': 3}]
        >>> df = FlickerDataFrame.from_records(spark, records)
        >>> df()
          col1 col2
        0    1    1
        1    2    2
        2    3    3
        """
        records = [dict(record) for record in records]  # two-level copy to avoid overwriting input value
        if nan_to_none:
            for record in records:
                for name, element in record.items():
                    record[name] = None if is_nan_scalar(element) else element
        else:
            for record in records:
                for name, element in record.items():
                    record[name] = element
        return cls(spark.createDataFrame(data=records))

    @classmethod
    def from_dict(cls, spark: SparkSession, data: dict, nan_to_none: bool = True) -> FlickerDataFrame:
        """ Create a ``FlickerDataFrame`` object from a dictionary, in which, dict keys represent column names and
        dict values represent column values.

        Parameters
        ----------
        spark: SparkSession
        data: dict
            The dictionary containing column names as keys and column values as values. For example,
            ``{'col1': [1, 2, 3], 'col2': [4, 5, 6]}``
        nan_to_none: bool, optional
            Flag indicating whether to convert all NaN values to None. Default and recommended value is True.

        Returns
        -------
        FlickerDataFrame

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        >>> df = FlickerDataFrame.from_dict(spark, data)
        >>> df()
          col1 col2
        0    1    4
        1    2    5
        2    3    6
        """
        names = list(data.keys())
        ncols = len(names)
        maybe_nrows = set([len(column) for column in data.values()])
        if len(maybe_nrows) > 1:
            raise ValueError(f'columns contain different number of rows: {maybe_nrows}')
        nrows = maybe_nrows.pop()
        rows = [[None] * ncols for _ in range(nrows)]  # Python containers: must use "for" outer loop
        if nan_to_none:
            for j, column in enumerate(data.values()):
                for i, element in enumerate(column):
                    rows[i][j] = None if is_nan_scalar(element) else element
        else:
            for j, column in enumerate(data.values()):
                for i, element in enumerate(column):
                    rows[i][j] = element
        return cls(spark.createDataFrame(data=rows, schema=names))

    def to_dict(self, n: int | None = 5) -> dict:
        """ Converts the ``FlickerDataFrame`` into a dictionary representation, in which, dict keys represent
        column names and dict values represent column values.

        Parameters
        ----------
        n: int | None, optional
            Number of rows to return. If not specified, defaults to 5.
            If ``df.nrows < n``, only df.nrows are returned.
            If ``n=None``, all rows are returned.

        Returns
        -------
        dict
            A dictionary representation of the ``FlickerDataFrame`` where keys are
            column names and values are lists containing up to ``n`` values from each column.

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = FlickerDataFrame.from_shape(spark, 3, 2, names=['col1', 'col2'], fill='colseq')
        >>> df()
          col1 col2
        0    0    3
        1    1    4
        2    2    5
        >>> df.to_dict(n=2)
        {'col1': [0, 1], 'col2': [3, 4]}
        """
        return get_columns_as_dict(self._df, n)

    @classmethod
    def from_pandas(cls, spark: SparkSession, df: pd.DataFrame, nan_to_none: bool = True) -> FlickerDataFrame:
        """Create a ``FlickerDataFrame`` from a ``pandas.DataFrame``

        Parameters
        ----------
        spark: SparkSession
        df: pd.DataFrame
            The pandas DataFrame to convert to a FlickerDataFrame.
        nan_to_none: bool, optional
            Flag indicating whether to convert all NaN values to None. Default and recommended value is True.

        Returns
        -------
        FlickerDataFrame

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> pandas_df = pd.DataFrame({'col1': [1, np.nan, 3], 'col2': [4, 5, np.nan]})
        >>> pandas_df
           col1  col2
        0   1.0   4.0
        1   NaN   5.0
        2   3.0   NaN
        >>> df = FlickerDataFrame.from_pandas(spark, pandas_df, nan_to_none=True)
        >>> df()
           col1  col2
        0   1.0   4.0
        1  None   5.0
        2   3.0  None

        >>> df = FlickerDataFrame.from_pandas(spark, pandas_df, nan_to_none=False)
        >>> df()
          col1 col2
        0  1.0  4.0
        1  NaN  5.0
        2  3.0  NaN
        """
        df = df.copy(deep=True)  # deep copy to avoid overwriting input variable
        if nan_to_none:
            nrows, ncols = df.shape
            for j in range(ncols):
                df.iloc[:, j] = df.iloc[:, j].astype(object)
                for i in range(nrows):
                    if is_nan_scalar(df.iloc[i, j]):
                        df.iloc[i, j] = None
        return cls(spark.createDataFrame(data=df))

    def to_pandas(self) -> pd.DataFrame:
        """Converts a ``FlickerDataFrame`` to a ``pandas.DataFrame``. Calling this method on a big
        ``FlickerDataFrame`` may result in out-of-memory errors.

        This method is simply a pass through to ``pyspark.sql.DataFrame.to_pandas()``. Consider using
        ``FlickerDataFrame.__call___()`` instead of ``FlickerDataFrame.to_pandas()`` because
        ``pyspark.sql.DataFrame.to_pandas()`` can cause unwanted None to NaN conversions. See example below.

        Returns
        --------
        pandas.DataFrame
            pandas DataFrame

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> pandas_df = pd.DataFrame({'col1': [1.0, np.nan, None], 'col2': [4.0, 5.0, np.nan]}, dtype=object)
        >>> pandas_df
           col1 col2
        0   1.0  4.0
        1   NaN  5.0
        2  None  NaN
        >>> df = FlickerDataFrame.from_pandas(spark, pandas_df, nan_to_none=False)
        >>> df()
           col1 col2
        0   1.0  4.0
        1   NaN  5.0
        2  None  NaN
        >>> df.to_pandas()  # causes unwanted None to NaN conversion in df.to_pandas().iloc[2, 0]
           col1  col2
        0   1.0   4.0
        1   NaN   5.0
        2   NaN   NaN
        """
        return self._df.toPandas()

    def head(self, n: int | None = 5) -> FlickerDataFrame:
        """Return top ``n`` rows as a ``FlickerDataFrame``. This method differs from ``FlickerDataFrame.__call__()``,
        which returns a ``pandas.DataFrame``.

        Parameters
        ----------
        n: int | None, optional
            Number of rows to return. If not specified, defaults to 5.
            If ``df.nrows < n``, only df.nrows are returned.
            If ``n=None``, all rows are returned.

        Returns
        -------
        FlickerDataFrame
            A new instance of FlickerDataFrame containing top (at most) ``n`` rows

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = FlickerDataFrame.from_shape(spark, 10, 2, names=['col1', 'col2'], fill='zero')
        >>> df.head(3)
        FlickerDataFrame[col1: bigint, col2: bigint]
        """
        if n is None:
            df = self._df
        else:
            df = self._df.limit(n)
        return self.__class__(df)

    def take(self, n: int | None = 5, convert_to_dict: bool = True) -> list[dict | Row]:
        """ Return top ``n`` rows as a list.

        Parameters
        ----------
        n: int | None, optional
            Number of rows to return. If not specified, defaults to 5.
            If ``df.nrows < n``, only df.nrows are returned.
            If ``n=None``, all rows are returned.
        convert_to_dict: bool, optional
            If False, output is a list of ``pyspark.sql.Row`` objects.
            If True, output is a list of ``dict`` objects.

        Returns
        -------
        list[dict | Row]
            A list of at most n items. Each item is either a pyspark.sql.Row or a dict object.

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> rows = [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]
        >>> df = FlickerDataFrame.from_rows(spark, rows, names=['col1', 'col2'])
        >>> df.take(2, convert_to_dict=True)
        [{'col1': 1, 'col2': 'a'}, {'col1': 2, 'col2': 'b'}]
        >>> df.take(2, convert_to_dict=False)
        [Row(col1=1, col2='a'), Row(col1=2, col2='b')]
        """
        if n is None:
            n = self._df.count()
        if convert_to_dict:
            return [row.asDict(recursive=True) for row in self._df.take(n)]
        else:
            return self._df.take(n)

    def drop(self, names: list[str]) -> FlickerDataFrame:
        """Delete columns by name. This is the non-mutating form of the ``__del__`` method.

        Parameters
        ----------
        names: list[str]
            A list of column names to delete from the FlickerDataFrame.

        Returns
        -------
        FlickerDataFrame
            A new instance of the FlickerDataFrame class with the specified columns deleted

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = FlickerDataFrame.from_shape(spark, 3, 4, names=['col1', 'col2', 'col3', 'col4'], fill='zero')
        >>> df
        FlickerDataFrame[col1: bigint, col2: bigint, col3: bigint, col4: bigint]
        >>> df.drop(['col2', 'col4'])
        FlickerDataFrame[col1: bigint, col3: bigint]
        >>> df
        FlickerDataFrame[col1: bigint, col2: bigint, col3: bigint, col4: bigint]
        """
        self._check_names(names)
        return self.__class__(self._df.drop(*names))

    def rename(self, from_to_mapper: dict[str, str]) -> FlickerDataFrame:
        """Renames columns in the FlickerDataFrame based on the provided mapping of the form
        ``{'old_col_name1': 'new_col_name1', 'old_col_name2': 'new_col_name2', ...}``.
        This is a non-mutating method.

        Parameters
        ----------
        from_to_mapper: dict[str, str]
            A dictionary containing the mapping of current column names to new column names

        Returns
        -------
        FlickerDataFrame
            A new instance of ``FlickerDataFrame`` with renamed columns

        Raises
        ------
        TypeError
            If the provided ``from_to_mapper`` is not a dictionary
        KeyError
            If any of the keys in ``from_to_mapper`` do not match existing column names in the ``FlickerDataFrame``

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = FlickerDataFrame.from_shape(spark, 3, 4, names=['col1', 'col2', 'col3', 'col4'], fill='zero')
        >>> df
        FlickerDataFrame[col1: bigint, col2: bigint, col3: bigint, col4: bigint]
        >>> df.rename({'col1': 'col_a', 'col3': 'col_c'})
        FlickerDataFrame[col_a: bigint, col2: bigint, col_c: bigint, col4: bigint]
        >>> df  # df is not mutated
        FlickerDataFrame[col1: bigint, col2: bigint, col3: bigint, col4: bigint]
        """
        self._check_names(from_to_mapper.keys())
        return self.__class__(self._df.withColumnsRenamed(from_to_mapper))

    def sort(self, names: list[str], ascending: bool = True) -> FlickerDataFrame:
        """ Returns a new :class:`DataFrame` sorted by the specified column name(s). This non-mutating method is
        a pass-through to ``pyspark.sql.DataFrame.sort`` but with some checks and a slightly different function
        signature.

        Parameters
        ----------
        names: list[str]
            The list of column names to sort the DataFrame by

        ascending: bool, optional (default=True)
            Whether to sort the DataFrame in ascending order or not

        Returns
        -------
        FlickerDataFrame

        Raises
        ------
        KeyError
            If ``names`` contains a non-existant column name

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> rows = [(10, 1), (1, 2), (100, 3)]
        >>> df = FlickerDataFrame.from_rows(spark, rows, names=['x', 'y'])
        >>> df()
             x  y
        0   10  1
        1    1  2
        2  100  3
        >>> df.sort(['x'])
        FlickerDataFrame[x: bigint, y: bigint]
        >>> df.sort(['x'])()
             x  y
        0    1  2
        1   10  1
        2  100  3
        >>> df  # df is not mutated
        FlickerDataFrame[x: bigint, y: bigint]
        """
        self._check_names(names)
        return self.__class__(self._df.sort(*names, ascending=ascending))

    def unique(self) -> FlickerDataFrame:
        """ Returns a new FlickerDataFrame with unique rows. This non-mutating method is just a pass-through to
        ``pyspark.sql.DataFrame.distinct``.

        Returns
        -------
        FlickerDataFrame
            A new FlickerDataFrame with unique rows

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = FlickerDataFrame.from_shape(spark, 3, 2, names=['col1', 'col2'], fill='zero')
        >>> df()
          col1 col2
        0    0    0
        1    0    0
        2    0    0
        >>> df.unique()
        FlickerDataFrame[col1: bigint, col2: bigint]
        >>> df.unique()()
          col1 col2
        0    0    0
        >>> df.shape  # df is not mutated
        (3, 2)
        """
        return self.__class__(self._df.distinct())

    def describe(self) -> pd.DataFrame:
        """ Returns a ``pandas.DataFrame`` with statistical summary of the FlickerDataFrame. This method supports
        numeric (int, bigint, float, double), string, timestamp, boolean columns. Unsupported columns are ignored
        without an error. This method returns a different and better dtyped output than
        ``pyspark.sql.DataFrame.describe``.

        The output contains count, mean, stddev, min, and max values.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame with statistical summary of the FlickerDataFrame

        Examples
        --------
        >>> from datetime import datetime, timedelta
        >>> spark = SparkSession.builder.getOrCreate()
        >>> t = datetime(2023, 1, 1)
        >>> dt = timedelta(days=1)
        >>> rows = [('Bob', 23, 100.0, t - dt, False), ('Alice', 22, 110.0, t, True), ('Tom', 21, 120.0, t + dt, False)]
        >>> names = ['name', 'age', 'weight', 'time', 'is_jedi']
        >>> df = FlickerDataFrame.from_rows(spark, rows, names)
        >>> df()
            name age weight                 time is_jedi
        0    Bob  23  100.0  2022-12-31 00:00:00   False
        1  Alice  22  110.0  2023-01-01 00:00:00    True
        2    Tom  21  120.0  2023-01-02 00:00:00   False
        >>> df.describe()
                 name   age weight                 time   is_jedi
        count       3     3      3                    3         3
        max       Tom    23  120.0  2023-01-02 00:00:00      True
        mean      NaN  22.0  110.0  2023-01-01 00:00:00  0.333333
        min     Alice    21  100.0  2022-12-31 00:00:00     False
        stddev    NaN   1.0   10.0       1 day, 0:00:00   0.57735
        >>> df.describe()['time']['stddev']  # output contains appropriately typed values instead of strings
        datetime.timedelta(days=1)
        """
        return get_summary(self._df)

    def concat(self, other: FlickerDataFrame | DataFrame, ignore_names: bool = False) -> FlickerDataFrame:
        """ Return a new FlickerDataFrame with rows from this and other dataframe concatenated together. This is a
        non-mutating method that calls ``pyspark.sql.DataFrame.union`` after some checks.
        Resulting concatenated DataFrame will always contain the same column names in the same order as that in the
        current DataFrame.

        Parameters
        ----------
        other : FlickerDataFrame | pyspark.sql.DataFrame
            The DataFrame to concatenate with the current DataFrame
        ignore_names : bool, optional (default=False)
            If ``True``, the column names of the ``other`` dataframe are ignored when concatenating. Concatenation
            happens by column order and resulting dataframe will have column names in the same order as the current
            dataframe.
            If ``False``, this method checks that current and ``other`` dataframe have the same column names (even
            if not in the same order). If this check fails, a ``KeyError`` is raised.

        Returns
        -------
        FlickerDataFrame
            The concatenated DataFrame

        Raises
        ------
        KeyError
            If the DataFrames have different sets of column names and ``ignore_names=False``

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df_zero = FlickerDataFrame.from_shape(spark, 2, 2, names=['a', 'b'], fill='zero')
        >>> df_one = FlickerDataFrame.from_shape(spark, 2, 2, names=['a', 'b'], fill='one')
        >>> df_rand = FlickerDataFrame.from_shape(spark, 2, 2, names=['b', 'c'], fill='rand')
        >>> df_zero.concat(df_one)
        FlickerDataFrame[a: bigint, b: bigint]
        >>> df_zero.concat(df_one, ignore_names=False)()
           a  b
        0  0  0
        1  0  0
        2  1  1
        3  1  1
        >>> df_zero.concat(df_one, ignore_names=True)()  # ignore_names has no effect
           a  b
        0  0  0
        1  0  0
        2  1  1
        3  1  1
        >>> df_zero.concat(df_rand, ignore_names=True)()
                  a         b
        0       0.0       0.0
        1       0.0       0.0
        2   0.85428  0.148739
        3  0.031665   0.14922
        >>> df_zero.concat(df_rand, ignore_names=False)  # KeyError
        """
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
        """ Merge the current FlickerDataFrame with another dataframe. This non-mutating method returns the merged
        dataframe as a FlickerDataFrame.

        Note that ``FlickerDataFrame.merge`` is different from ``FlickerDataFrame.join`` in both function signature
        and the merged/joined result.

        Parameters
        ----------
        right: FlickerDataFrame or DataFrame
            The right dataframe to merge with
        on: Iterable[str]
            Column names to 'join' on. The column names must exist in both left and right dataframes.
            The column names provided in ``on`` are not duplicated and are not renamed using prefixes/suffixes.
        how: str, optional (default='inner')
            Type of join to perform. Possible values are ``{'inner', 'outer', 'left', 'right'}``.
        lprefix: str, optional (default='')
            Prefix to add to column names from the left dataframe that are duplicated in the merge result
        lsuffix: str, optional (default='_l')
            Suffix to add to column names from the left dataframe that are duplicated in the merge result
        rprefix: str, optional (default='')
            Prefix to add to column names from the right dataframe that are duplicated in the merge result
        rsuffix: str, optional (default='_r')
            Suffix to add to column names from the right dataframe that are duplicated in the merge result

        Returns
        -------
        FlickerDataFrame

        Raises
        ------
        TypeError
            If `on` is not an ``Iterable[str]`` or if it is a ``dict``
        ValueError
            If `on` is an empty ``Iterable[str]``
        TypeError
            If any element in `on` is not a ``str``
        KeyError
            If renaming results in duplicate column names in the left dataframe
        KeyError
            If renaming results in duplicate column names in the right dataframe

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> left = FlickerDataFrame.from_rows(spark, [('a', 1), ('b', 2), ('c', 3), ], ['name', 'number'])
        >>> right = FlickerDataFrame.from_rows(spark, [('a', 4), ('d', 5), ('e', 6), ], ['name', 'number'])
        >>> inner_merge = left.merge(right, on=['name'], how='inner')
        >>> inner_merge()
          name number_l number_r
        0    a        1        4
        >>> left_merge = left.merge(right, on=['name'], how='left')
        >>> left_merge()
          name number_l number_r
        0    a        1        4
        1    b        2     None
        2    c        3     None
        """
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
        """ Join the current FlickerDataFrame with another dataframe. This non-mutating method returns the joined
        dataframe as a FlickerDataFrame.

        This method preserves duplicate column names (that are joined on) by renaming them in the join result.
        Note that ``FlickerDataFrame.join`` is different from ``FlickerDataFrame.merge`` in both function signature
        and the merged/joined result.

        Parameters
        ----------
        right: FlickerDataFrame | DataFrame
            The right DataFrame to join with the left DataFrame.
        on: dict[str, str]
            Dictionary specifying which column names to join on. Keys represent column names from the left dataframe
            and values represent column names from the right dataframe.
        how: str, optional (default='inner')
            The type of join to perform
            - 'inner': Returns only the matching rows from both DataFrames
            - 'left': Returns all the rows from the left DataFrame and the matching rows from the right DataFrame
            - 'right': Returns all the rows from the right DataFrame and the matching rows from the left DataFrame
            - 'outer': Returns all the rows from both DataFrames, including unmatched rows, with `null` values for
                       non-matching columns
        lprefix: str, optional (default='')
            Prefix to add to column names from the left dataframe that are duplicated in the join result
        lsuffix: str, optional (default='_l')
            Suffix to add to column names from the left dataframe that are duplicated in the join result
        rprefix: str, optional (default='')
            Prefix to add to column names from the right dataframe that are duplicated in the join result
        rsuffix: str, optional (default='_r')
            Suffix to add to column names from the right dataframe that are duplicated in the join result

        Returns
        -------
        FlickerDataFrame

        Raises
        ------
        TypeError
            If the `on` parameter is not a dictionary
        ValueError
            If the `on` parameter is an empty dictionary
        TypeError
            If the keys or values of the `on` parameter are not of str type
        KeyError
            If the left or right DataFrame contains duplicate column names after renaming
        NotImplementedError
            To prevent against unexpected changes in the underlying ``pyspark.sql.DataFrame.join``

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> left = FlickerDataFrame.from_rows(spark, [('a', 1), ('b', 2), ('c', 3), ], ['x', 'number'])
        >>> right = FlickerDataFrame.from_rows(spark, [('a', 4), ('d', 5), ('e', 6), ], ['x', 'number'])
        >>> inner_join = left.join(right, on={'x': 'x'}, how='inner')
        >>> inner_join()  # 'x' columns from both left and right dataframes is preserved
          x_l number_l x_r number_r
        0   a        1   a        4

        >>> spark = SparkSession.builder.getOrCreate()
        >>> left = FlickerDataFrame.from_rows(spark, [('a', 1), ('b', 2), ('c', 3), ], ['x1', 'number'])
        >>> right = FlickerDataFrame.from_rows(spark, [('a', 4), ('d', 5), ('e', 6), ], ['x2', 'number'])
        >>> inner_join = left.join(right, on={'x1': 'x2'}, how='inner')
        >>> inner_join()  # renaming happens only when needed
          x1 number_l x2 number_r
        0  a        1  a        4
        """
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
        """ Groups the rows of the DataFrame based on the specified column names, so we can run aggregation on them.
        Returns a ``FlickerGroupedData`` object. This method is a pass-through to ``pyspark.sql.DataFrame`` but
        returns a  ``FlickerGroupedData`` object instead of a ``pyspark.sql.GroupedData`` object.

        Parameters
        ----------
        names: list[str]
            The column names based on which the DataFrame rows should be grouped

        Returns
        -------
        FlickerGroupedData

        Examples
        --------
        >>> spark = SparkSession.builder.getOrCreate()
        >>> rows = [('spark', 10), ('pandas', 10), ('spark', 100)]
        >>> df = FlickerDataFrame.from_rows(spark, rows, names=['name', 'number'])
        >>> df.groupby(['name'])
        FlickerGroupedData[grouping expressions: [name], value: [name: string, number: bigint], type: GroupBy]
        >>> df.groupby(['name']).count()
        FlickerDataFrame[name: string, count: bigint]
        >>> df.groupby(['name']).count()()
             name count
        0   spark     2
        1  pandas     1
        """
        return FlickerGroupedData(self._df, self._df.groupBy(*names))


# Import here to avoid circular imports
# https://github.com/ankur-gupta/rain/tree/v1#circular-imports-or-dependencies
from .column import FlickerColumn
from .group import FlickerGroupedData
