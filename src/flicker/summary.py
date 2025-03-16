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
import pandas as pd
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from datetime import datetime, timedelta

from .mkname import mkname
from .utils import get_names_by_dtype


def get_column_min(df: DataFrame, name: str):
    return df.select(F.min(df[name])).collect()[0][f'min({name})']


def get_column_max(df: DataFrame, name: str):
    return df.select(F.max(df[name])).collect()[0][f'max({name})']


def get_column_mean(df: DataFrame, name: str) -> float | None:
    return df.select(F.mean(df[name])).collect()[0][f'avg({name})']


def get_column_stddev(df: DataFrame, name: str) -> float | None:
    # Uses (n-1) for denominator
    return df.select(F.stddev(df[name])).collect()[0][f'stddev({name})']


def get_boolean_column_mean(df: DataFrame, name: str) -> float | None:
    int_name = mkname(df.columns, prefix=f'{name}_as_int_', suffix='')
    df_with_column_as_int = df.withColumn(int_name, df[name].astype('int'))[[int_name]]
    return get_column_mean(df_with_column_as_int, int_name)


def get_boolean_column_stddev(df: DataFrame, name: str) -> float | None:
    int_name = mkname(df.columns, prefix=f'{name}_as_int_', suffix='')
    df_with_column_as_int = df.withColumn(int_name, df[name].astype('int'))[[int_name]]
    return get_column_stddev(df_with_column_as_int, int_name)


def get_timestamp_column_mean(df: DataFrame, name: str) -> datetime | None:
    name_seconds = mkname(df.columns, prefix=f'{name}_as_seconds_', suffix='')
    df_with_column_as_seconds = df.withColumn(name_seconds, df[name].astype('double'))[[name_seconds]]
    mean_in_seconds = get_column_mean(df_with_column_as_seconds, name_seconds)
    if mean_in_seconds is None:
        return None
    else:
        return datetime.fromtimestamp(mean_in_seconds)


def get_timestamp_column_stddev(df: DataFrame, name: str) -> timedelta | None:
    name_seconds = mkname(df.columns, prefix=f'{name}_as_seconds_', suffix='')
    df_with_column_as_seconds = df.withColumn(name_seconds, df[name].astype('double'))[[name_seconds]]
    stddev_in_seconds = get_column_stddev(df_with_column_as_seconds, name_seconds)
    if stddev_in_seconds is None:
        return None
    else:
        return timedelta(seconds=stddev_in_seconds)


def get_columns_as_dict(df: DataFrame, n: int | None) -> dict:
    """ Convert spark DataFrame into a dictionary of the form
        {'col_name_1': [column 1 data], 'col_name_2': [column 2 data], ..., 'col_name_n': [column 3 data]}

        User should exercise care to not run this function on a big dataframe to avoid OOM errors.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame from which to extract column data.
    n : int | None
        The number of rows to be considered when creating the dictionary. If None, all rows are considered.

    Returns
    -------
    dict
        A dictionary where the keys are the column names and the values are lists of column data.

    """
    if n is None:
        rows = df.collect()
    else:
        rows = df.limit(n).collect()

    data = {name: [] for name in df.columns}
    for row in rows:
        for name in df.columns:
            data[name].append(row[name])
    return data


def get_summary_spark_supported_dtypes(df: DataFrame) -> pd.DataFrame:
    """ Process the output of pyspark.sql.DataFrame.describe() to make it more useful.

        This method converts the output of pyspark.sql.DataFrame.describe() into a pandas DataFrame with better dtypes
        instead of the pyspark's default string representation of all dtypes.

        Note that pyspark.sql.DataFrame.describe() ignores many dtypes including boolean and timestamp dtypes.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The input DataFrame that contains the data for which the summary is to be computed.

    Returns
    -------
    summary : pandas.DataFrame
        Contains the summary statistics for all supported columns of the input DataFrame.

    Example
    -------
    df = spark.createDataFrame([(1, 'A', 10.0), (2, 'B', 20.0), (3, 'C', 30.0)], ['id', 'name', 'value'])
    summary = get_summary_spark_supported_dtypes(df)
    print(summary)

    # Output:
    #               id name value
    # summary
    # count      3    3     3
    # mean     2.0  NaN  20.0
    # stddev   1.0  NaN  10.0
    # min        1    A  10.0
    # max        3    C  30.0
    """
    columns_as_dict = get_columns_as_dict(df.describe(), None)
    summary = pd.DataFrame.from_dict(columns_as_dict, dtype=object)

    # First column is always the "true" summary column even if we already had a "summary" column in `self`.
    # Convert it to index always whether there was a "summary" column in `self` or not.
    # This way, we don't have to worry about name conflicts.
    summary.index = summary.iloc[:, 0]
    summary = summary.iloc[:, 1:]

    # By default, all elements of summary are strings, which is not useful. We modify these.
    summary.loc['count'] = summary.loc['count'].astype(int)
    summary.loc['mean'] = summary.loc['mean'].astype(float)
    summary.loc['stddev'] = summary.loc['stddev'].astype(float)

    # Convert each value to its appropriate dtype
    dtypes = {name: dtype for name, dtype in df.dtypes}
    for stat_name in ['min', 'max']:
        stat_vector = list(summary.loc[stat_name])
        for i, name in enumerate(summary.loc[stat_name].index):
            dtype = dtypes[name]
            if dtype in {'int', 'bigint'}:
                stat_vector[i] = int(stat_vector[i])
            elif dtype == 'double':
                stat_vector[i] = float(stat_vector[i])
            # spark ignores boolean columns during `df.describe()` call
            # FIXME: Find all other supported dtypes?
        summary.loc[stat_name] = stat_vector
    return summary


def get_summary_boolean_columns(df: DataFrame) -> pd.DataFrame:
    if len(set(df.columns)) != len(df.columns):
        raise KeyError(f'duplicate dataframe column names are not supported')
    boolean_names = get_names_by_dtype(df, 'boolean')
    boolean_df = df[boolean_names].withColumns({
        name: df[name].astype('int')  # Convert to int so we can use the built-in summary function
        for name in boolean_names
    })
    boolean_summary = get_summary_spark_supported_dtypes(boolean_df)
    for name in boolean_summary.columns:
        if boolean_summary.loc['min', name] == 0:
            boolean_summary.loc['min', name] = False
        elif boolean_summary.loc['min', name] == 1:
            boolean_summary.loc['min', name] = True

        if boolean_summary.loc['max', name] == 0:
            boolean_summary.loc['max', name] = False
        elif boolean_summary.loc['max', name] == 1:
            boolean_summary.loc['max', name] = True
    return boolean_summary


def get_summary_timestamp_columns(df: DataFrame) -> pd.DataFrame:
    timestamp_names = get_names_by_dtype(df, 'timestamp')
    timestamp_summary_dict = {}
    count = df.count()
    for name in timestamp_names:
        timestamp_summary_dict[name] = {
            'count': count,
            'mean': datetime.fromtimestamp(get_column_mean(df, name)),
            'stddev': get_timestamp_column_stddev(df, name),
            'min': get_column_min(df, name),
            'max': get_column_max(df, name)
        }
    return pd.DataFrame.from_dict(timestamp_summary_dict)


def get_summary(df: DataFrame) -> pd.DataFrame:
    non_boolean_summary = get_summary_spark_supported_dtypes(df)
    boolean_summary = get_summary_boolean_columns(df)
    timestamp_summary = get_summary_timestamp_columns(df)
    summary = pd.merge(non_boolean_summary, boolean_summary, how='outer', left_index=True, right_index=True)
    summary = pd.merge(summary, timestamp_summary, how='outer', left_index=True, right_index=True)
    ordered_names = [name for name in df.columns if name in summary.columns]
    return summary[ordered_names]
