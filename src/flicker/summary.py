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

from .utils import get_names_by_dtype


def get_columns_as_dict(df: DataFrame, n: int | None) -> dict:
    if n is None:
        rows = df.collect()
    else:
        rows = df.limit(n).collect()

    data = {name: [] for name in df.columns}
    for row in rows:
        for name in df.columns:
            data[name].append(row[name])
    return data


def get_summary_non_boolean(df: DataFrame):
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
            elif dtype == 'boolean':
                stat_vector[i] = bool(stat_vector[i])
            # FIXME: add datetime and other dtypes?
        summary.loc[stat_name] = stat_vector
    return summary


def get_summary_boolean_only(df: DataFrame):
    boolean_names = get_names_by_dtype(df, 'boolean')
    boolean_df = df[boolean_names].withColumns({
        name: df[name].astype('int')
        for name in boolean_names
    })
    boolean_summary = get_summary_non_boolean(boolean_df)
    for name in boolean_summary.columns:
        if boolean_summary[name].loc['min'] == 0:
            boolean_summary[name].loc['min'] = False
        elif boolean_summary[name].loc['min'] == 1:
            boolean_summary[name].loc['min'] = True

        if boolean_summary[name].loc['max'] == 0:
            boolean_summary[name].loc['max'] = False
        elif boolean_summary[name].loc['max'] == 1:
            boolean_summary[name].loc['max'] = True
    return boolean_summary
