# Copyright 2025 Flicker Contributors
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

from typing import Iterable
from pyspark.sql import DataFrame
from flicker.dataframe import FlickerDataFrame
from functools import reduce


def concat(dfs: Iterable[FlickerDataFrame | DataFrame], ignore_names: bool = False) -> FlickerDataFrame:
    """ Return a new FlickerDataFrame with rows from all dataframes concatenated together.
        Resulting concatenated DataFrame will always contain the same column names in the same order as that in the
        current first DataFrame.

        Parameters
        ----------
        dfs : Iterable[FlickerDataFrame | pyspark.sql.DataFrame]
            DataFrames to concatenate
        ignore_names : bool, optional (default=False)
            If `True`, the column names of all dataframes except the first dataframe in `dfs` are ignored when
            concatenating. Concatenation happens by column order and resulting dataframe will have column names in the
            same order as the first dataframe.
            If `False`, this method checks that all dataframes in `dfs` have the same column names (even
            if not in the same order). If this check fails, a ``KeyError`` is raised.

        Returns
        -------
        FlickerDataFrame
            The concatenated DataFrame
    """
    dfs = [
        FlickerDataFrame(df) if isinstance(df, DataFrame) else df
        for df in dfs
    ]
    if len(dfs) == 0:
        raise ValueError(f'No dataframes to concat')

    column_counts = [df.ncols for df in dfs]
    if len(set(column_counts)) > 1:
        raise ValueError(f'Dataframes have differing number of columns. Cannot concat them at all.')

    if not ignore_names:
        sets_of_column_names = [frozenset(df.names) for df in dfs]
        if len(set(sets_of_column_names)) > 1:
            raise KeyError(f'Dataframes have different sets of column names. Cannot concat when ignore_names=False')
        column_names = dfs[0].names
        dfs = [df[column_names] for df in dfs]

    return reduce(lambda first, second: first.concat(second, ignore_names=ignore_names), dfs)
