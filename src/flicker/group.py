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
from typing import Callable

from pyspark.sql import GroupedData, DataFrame
from pyspark.sql.types import StructType
from pyspark.sql.functions import pandas_udf


class FlickerGroupedData:
    _df: DataFrame
    _grouped: GroupedData

    def __init__(self, df: DataFrame, grouped: GroupedData):
        if not isinstance(df, DataFrame):
            raise TypeError(f'df must be of type pyspark.sql.DataFrame; you provided type(df)={type(df)}')
        if len(df.columns) != len(set(df.columns)):
            # pyspark.sql.DataFrame:
            # 1. does NOT enforce uniqueness names
            # 2. enforces str-type for column names
            raise ValueError(f'df contains duplicate columns names which is not supported. '
                             f'Rename the columns to be unique.')
        if not isinstance(grouped, GroupedData):
            raise TypeError(f'grouped must be of type pyspark.sql.GroupedData but you provided {type(grouped)}')
        self._df = df
        self._grouped = grouped

    def __repr__(self):
        return 'Flicker' + repr(self._grouped)

    def __str__(self):
        return repr(self)

    def count(self) -> FlickerDataFrame:
        return FlickerDataFrame(self._grouped.count())

    def max(self, names: list[str]) -> FlickerDataFrame:
        return FlickerDataFrame(self._grouped.max(*names))

    def min(self, names: list[str]) -> FlickerDataFrame:
        return FlickerDataFrame(self._grouped.min(*names))

    def sum(self, names: list[str]) -> FlickerDataFrame:
        return FlickerDataFrame(self._grouped.sum(*names))

    def mean(self, names: list[str]) -> FlickerDataFrame:
        # FIXME: Returns columns named 'avg(a)' instead of 'mean(a)'
        return FlickerDataFrame(self._grouped.mean(*names))

    def agg(self, exprs: list) -> FlickerDataFrame:
        return FlickerDataFrame(self._grouped.agg(*exprs))

    def apply(self, f: Callable, schema: StructType | str) -> FlickerDataFrame:
        return FlickerDataFrame(self._grouped.applyInPandas(f, schema))

    def apply_with_state(self, f: 'PandasGroupedMapFunctionWithState',
                         outputStructType: StructType | str, stateStructType: StructType | str,
                         outputMode: str, timeoutConf: str) -> FlickerDataFrame:
        return FlickerDataFrame(
            self._grouped.applyInPandasWithState(f, outputStructType, stateStructType, outputMode, timeoutConf)
        )

    def apply_spark(self, udf: pandas_udf) -> FlickerDataFrame:
        return FlickerDataFrame(self._grouped.apply(udf))

    def cogroup(self, other: GroupedData | FlickerGroupedData) -> 'PandasCogroupedOps':
        if isinstance(other, FlickerGroupedData):
            other = other._grouped
        return self._grouped.cogroup(other)

    def pivot(self, pivot_col_name: str, values=None) -> FlickerGroupedData:
        g = self._grouped.pivot(pivot_col_name, values)
        return FlickerGroupedData(g._df, g)


# Import here to avoid circular imports
# https://github.com/ankur-gupta/rain/tree/v1#circular-imports-or-dependencies
from .dataframe import FlickerDataFrame
