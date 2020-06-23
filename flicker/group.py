# Copyright 2020 Flicker Contributors
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

from pyspark.sql import GroupedData, DataFrame


class FlickerGroupedData(object):
    """
    A thin wrapper over pyspark.sql.GroupedData to avoid returning
    pyspark.sql.DataFrame object when FlickerDataFrame.groupby is called.
    See pyspark.sql.GroupedData for documentation of the methods of this
    class.
    """

    def __init__(self, grouped):
        if not isinstance(grouped, GroupedData):
            msg = ('grouped must be of type pyspark.sql.GroupedData but you '
                   'provided type(grouped)="{}"')
            msg = msg.format(str(type(grouped)))
            raise TypeError(msg)
        self._grouped = grouped

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self._grouped))

    def __str__(self):
        return repr(self)

    def agg(self, *exprs):
        out = self._grouped.agg(*exprs)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def count(self):
        out = self._grouped.count()
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def mean(self, *cols):
        out = self._grouped.mean(*cols)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def avg(self, *cols):
        out = self._grouped.avg(*cols)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def max(self, *cols):
        out = self._grouped.max(*cols)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def min(self, *cols):
        out = self._grouped.min(*cols)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def sum(self, *cols):
        out = self._grouped.sum(*cols)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def pivot(self, pivot_col, values=None):
        out = self._grouped.mean(pivot_col=pivot_col, values=values)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def apply(self, udf):
        out = self._grouped.apply(udf)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out


# This is to avoid ImportError due to circular imports.
# See https://github.com/ankur-gupta/rain#circular-imports-or-dependencies.
from flicker.dataframe import FlickerDataFrame
