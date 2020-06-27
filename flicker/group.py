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
        """
        Default constructor for FlickerGroupedData.

        Parameters
        ----------
        grouped: pyspark.sql.GroupedData

        Returns
        -------
            FlickerGroupedData

        Examples
        --------
        >>> df = FlickerDataFrame.from_dict(spark, {
            'a': [1, 2, 3, 1],
            'b': ['1', 'a', '2', 'b'],
            'c': [True, None, True, True]
        })
        >>> df.groupby(['a']).count()()
           a  count
        0  1      2
        1  3      1
        2  2      1
        """
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
        """Compute aggregates.

        Parameters
        ----------
        exprs: Any
            See pyspark.sql.GroupedData.agg.

        Returns
        -------
            FlickerDataFrame or Any
        """
        out = self._grouped.agg(*exprs)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def count(self):
        """Counts the number of records for each group.

        Returns
        -------
            FlickerDataFrame or Any
        """
        out = self._grouped.count()
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def mean(self, *cols):
        """Computes average values for each numeric columns for each group.

        Parameters
        ----------
        cols: Any
            See pyspark.sql.GroupedData.mean

        Returns
        -------
            FlickerDataFrame or Any
        """
        out = self._grouped.mean(*cols)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def avg(self, *cols):
        """Computes average values for each numeric columns for each group.

        Parameters
        ----------
        cols: Any
            See pyspark.sql.GroupedData.avg

        Returns
        -------
            FlickerDataFrame or Any
        """
        out = self._grouped.avg(*cols)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def max(self, *cols):
        """Computes the max value for each numeric columns for each group.

        Parameters
        ----------
        cols: Any
            See pyspark.sql.GroupedData.max

        Returns
        -------
            FlickerDataFrame or Any
        """
        out = self._grouped.max(*cols)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def min(self, *cols):
        """Computes the min value for each numeric columns for each group.

        Parameters
        ----------
        cols: Any
            See pyspark.sql.GroupedData.min

        Returns
        -------
            FlickerDataFrame or Any
        """
        out = self._grouped.min(*cols)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def sum(self, *cols):
        """Compute the sum for each numeric columns for each group.

        Parameters
        ----------
        cols: Any
            See pyspark.sql.GroupedData.sum

        Returns
        -------
            FlickerDataFrame or Any
        """
        out = self._grouped.sum(*cols)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def pivot(self, pivot_col, values=None):
        """See pyspark.sql.GroupedData.pivot.

        Parameters
        ----------
        pivot_col: str
            Name of the column to pivot.
        values: list of str
            List of values that will be translated to columns in the output
            DataFrame.

        Returns
        -------
            FlickerDataFrame or Any
        """
        out = self._grouped.mean(pivot_col=pivot_col, values=values)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out

    def apply(self, udf):
        """See pyspark.sql.GroupedData.apply.

        Parameters
        ----------
        udf: Any
            See pyspark.sql.GroupedData.apply.

        Returns
        -------
            FlickerDataFrame or Any
        """
        out = self._grouped.apply(udf)
        if isinstance(out, DataFrame):
            out = FlickerDataFrame(out)
        return out


# This is to avoid ImportError due to circular imports.
# See https://github.com/ankur-gupta/rain#circular-imports-or-dependencies.
from flicker.dataframe import FlickerDataFrame
