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
import pytest
from flicker import FlickerDataFrame, FlickerGroupedData


def test_grouped_count(spark):
    df = FlickerDataFrame.from_dict(spark, {'a': [1, 2, 1, 2], 'b': [1.0, 2.0, 3.0, 4.0]})
    g = df.groupby(['a'])
    assert isinstance(g, FlickerGroupedData)
    df_count_by_a = g.count()
    assert isinstance(df_count_by_a, FlickerDataFrame)
    assert df_count_by_a.nrows == 2
    assert df_count_by_a.ncols == 2
    assert df_count_by_a.take(None, convert_to_dict=True) == [
        {'a': 1, 'count': 2},
        {'a': 2, 'count': 2}
    ]


def test_grouped_sum(spark):
    df = FlickerDataFrame.from_dict(spark, {'a': [1, 2, 1, 2], 'b': [1.0, 2.0, 3.0, 4.0]})
    g = df.groupby(['a'])
    assert isinstance(g, FlickerGroupedData)
    df_sum_by_a = g.sum(['b'])
    assert isinstance(df_sum_by_a, FlickerDataFrame)
    assert df_sum_by_a.nrows == 2
    assert df_sum_by_a.ncols == 2
    assert df_sum_by_a.take(None, convert_to_dict=True) == [
        {'a': 1, 'sum(b)': 4.0},
        {'a': 2, 'sum(b)': 6.0}
    ]


def test_grouped_min(spark):
    df = FlickerDataFrame.from_dict(spark, {'a': [1, 2, 1, 2], 'b': [1.0, 2.0, 3.0, 4.0]})
    g = df.groupby(['a'])
    assert isinstance(g, FlickerGroupedData)
    df_min_by_a = g.min(['b'])
    assert isinstance(df_min_by_a, FlickerDataFrame)
    assert df_min_by_a.nrows == 2
    assert df_min_by_a.ncols == 2
    assert df_min_by_a.take(None, convert_to_dict=True) == [
        {'a': 1, 'min(b)': 1.0},
        {'a': 2, 'min(b)': 2.0}
    ]


def test_grouped_max(spark):
    df = FlickerDataFrame.from_dict(spark, {'a': [1, 2, 1, 2], 'b': [1.0, 2.0, 3.0, 4.0]})
    g = df.groupby(['a'])
    assert isinstance(g, FlickerGroupedData)
    df_max_by_a = g.max(['b'])
    assert isinstance(df_max_by_a, FlickerDataFrame)
    assert df_max_by_a.nrows == 2
    assert df_max_by_a.ncols == 2
    assert df_max_by_a.take(None, convert_to_dict=True) == [
        {'a': 1, 'max(b)': 3.0},
        {'a': 2, 'max(b)': 4.0}
    ]


def test_grouped_mean(spark):
    df = FlickerDataFrame.from_dict(spark, {'a': [1, 2, 1, 2], 'b': [1.0, 2.0, 3.0, 4.0]})
    g = df.groupby(['a'])
    assert isinstance(g, FlickerGroupedData)
    df_mean_by_a = g.mean(['b'])
    assert isinstance(df_mean_by_a, FlickerDataFrame)
    assert df_mean_by_a.nrows == 2
    assert df_mean_by_a.ncols == 2
    assert df_mean_by_a.take(None, convert_to_dict=True) == [
        {'a': 1, 'avg(b)': 2.0},
        {'a': 2, 'avg(b)': 3.0}
    ]


def test_grouped_agg(spark):
    df = FlickerDataFrame.from_dict(spark, {'a': [1, 2, 1, 2], 'b': [1.0, 2.0, 3.0, 4.0]})
    g = df.groupby(['a'])
    assert isinstance(g, FlickerGroupedData)
    df_agg = g.agg([{'b': 'min'}])
    assert isinstance(df_agg, FlickerDataFrame)
    assert df_agg.nrows == 2
    assert df_agg.ncols == 2
    assert df_agg.take(None, convert_to_dict=True) == [
        {'a': 1, 'min(b)': 1.0},
        {'a': 2, 'min(b)': 2.0}
    ]
