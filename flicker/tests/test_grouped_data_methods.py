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

import pytest
import six

from flicker import FlickerDataFrame, FlickerGroupedData


def test_repr_str(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, None],
        'b': ['a', 'v', 'r', None, 't'],
        'c': [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    assert isinstance(repr(df), six.string_types)
    assert isinstance(str(df), six.string_types)


def test_groupby_count(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, None],
        'b': ['a', 'v', 'r', None, 't'],
        'c': [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    grouped = df.groupby()
    assert isinstance(grouped, FlickerGroupedData)

    grouped_count = grouped.count()
    assert isinstance(grouped_count, FlickerDataFrame)
    assert grouped_count[['count']].collect()[0][0] == df.nrows


def test_groupby_sum(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, 1],
        'b': [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    grouped = df.groupby(['a'])
    assert isinstance(grouped, FlickerGroupedData)

    grouped_sum = grouped.sum('b')
    assert isinstance(grouped_sum, FlickerDataFrame)
    assert set(grouped_sum.names) == {'a', 'sum(b)'}
    grouped_sum_dict = {
        a: sum_b
        for a, sum_b in grouped_sum.collect()
    }
    assert grouped_sum_dict == {1: 3.0, 2: 1.0, 3: 1.0}


def test_groupby_max(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, 1],
        'b': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    grouped = df.groupby(['a'])
    assert isinstance(grouped, FlickerGroupedData)

    grouped_max = grouped.max('b')
    assert isinstance(grouped_max, FlickerDataFrame)
    assert set(grouped_max.names) == {'a', 'max(b)'}
    grouped_max_dict = {
        a: max_b
        for a, max_b in grouped_max.collect()
    }
    assert grouped_max_dict == {1: 5.0, 2: 2.0, 3: 3.0}


def test_groupby_min(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, 1],
        'b': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    grouped = df.groupby(['a'])
    assert isinstance(grouped, FlickerGroupedData)

    grouped_min = grouped.min('b')
    assert isinstance(grouped_min, FlickerDataFrame)
    assert set(grouped_min.names) == {'a', 'min(b)'}
    grouped_min_dict = {
        a: min_b
        for a, min_b in grouped_min.collect()
    }
    assert grouped_min_dict == {1: 1.0, 2: 2.0, 3: 3.0}


def test_groupby_mean(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, 1],
        'b': [1.0, 2.0, 3.0, 4.0, 4.0]
    })
    grouped = df.groupby(['a'])
    assert isinstance(grouped, FlickerGroupedData)

    grouped_mean = grouped.mean('b')
    assert isinstance(grouped_mean, FlickerDataFrame)
    assert set(grouped_mean.names) == {'a', 'avg(b)'}
    grouped_mean_dict = {
        a: mean_b
        for a, mean_b in grouped_mean.collect()
    }
    assert grouped_mean_dict == {1: 3.0, 2: 2.0, 3: 3.0}


def test_groupby_avg(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, 1],
        'b': [1.0, 2.0, 3.0, 4.0, 4.0]
    })
    grouped = df.groupby(['a'])
    assert isinstance(grouped, FlickerGroupedData)

    grouped_avg = grouped.avg('b')
    assert isinstance(grouped_avg, FlickerDataFrame)
    assert set(grouped_avg.names) == {'a', 'avg(b)'}
    grouped_avg_dict = {
        a: avg_b
        for a, avg_b in grouped_avg.collect()
    }
    assert grouped_avg_dict == {1: 3.0, 2: 2.0, 3: 3.0}


def test_groupby_agg(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, 1],
        'b': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    grouped = df.groupby(['a'])
    assert isinstance(grouped, FlickerGroupedData)

    grouped_agg = grouped.agg({'b': 'max'})
    assert isinstance(grouped_agg, FlickerDataFrame)
    assert set(grouped_agg.names) == {'a', 'max(b)'}
    grouped_agg_dict = {
        a: agg_b
        for a, agg_b in grouped_agg.collect()
    }
    assert grouped_agg_dict == {1: 5.0, 2: 2.0, 3: 3.0}
