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
from flicker import FlickerDataFrame


def test_groupby_empty(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, None],
        'b': ['a', 'v', 'r', None, 't'],
        'c': [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    assert df.groupby().count().count() == 1


def test_groupby_single_name(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, None],
        'b': ['a', 'v', 'r', None, 't'],
        'c': [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    # Number of distinct items in a column
    assert df.groupby('a').count().count() == 4
    assert df.groupby(['a']).count().count() == 4
    assert df.groupby('b').count().count() == 5
    assert df.groupby(['b']).count().count() == 5
    assert df.groupby('c').count().count() == 1
    assert df.groupby(['c']).count().count() == 1


def test_groupby_single_column(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, None],
        'b': ['a', 'v', 'r', None, 't'],
        'c': [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    # Number of distinct items in a column
    assert df.groupby(df['a']).count().count() == 4
    assert df.groupby([df['a']]).count().count() == 4
    assert df.groupby(df['b']).count().count() == 5
    assert df.groupby([df['b']]).count().count() == 5
    assert df.groupby(df['c']).count().count() == 1
    assert df.groupby([df['c']]).count().count() == 1


def test_groupby_mulitple_names(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 1, 2, 2, None, None],
        'b': ['a', 'b', 'a', 'b', 'a', 'b'],
        'c': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    })
    assert df.groupby('a', 'b').count().count() == 6
    assert df.groupby(['a', 'b']).count().count() == 6
    assert df.groupby('b', 'c').count().count() == 2
    assert df.groupby(['b', 'c']).count().count() == 2
    assert df.groupby('a', 'c').count().count() == 3
    assert df.groupby(['a', 'c']).count().count() == 3
    assert df.groupby('a', 'b', 'c').count().count() == 6
    assert df.groupby(['a', 'b', 'c']).count().count() == 6


def test_groupby_mulitple_columns(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 1, 2, 2, None, None],
        'b': ['a', 'b', 'a', 'b', 'a', 'b'],
        'c': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    })
    assert df.groupby(df['a'], df['b']).count().count() == 6
    assert df.groupby([df['a'], df['b']]).count().count() == 6
    assert df.groupby(df['b'], df['c']).count().count() == 2
    assert df.groupby([df['b'], df['c']]).count().count() == 2
    assert df.groupby(df['a'], df['c']).count().count() == 3
    assert df.groupby([df['a'], df['c']]).count().count() == 3
    assert df.groupby(df['a'], df['b'], df['c']).count().count() == 6
    assert df.groupby([df['a'], df['b'], df['c']]).count().count() == 6


def test_groupby_names_and_columns(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 1, 2, 2, None, None],
        'b': ['a', 'b', 'a', 'b', 'a', 'b'],
        'c': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    })
    assert df.groupby('a', df['b']).count().count() == 6
    assert df.groupby(df['a'], 'b').count().count() == 6
    assert df.groupby('b', df['c']).count().count() == 2
    assert df.groupby(df['b'], 'c').count().count() == 2
    assert df.groupby('a', df['c']).count().count() == 3
    assert df.groupby(df['a'], 'c').count().count() == 3
    assert df.groupby('a', df['b'], df['c']).count().count() == 6
