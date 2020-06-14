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
from datetime import datetime

import pytest
import numpy as np
from flicker import FlickerDataFrame


def test_empty_dataframe(spark):
    df = FlickerDataFrame(spark.createDataFrame([], 'a TIMESTAMP'))
    assert df.max('a') is None
    assert df.min('a') is None


def test_errors(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [True, True, False, True, None]
    })
    with pytest.raises(KeyError):
        df.isnull('non-existent-column')
    with pytest.raises(KeyError):
        df.isnull('')
    with pytest.raises(KeyError):
        df.isnull('aa')
    with pytest.raises(TypeError):
        df._validate_column_name(1)
    with pytest.raises(TypeError):
        df._validate_column_name(1.0)
    with pytest.raises(TypeError):
        df._validate_column_name(None)
    with pytest.raises(TypeError):
        df._validate_column_name(True)
    with pytest.raises(TypeError):
        df._validate_column_name([])


def test_boolean_column(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [True, True, False, True, False],
        'b': [None, True, None, True, False]
    })
    assert df.max('a') is True
    assert df.min('a') is False

    # None(s) are ignored
    assert df.max('b') is True
    assert df.min('b') is False


def test_string_column(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': ['b', 'a', 'c', 'd'],
        'b': ['b', None, 'c', None]
    })
    assert df.max('a') == 'd'
    assert df.min('a') == 'a'

    # None(s) are ignored
    assert df.max('b') == 'c'
    assert df.min('b') == 'b'


def test_int_column(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [5, 3, 2, 1, 8, -1]
    })
    assert df.max('a') == 8
    assert df.min('a') == -1


def test_datetime_column(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': ['2016-04-06 14:36', '2016-05-06 16:36', '2016-04-06 14:35'],
        'b': ['2016-04-06 14:36', None, None],
    })
    df['a'] = df['a'].cast('timestamp')
    df['b'] = df['b'].cast('timestamp')

    assert df.max('a') == datetime(2016, 5, 6, 16, 36)
    assert df.min('a') == datetime(2016, 4, 6, 14, 35)

    # None(s) are ignored
    assert df.max('b') == datetime(2016, 4, 6, 14, 36)
    assert df.min('b') == datetime(2016, 4, 6, 14, 36)


def test_double_column(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1.4, 5.6, 9.3, -0.9],
        'b': [1.4, 5.6, np.nan, np.nan]
    })
    assert df.max('a') == 9.3
    assert df.min('a') == -0.9
    assert df.max('b', ignore_nan=True) == 5.6
    assert df.min('b', ignore_nan=True) == 1.4

    # These are weird because max() returns np.nan but min() returns 1.4.
    # Comparisons across np.nan are unreliable. This is why we don't
    # check these.
    # assert np.isnan(df.max('b', ignore_nan=False))
    # assert np.isnan(df.min('b', ignore_nan=False))
