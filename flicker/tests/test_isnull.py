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
import numpy as np
from pyspark.sql import Column
from flicker import FlickerDataFrame


def test_empty_dataframe(spark):
    df = FlickerDataFrame(spark.createDataFrame([], 'a INT'))
    output = df.isnull('a')
    assert isinstance(output, Column)
    assert df[output].count() == 0


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


def test_boolean_example_0(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [True, True, False, True, False]
    })
    output = df.isnull('a')
    assert isinstance(output, Column)
    assert df[output].count() == 0
    assert df[~output].count() == 5


def test_boolean_example_1(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [True, True, False, True, None]
    })
    output = df.isnull('a')
    assert isinstance(output, Column)
    assert df[output].count() == 1
    assert df[~output].count() == 4


def test_boolean_example_2(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [True, True, None, True, None]
    })
    output = df.isnull('a')
    assert isinstance(output, Column)
    assert df[output].count() == 2
    assert df[~output].count() == 3


def test_boolean_example_1(spark):
    data = [(None,), (None,), (None,), (None,), (None,)]
    df = FlickerDataFrame(spark.createDataFrame(data, 'a BOOLEAN'))
    output = df.isnull('a')
    assert isinstance(output, Column)
    assert df[output].count() == 5
    assert df[~output].count() == 0


def test_double_example_1(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1.0, 3.4, 6.7, np.nan]
    })
    assert df[df.isnull('a')].count() == 0
    assert df[~df.isnull('a')].count() == 4
