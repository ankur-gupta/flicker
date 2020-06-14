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


def test_non_boolean_empty_column(spark):
    df = FlickerDataFrame(spark.createDataFrame([], 'a INT'))
    with pytest.raises(TypeError):
        df.all('a')


def test_boolean_empty_column(spark):
    df = FlickerDataFrame(spark.createDataFrame([], 'a BOOLEAN'))
    assert df.all('a') is True


def test_non_boolean_non_empty_column(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3],
        'b': ['True', 'True', 'False'],
        'c': [1, 1, 1],
        'd': [1.0, 1.0, 1.0]
    })
    with pytest.raises(TypeError):
        df.all('a')
    with pytest.raises(TypeError):
        df.all('b')
    with pytest.raises(TypeError):
        df.all('c')
    with pytest.raises(TypeError):
        df.all('d')


def test_boolean_none_only_column(spark):
    df = FlickerDataFrame(spark.createDataFrame(
        [(None,), (None,), (None,)], 'a BOOLEAN'))
    assert df.all('a') is False
    assert df.all('a', ignore_null=True) is True


def test_boolean_true_only_column(spark):
    df = FlickerDataFrame(spark.createDataFrame(
        [(True,), (True,), (True,)], 'a BOOLEAN'))
    assert df.all('a') is True
    assert df.all('a', ignore_null=True) is True


def test_boolean_false_only_column(spark):
    df = FlickerDataFrame(spark.createDataFrame(
        [(False,), (False,), (False,)], 'a BOOLEAN'))
    assert df.all('a') is False
    assert df.all('a', ignore_null=True) is False


def test_boolean_true_and_none_column(spark):
    df = FlickerDataFrame(spark.createDataFrame(
        [(True,), (None,), (True,)], 'a BOOLEAN'))
    assert df.all('a') is False
    assert df.all('a', ignore_null=True) is True


def test_boolean_false_and_none_column(spark):
    df = FlickerDataFrame(spark.createDataFrame(
        [(None,), (None,), (False,)], 'a BOOLEAN'))
    assert df.all('a') is False
    assert df.all('a', ignore_null=True) is False


def test_boolean_true_and_false_column(spark):
    df = FlickerDataFrame(spark.createDataFrame(
        [(True,), (True,), (False,)], 'a BOOLEAN'))
    assert df.all('a') is False
    assert df.all('a', ignore_null=True) is False


def test_general(spark):
    df = FlickerDataFrame(spark.createDataFrame(
        [(True,), (None,), (False,)], 'a BOOLEAN'))
    assert df.all('a') is False
    assert df.all('a', ignore_null=True) is False
