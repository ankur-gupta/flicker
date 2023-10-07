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
from flicker import FlickerDataFrame


def test_valid(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    out = df.rename({'a': 'g'})
    assert isinstance(out, FlickerDataFrame)
    assert set(out.names) == set(list('gbcdef'))


def test_valid_two(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    out = df.rename({'a': 'g', 'b': 'h'})
    assert isinstance(out, FlickerDataFrame)
    assert set(out.names) == set(list('ghcdef'))


def test_duplicate_name(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    with pytest.raises(Exception):
        df.rename({'a': 'f'})


def test_duplicate_name_two(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    with pytest.raises(Exception):
        df.rename({'a': 'g', 'b': 'f'})


def test_duplicate_name_three(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    with pytest.raises(Exception):
        df.rename({'a': 'g', 'b': 'g'})
    with pytest.raises(Exception):
        df.rename({'a': 'g', 'b': 'g', 'f': 'h'})


def test_invalid_to_name_type(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    with pytest.raises(Exception):
        df.rename({'a': 1})


def test_invalid_from_name(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    with pytest.raises(Exception):
        df.rename({'unknown-name': 'f'})
    with pytest.raises(Exception):
        df.rename({1: 'f'})
