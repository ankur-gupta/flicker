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


def test_drop_single_valid_name(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    assert isinstance(df.drop(['a']), FlickerDataFrame)
    assert df.drop(['a']).shape == (df.nrows, df.ncols - 1)
    assert set(df.drop(['a']).names) == set(list('bcdef'))

    df1 = df.drop(['a'])
    assert isinstance(df1.drop(['b']), FlickerDataFrame)
    assert df1.drop(['b']).shape == (df.nrows, df.ncols - 2)
    assert set(df1.drop(['b']).names) == set(list('cdef'))

    df2 = df1.drop(['b'])
    assert isinstance(df2.drop(['f']), FlickerDataFrame)
    assert df2.drop(['f']).shape == (df.nrows, df.ncols - 3)
    assert set(df2.drop(['f']).names) == set(list('cde'))


def test_drop_multiple_valid_names(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    out = df.drop(['a', 'f'])
    assert isinstance(out, FlickerDataFrame)
    assert out.shape == (df.nrows, df.ncols - 2)
    assert set(out.names) == set(list('bcde'))


def test_drop_single_invalid_name(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    with pytest.raises(Exception):
        df.drop(['unknown'])


def test_drop_multiple_invalid_names(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    with pytest.raises(Exception):
        df.drop(['unknown1', 'unknown2'])


def test_drop_mix_of_valid_and_invalid_names(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    with pytest.raises(Exception):
        df.drop(['a', 'unknown'])
    with pytest.raises(Exception):
        df.drop(['a', 'unknown', 'b'])
    with pytest.raises(Exception):
        df.drop(['a', 'b', 'unknown', 'f'])
