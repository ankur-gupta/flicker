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


def test_basic(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 4)
    assert df.shape == (3, 4)


def test_names(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 4, names=list('abcd'))
    assert df.shape == (3, 4)
    assert set(df.names) == set(list('abcd'))


def test_duplicated_names_failure(spark):
    with pytest.raises(Exception):
        FlickerDataFrame.from_shape(spark, 3, 5, names=list('aabcd'))


def test_zero_fill(spark):
    df = FlickerDataFrame.from_shape(spark, 2, 4, names=list('abcd'), fill='zero')
    assert df.shape == (2, 4)
    for name in df.names:
        assert (df[name] == 0).all()


def test_one_fill(spark):
    df = FlickerDataFrame.from_shape(spark, 1, 4, names=list('abcd'), fill='one')
    assert df.shape == (1, 4)
    for name in df.names:
        assert (df[name] == 1).all()


def test_rand_fill(spark):
    df = FlickerDataFrame.from_shape(spark, 10, 4, names=list('abcd'), fill='rand')
    assert df.shape == (10, 4)
    for name in df.names:
        condition = (df[name] >= 0) & (df[name] <= 1)
        assert condition.all()
    for _, dtype in df.dtypes.items():
        assert dtype == 'double'


def test_randn_fill(spark):
    df = FlickerDataFrame.from_shape(spark, 15, 4, names=list('abcd'), fill='randn')
    assert df.shape == (15, 4)
    for _, dtype in df.dtypes.items():
        assert dtype == 'double'


def test_rowseq_fill(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, names=list('ab'), fill='rowseq')
    assert df.shape == (3, 2)
    for _, dtype in df.dtypes.items():
        assert dtype in {'int', 'bigint'}
    assert all(df(None)['a'].values == [0, 2, 4])
    assert all(df(None)['b'].values == [1, 3, 5])


def test_colseq_fill(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, names=list('ab'), fill='colseq')
    assert df.shape == (3, 2)
    for _, dtype in df.dtypes.items():
        assert dtype in {'int', 'bigint'}
    assert all(df(None)['a'].values == [0, 1, 2])
    assert all(df(None)['b'].values == [3, 4, 5])


def test_unsupported_fill(spark):
    with pytest.raises(Exception):
        FlickerDataFrame.from_shape(spark, 100, 4, names=list('ab'), fill='unsupported-fill')
