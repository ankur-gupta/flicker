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
from flicker import FlickerDataFrame, FlickerColumn


def test_by_name(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=2, ncols=2, names=['a', 'b'], fill='zero')
    assert isinstance(df['a'], FlickerColumn)


def test_by_spark_column(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=2, ncols=2, names=['a', 'b'], fill='zero')
    df['c'] = False
    subset_df = df[df._df['c']]
    assert isinstance(subset_df, FlickerDataFrame)
    assert subset_df.nrows == 0


def test_by_flicker_column(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=2, ncols=2, names=['a', 'b'], fill='zero')
    df['c'] = False
    subset_df = df[df['c']]
    assert isinstance(subset_df, FlickerDataFrame)
    assert subset_df.nrows == 0


def test_by_single_slice(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=2, ncols=6, names=['a', 'b', 'c', 'd', 'e', 'f'], fill='zero')
    assert isinstance(df[:], FlickerDataFrame)
    assert df[:].ncols == df.ncols

    assert isinstance(df[1:], FlickerDataFrame)
    assert df[1:].ncols == (df.ncols - 1)

    assert isinstance(df[2:], FlickerDataFrame)
    assert df[2:].ncols == (df.ncols - 2)

    assert isinstance(df[:1], FlickerDataFrame)
    assert df[:1].ncols == 1

    assert isinstance(df[:2], FlickerDataFrame)
    assert df[:2].ncols == 2

    assert isinstance(df[:df.ncols], FlickerDataFrame)
    assert df[:df.ncols].ncols == df.ncols

    if df.ncols < 10:
        assert isinstance(df[:10], FlickerDataFrame)
        assert df[:10].ncols == df.ncols

    assert isinstance(df[::1], FlickerDataFrame)
    assert df[::1].ncols == df.ncols

    assert isinstance(df[::2], FlickerDataFrame)
    assert df[::2].ncols == (df.ncols // 2)

    assert isinstance(df[::-1], FlickerDataFrame)
    assert df[::-1].ncols == df.ncols

    bool_df = df.to_pandas() == df[:].to_pandas()
    for name in bool_df.columns:
        assert all(bool_df[name])


def test_by_tuple_of_slices(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=['a', 'b', 'c', 'd', 'e', 'f'], fill='zero')
    assert isinstance(df[:, :], FlickerDataFrame)
    assert df[:, :].shape == df.shape

    assert isinstance(df[:1, :1], FlickerDataFrame)
    assert df[:1, :1].shape == (1, 1)

    assert isinstance(df[:2, :2], FlickerDataFrame)
    assert df[:2, :2].shape == (2, 2)

    assert isinstance(df[:2, ::2], FlickerDataFrame)
    assert df[:2, ::2].shape == (2, df.ncols // 2)

    assert isinstance(df[:6, ::-1], FlickerDataFrame)
    assert df[:6, ::-1].shape == (6, df.ncols)

    assert isinstance(df[:, ::-1], FlickerDataFrame)
    assert df[:, ::-1].shape == df.shape

    with pytest.raises(Exception):
        _ = df[1:, 1:]

    with pytest.raises(Exception):
        _ = df[1:, :]

    with pytest.raises(Exception):
        _ = df[1:, ::3]

    with pytest.raises(Exception):
        _ = df[1:3, ::3]

    with pytest.raises(Exception):
        _ = df[1:3:6, :3]

    with pytest.raises(Exception):
        _ = df[:3:6, :3]

    with pytest.raises(Exception):
        _ = df[::6, :3]


def test_by_list_of_names(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=['a', 'b', 'c', 'd', 'e', 'f'], fill='zero')
    assert isinstance(df[['a']], FlickerDataFrame)
    assert set(df[['a']].names) == {'a'}
    assert df[['a']].shape == (df.nrows, 1)

    assert isinstance(df[['a', 'b']], FlickerDataFrame)
    assert set(df[['a', 'b']].names) == {'a', 'b'}
    assert df[['a', 'b']].shape == (df.nrows, 2)

    assert isinstance(df[[]], FlickerDataFrame)
    assert set(df[[]].names) == set()
    assert df[[]].shape == (df.nrows, 0)


def test_by_list_of_columns(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=['a', 'b', 'c', 'd', 'e', 'f'], fill='zero')
    out = df[[df['a']]]
    assert isinstance(out, FlickerDataFrame)
    assert set(out.names) == {'a'}
    assert out.shape == (df.nrows, 1)


def test_unsupported_index(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=['a', 'b', 'c', 'd', 'e', 'f'], fill='zero')
    with pytest.raises(Exception):
        df[[{}]]
    with pytest.raises(Exception):
        df[{}, {}]
    with pytest.raises(Exception):
        df[:, :, :]
    with pytest.raises(Exception):
        df[None]

