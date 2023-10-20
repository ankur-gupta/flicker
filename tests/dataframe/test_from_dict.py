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
import numpy as np
from flicker import FlickerDataFrame


def test_basic_usage(spark):
    data = {'a': [0, 1], 'b': [3.4, 5.6], 'c': ['a', 'b']}
    df = FlickerDataFrame.from_dict(spark, data)
    assert isinstance(df, FlickerDataFrame)
    assert df.ncols == len(data)
    assert df.nrows == 2
    assert df.shape == (2, len(data))
    assert set(df.names) == set(data.keys())
    for name in df.names:
        actual = df[[name]].to_pandas()[name].to_numpy()
        expected = np.array(data[name])
        assert np.all(actual == expected)


def test_unequal_rows(spark):
    data = {'a': [0, 1], 'b': [3.4, 5.6, 6.7], 'c': ['a', 'b']}
    with pytest.raises(Exception):
        FlickerDataFrame.from_dict(spark, data)


def test_nones_have_no_effect_in_non_float_columns(spark):
    data = {'n': [1, None, 3, 4], 's': ['hello', 'spark', 'flicker', None]}

    df = FlickerDataFrame.from_dict(spark, data, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    assert df['n'].take(None) == data['n']
    assert df['s'].take(None) == data['s']

    df = FlickerDataFrame.from_dict(spark, data, nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    assert df['n'].take(None) == data['n']
    assert df['s'].take(None) == data['s']


def test_nans_have_some_effect_in_non_float_columns(spark):
    data_with_nones = {'n': [1, None, 3, 4], 's': ['hello', 'spark', 'flicker', None]}
    data_with_nans = {'n': [1, np.nan, 3, 4], 's': ['hello', 'spark', 'flicker', np.nan]}

    df = FlickerDataFrame.from_dict(spark, data_with_nans, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    assert df['n'].take(None) == data_with_nones['n']
    assert df['s'].take(None) == data_with_nones['s']

    with pytest.raises(Exception):
        FlickerDataFrame.from_dict(spark, data_with_nans, nan_to_none=False)


def test_nones_have_no_effect_in_float_columns(spark):
    data_with_nones = {'x': [1.0, None, 3.0, 4.0], 'y': [3.0, 4.0, 5.0, None]}

    df = FlickerDataFrame.from_dict(spark, data_with_nones, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    assert df['x'].take(None) == data_with_nones['x']
    assert df['y'].take(None) == data_with_nones['y']

    df = FlickerDataFrame.from_dict(spark, data_with_nones, nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    assert df['x'].take(None) == data_with_nones['x']
    assert df['y'].take(None) == data_with_nones['y']


def test_nans_have_some_effect_in_float_columns(spark):
    data_with_nones = {'x': [1.0, None, 3.0, 4.0], 'y': [3.0, 4.0, 5.0, None]}
    data_with_nans = {'x': [1.0, np.nan, 3.0, 4.0], 'y': [3.0, 4.0, 5.0, np.nan]}

    df = FlickerDataFrame.from_dict(spark, data_with_nans, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    assert df['x'].take(None) == data_with_nones['x']
    assert df['y'].take(None) == data_with_nones['y']

    df = FlickerDataFrame.from_dict(spark, data_with_nans, nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    actual_data = {'x': df['x'].take(None), 'y': df['y'].take(None)}
    assert actual_data.keys() == data_with_nans.keys()
    for name in ['x', 'y']:
        for actual_value, expected_value in zip(actual_data[name], data_with_nans[name]):
            if np.isnan(expected_value):
                assert np.isnan(actual_value)
            else:
                assert actual_value == expected_value


def test_nans_and_nones_in_float_columns(spark):
    data_with_nans_and_nones = {'x': [1.0, None, np.nan, 4.0], 'y': [None, 4.0, 5.0, np.nan]}
    data_with_all_nones = {'x': [1.0, None, None, 4.0], 'y': [None, 4.0, 5.0, None]}

    df = FlickerDataFrame.from_dict(spark, data_with_nans_and_nones, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    assert df['x'].take(None) == data_with_all_nones['x']
    assert df['y'].take(None) == data_with_all_nones['y']

    df = FlickerDataFrame.from_dict(spark, data_with_nans_and_nones, nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    actual_data = {'x': df['x'].take(None), 'y': df['y'].take(None)}
    assert actual_data.keys() == data_with_nans_and_nones.keys()
    for name in ['x', 'y']:
        for actual_value, expected_value in zip(actual_data[name], data_with_nans_and_nones[name]):
            if expected_value is None:
                assert actual_value is None
            elif np.isnan(expected_value):
                assert np.isnan(actual_value)
            else:
                assert actual_value == expected_value
