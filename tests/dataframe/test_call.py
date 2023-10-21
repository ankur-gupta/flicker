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
import numpy as np
import pandas as pd
import pytest

from flicker import FlickerDataFrame


def test_call_default(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    assert isinstance(df(), pd.DataFrame)
    assert df().shape[1] == df.shape[1]


def test_call_n_int(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    assert isinstance(df(2), pd.DataFrame)
    assert df(2).shape == (2, df.ncols)


def test_call_n_none(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    assert isinstance(df(None), pd.DataFrame)
    assert df(None).shape == df.shape


def test_call_dont_use_pandas_dtypes(spark):
    data = {
        'double_no_null': (1.2, 3.98, 4.5, -0.58, 0.0),
        'double_with_null': (1.2, 3.98, None, -0.58, 0.0),
        'int_no_null': (1, 2, 3, 4, 5),
        'int_with_null': (1, None, 3, 4, None),
    }
    df = FlickerDataFrame.from_dict(spark, data)

    pdf = df(use_pandas_dtypes=False)
    assert isinstance(pdf, pd.DataFrame)
    assert all(pdf.dtypes == object)

    pdf = df(n=2, use_pandas_dtypes=False)
    assert isinstance(pdf, pd.DataFrame)
    assert pdf.shape == (2, df.ncols)
    assert all(pdf.dtypes == object)


def test_call_use_pandas_dtypes(spark):
    data = {
        'double_no_null': (1.2, 3.98, 4.5, -0.58, 0.0),
        'double_with_null': (1.2, 3.98, None, -0.58, 0.0),
        'int_no_null': (1, 2, 3, 4, 5),
        'int_with_null': (1, None, 3, 4, None),
    }
    df = FlickerDataFrame.from_dict(spark, data)

    pdf = df(use_pandas_dtypes=True)
    assert isinstance(pdf, pd.DataFrame)

    pdf = df(n=2, use_pandas_dtypes=True)
    assert isinstance(pdf, pd.DataFrame)
    assert pdf.shape == (2, df.ncols)


def test_call_with_zero_n(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'])
    df(0)  # ensure that this doesn't fail
    assert isinstance(df(0), pd.DataFrame)
    assert df(0).shape == (0, 2)
    assert set(df(0).columns) == set(df.names)


def test_call_empty_dataframe(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'])
    df['c'] = False

    subset_df = df[df['c']]
    assert subset_df.nrows == 0
    subset_df()  # ensure that this doesn't fail
    assert isinstance(subset_df(), pd.DataFrame)
    assert subset_df().shape == (0, 3)
    assert set(subset_df().columns) == set(subset_df.names)


def test_nans_and_nones_in_float_columns(spark):
    data_with_nans_and_nones = {'x': [1.0, None, np.nan, 4.0], 'y': [None, 4.0, 5.0, np.nan]}
    expected = pd.DataFrame.from_dict(data_with_nans_and_nones, dtype=object)
    for name in expected.columns:
        assert expected[name].dtype == object
    df = FlickerDataFrame.from_dict(spark, data_with_nans_and_nones, nan_to_none=False)
    actual = df(None)
    assert all(expected.columns == actual.columns)
    for name in expected.columns:
        for expected_value, actual_value in zip(expected[name], actual[name]):
            if expected_value is None:
                assert actual_value is None
            elif np.isnan(expected_value):
                assert np.isnan(actual_value)
            else:
                assert actual_value == expected_value


def test_only_nones_in_float_columns(spark):
    data_only_nones = {'x': [1.0, None, None, 4.0], 'y': [None, 4.0, 5.0, None]}
    expected = pd.DataFrame.from_dict(data_only_nones, dtype=object)
    for name in expected.columns:
        assert expected[name].dtype == object
    df = FlickerDataFrame.from_dict(spark, data_only_nones, nan_to_none=False)
    actual = df(None)
    assert all(expected.columns == actual.columns)
    for name in expected.columns:
        for expected_value, actual_value in zip(expected[name], actual[name]):
            if expected_value is None:
                assert actual_value is None
            elif np.isnan(expected_value):
                assert np.isnan(actual_value)
            else:
                assert actual_value == expected_value


def test_nones_non_float_columns(spark):
    data_only_nones = {'n': [1, None, 3, 4], 's': ['hello', 'spark', 'flicker', None]}
    expected = pd.DataFrame.from_dict(data_only_nones, dtype=object)
    for name in expected.columns:
        assert expected[name].dtype == object
    df = FlickerDataFrame.from_dict(spark, data_only_nones, nan_to_none=False)
    actual = df(None)
    assert all(expected.columns == actual.columns)
    for name in expected.columns:
        for expected_value, actual_value in zip(expected[name], actual[name]):
            if expected_value is None:
                assert actual_value is None
            else:
                assert actual_value == expected_value


def test_nans_in_non_float_columns(spark):
    data_only_nans = {'n': [1, np.nan, 3, 4], 's': ['hello', 'spark', 'flicker', np.nan]}
    expected = pd.DataFrame.from_dict(data_only_nans, dtype=object)
    for name in expected.columns:
        assert expected[name].dtype == object
    with pytest.raises(Exception):
        FlickerDataFrame.from_dict(spark, data_only_nans, nan_to_none=False)
