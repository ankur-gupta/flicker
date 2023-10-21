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
import pandas as pd
import numpy as np
from flicker import FlickerDataFrame


def test_basic_usage(spark):
    records = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 25}, {'name': 'Charlie', 'age': 25}]
    pdf = pd.DataFrame(records)
    df = FlickerDataFrame.from_pandas(spark, pdf)
    assert df.shape == pdf.shape
    assert set(df.names) == set(pdf.columns)
    bool_df = df.to_pandas() == pdf
    for name in bool_df.columns:
        assert all(bool_df[name])


def test_duplicated_names_failure(spark):
    pdf = pd.DataFrame({'a': [0, 1], 'b': [3.4, 5.6]})
    pdf_duplicate_names = pd.concat([pdf, pdf], axis=1)
    with pytest.raises(Exception):
        FlickerDataFrame.from_pandas(spark, pdf_duplicate_names)


def test_nones_have_no_effect_in_non_float_columns(spark):
    pdf = pd.DataFrame({
        'n': [1, 2, 3, 4],
        's': ['spark', 'b', 'hello', None]
    })
    pdf['n'] = pdf['n'].astype(object)  # this is needed to prevent pandas from coercing int to float
    pdf.iloc[1, 0] = None

    df = FlickerDataFrame.from_pandas(spark, pdf, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == pdf.shape
    assert pdf.to_dict(orient='list') == df.to_dict(None)

    df = FlickerDataFrame.from_pandas(spark, pdf, nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == pdf.shape
    assert pdf.to_dict(orient='list') == df.to_dict(None)


def test_nans_have_some_effect_in_non_float_columns(spark):
    pdf_with_nones = pd.DataFrame({
        'n': [1, 2, 3, 4],
        's': ['spark', 'b', 'hello', None]
    })
    pdf_with_nones['n'] = pdf_with_nones['n'].astype(object)  # prevent pandas from coercing int to float
    pdf_with_nones.iloc[1, 0] = None

    df = FlickerDataFrame.from_pandas(spark, pdf_with_nones, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == pdf_with_nones.shape
    assert pdf_with_nones.to_dict(orient='list') == df.to_dict(None)

    pdf_with_nans = pd.DataFrame({
        'n': [1, 2, 3, 4],
        's': ['spark', 'b', 'hello', np.nan]
    })
    pdf_with_nans['n'] = pdf_with_nans['n'].astype(object)  # prevent pandas from coercing int to float
    pdf_with_nans.iloc[1, 0] = np.nan

    with pytest.raises(Exception):
        FlickerDataFrame.from_pandas(spark, pdf_with_nans, nan_to_none=False)


def test_nones_have_no_effect_in_float_columns(spark):
    pdf_with_nones = pd.DataFrame({
        'x': [1.0, 2.0, 3.0, 4.0],
        'y': [3.0, 4.0, 5.0, 2.0]
    })
    pdf_with_nones['x'] = pdf_with_nones['x'].astype(object)  # prevent pandas from coercing None to nan
    pdf_with_nones['y'] = pdf_with_nones['y'].astype(object)  # prevent pandas from coercing None to nan
    pdf_with_nones.iloc[1, 0] = None
    pdf_with_nones.iloc[-1, 1] = None

    df = FlickerDataFrame.from_pandas(spark, pdf_with_nones, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == pdf_with_nones.shape
    assert pdf_with_nones.to_dict(orient='list') == df.to_dict(None)

    df = FlickerDataFrame.from_pandas(spark, pdf_with_nones, nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == pdf_with_nones.shape
    assert pdf_with_nones.to_dict(orient='list') == df.to_dict(None)


def test_nans_have_some_effect_in_float_columns(spark):
    pdf_with_nones = pd.DataFrame({
        'x': [1.0, 2.0, 3.0, 4.0],
        'y': [3.0, 4.0, 5.0, 2.0]
    })
    pdf_with_nones['x'] = pdf_with_nones['x'].astype(object)  # prevent pandas from coercing None to nan
    pdf_with_nones['y'] = pdf_with_nones['y'].astype(object)  # prevent pandas from coercing None to nan
    pdf_with_nones.iloc[1, 0] = None
    pdf_with_nones.iloc[-1, 1] = None

    pdf_with_nans = pd.DataFrame({
        'x': [1.0, 2.0, 3.0, 4.0],
        'y': [3.0, 4.0, 5.0, 2.0]
    })
    pdf_with_nans.iloc[1, 0] = np.nan
    pdf_with_nans.iloc[-1, 1] = np.nan

    df = FlickerDataFrame.from_pandas(spark, pdf_with_nans, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == pdf_with_nans.shape
    assert pdf_with_nones.to_dict(orient='list') == df.to_dict(None)

    df = FlickerDataFrame.from_pandas(spark, pdf_with_nans, nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == pdf_with_nans.shape
    expected = pdf_with_nans.to_dict(orient='list')
    actual = df.to_dict(None)
    assert expected.keys() == actual.keys()
    for name in expected.keys():
        for expected_value, actual_value in zip(expected[name], actual[name]):
            if np.isnan(expected_value):
                assert np.isnan(actual_value)
            else:
                assert actual_value == expected_value


def test_nans_and_nones_in_float_columns(spark):
    pdf_with_nans_and_nones = pd.DataFrame({
        'x': [1.0, None, np.nan, 4.0],
        'y': [None, 4.0, 5.0, np.nan]
    })
    pdf_with_nans_and_nones['x'] = pdf_with_nans_and_nones['x'].astype(object)  # prevent coercion int to float
    pdf_with_nans_and_nones['y'] = pdf_with_nans_and_nones['y'].astype(object)  # prevent coercion int to float
    pdf_with_nans_and_nones.iloc[1, 0] = None
    pdf_with_nans_and_nones.iloc[0, 1] = None
    pdf_with_nans_and_nones.iloc[2, 0] = np.nan
    pdf_with_nans_and_nones.iloc[-1, 1] = np.nan

    pdf_with_all_nones = pdf_with_nans_and_nones.copy(deep=True)
    pdf_with_all_nones.iloc[2, 0] = None
    pdf_with_all_nones.iloc[-1, 1] = None

    df = FlickerDataFrame.from_pandas(spark, pdf_with_nans_and_nones, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == pdf_with_nans_and_nones.shape
    assert pdf_with_all_nones.to_dict(orient='list') == df.to_dict(None)

    df = FlickerDataFrame.from_pandas(spark, pdf_with_nans_and_nones, nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == pdf_with_nans_and_nones.shape
    expected = pdf_with_nans_and_nones.to_dict(orient='list')
    actual = df.to_dict(None)
    assert expected.keys() == actual.keys()
    for name in expected.keys():
        for expected_value, actual_value in zip(expected[name], actual[name]):
            if expected_value is None:
                assert actual_value is None
            elif np.isnan(expected_value):
                assert np.isnan(actual_value)
            else:
                assert actual_value == expected_value
