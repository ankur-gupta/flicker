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
    records = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 25}, {'name': 'Charlie', 'age': 25}]
    df = FlickerDataFrame.from_records(spark, records)
    assert isinstance(df, FlickerDataFrame)
    assert set(df.names) == {'name', 'age'}
    assert df.shape == (3, 2)
    for name in df.names:
        assert isinstance(name, str)


def test_column_number_mismatch(spark):
    records = [{'name': 'Alice', 'age': 25}, {'name': 'Bob'}, {'name': 'Charlie', 'age': 25}]
    df = FlickerDataFrame.from_records(spark, records)
    assert isinstance(df, FlickerDataFrame)
    assert set(df.names) == {'name', 'age'}
    assert df.shape == (3, 2)
    for name in df.names:
        assert isinstance(name, str)


def test_nones_have_no_effect_in_non_float_columns(spark):
    records = [{'n': 1, 's': 'spark'}, {'n': None, 's': 'b'}, {'n': 3, 's': 'hello'}, {'n': 4, 's': None}]

    df = FlickerDataFrame.from_records(spark, records, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    assert records == df.take(None, convert_to_dict=True)

    df = FlickerDataFrame.from_records(spark, records, nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    assert records == df.take(None, convert_to_dict=True)


def test_nans_have_some_effect_in_non_float_columns(spark):
    records_with_nones = [{'n': 1, 's': 'spark'}, {'n': None, 's': 'b'}, {'n': 3, 's': 'hello'}, {'n': 4, 's': None}]
    records_with_nans = [{'n': 1, 's': 'spark'}, {'n': np.nan, 's': 'b'}, {'n': 3, 's': 'hello'}, {'n': 4, 's': np.nan}]

    df = FlickerDataFrame.from_records(spark, records_with_nans, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    assert records_with_nones == df.take(None, convert_to_dict=True)

    with pytest.raises(Exception):
        FlickerDataFrame.from_records(spark, records_with_nans, nan_to_none=False)


def test_nones_have_no_effect_in_float_columns(spark):
    records_with_nones = [{'x': 1.0, 'y': 3.0}, {'x': None, 'y': 4.0}, {'x': 3.0, 'y': 5.0}, {'x': 4.0, 'y': None}]

    df = FlickerDataFrame.from_records(spark, records_with_nones, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    assert records_with_nones == df.take(None, convert_to_dict=True)

    df = FlickerDataFrame.from_records(spark, records_with_nones, nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    assert records_with_nones == df.take(None, convert_to_dict=True)


def test_nans_have_some_effect_in_float_columns(spark):
    records_with_nones = [{'x': 1.0, 'y': 3.0}, {'x': None, 'y': 4.0}, {'x': 3.0, 'y': 5.0}, {'x': 4.0, 'y': None}]
    records_with_nans = [{'x': 1.0, 'y': 3.0}, {'x': np.nan, 'y': 4.0}, {'x': 3.0, 'y': 5.0}, {'x': 4.0, 'y': np.nan}]

    df = FlickerDataFrame.from_records(spark, records_with_nans, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    assert records_with_nones == df.take(None, convert_to_dict=True)

    df = FlickerDataFrame.from_records(spark, records_with_nans, nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    actual_records = df.take(None, convert_to_dict=True)
    assert len(actual_records) == len(records_with_nans)
    for expected_record, actual_record in zip(records_with_nans, actual_records):
        for name in ['x', 'y']:
            if np.isnan(expected_record[name]):
                assert np.isnan(actual_record[name])
            else:
                assert actual_record[name] == expected_record[name]


def test_nans_and_nones_in_float_columns(spark):
    records_with_nans_and_nones = [{'x': 1.0, 'y': np.nan}, {'x': None, 'y': 4.0}, {'x': np.nan, 'y': 5.0},
                                   {'x': 4.0, 'y': None}]
    records_with_all_nones = [{'x': 1.0, 'y': None}, {'x': None, 'y': 4.0}, {'x': None, 'y': 5.0},
                              {'x': 4.0, 'y': None}]

    df = FlickerDataFrame.from_records(spark, records_with_nans_and_nones, nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    assert records_with_all_nones == df.take(None, convert_to_dict=True)

    df = FlickerDataFrame.from_records(spark, records_with_nans_and_nones, nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    actual_records = df.take(None, convert_to_dict=True)
    assert len(actual_records) == len(records_with_nans_and_nones)
    for expected_record, actual_record in zip(records_with_nans_and_nones, actual_records):
        for name in ['x', 'y']:
            if expected_record[name] is None:
                assert actual_record[name] is None
            elif np.isnan(expected_record[name]):
                assert np.isnan(actual_record[name])
            else:
                assert actual_record[name] == expected_record[name]
