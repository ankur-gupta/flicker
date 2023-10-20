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
    rows = [('Alice', 25), ('Bob', 30), ('Charlie', 35)]
    df = FlickerDataFrame.from_rows(spark, rows)
    assert isinstance(df, FlickerDataFrame)
    assert 'name' not in df
    assert 'age' not in df
    assert df.shape == (3, 2)
    for name in df.names:
        assert isinstance(name, str)


def test_with_names(spark):
    rows = [('Alice', 25), ('Bob', 25), ('Charlie', 25)]
    names = ['name', 'age']
    df = FlickerDataFrame.from_rows(spark, rows, names)
    assert isinstance(df, FlickerDataFrame)
    assert set(df.names) == {'name', 'age'}
    assert df.shape == (3, 2)
    assert (df['age'] == 25).all()


def test_duplicated_names_failure(spark):
    rows = [(1, 'spark'), (2, 'b'), (3, 'hello')]
    names = ['a', 'a']
    with pytest.raises(Exception):
        FlickerDataFrame.from_rows(spark, rows, names)


def test_rows_names_mismatch(spark):
    rows = [(1, 'spark'), (2, 'b'), (3, 'hello')]
    names = ['a', 'b', 'c']
    with pytest.raises(Exception):
        FlickerDataFrame.from_rows(spark, rows, names)


def test_unequal_number_of_columns(spark):
    rows = [(1, 'spark'), (2, 'b', 45), (3, 'hello')]
    with pytest.raises(Exception):
        FlickerDataFrame.from_rows(spark, rows)


def test_typical_usage(spark):
    rows = [(1, 'spark'), (2, 'b'), (3, 'hello')]
    expected_first_column = np.array([value[0] for value in rows])
    expected_second_column = np.array([value[1] for value in rows])

    df = FlickerDataFrame.from_rows(spark, rows)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (3, 2)

    first_name = df.names[0]
    first_column = df[[first_name]].to_pandas()[first_name].to_numpy()
    assert np.all(first_column == expected_first_column)

    second_name = df.names[1]
    second_column = df[[second_name]].to_pandas()[second_name].to_numpy()
    assert np.all(second_column == expected_second_column)


def test_nones_have_no_effect_in_non_float_columns(spark):
    rows = [(1, 'spark'), (None, 'b'), (3, 'hello'), (4, None)]

    df = FlickerDataFrame.from_rows(spark, rows, names=['n', 's'], nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    extracted_rows = [
        (row['n'], row['s'])
        for row in df.take(None, convert_to_dict=True)
    ]
    assert rows == extracted_rows

    df = FlickerDataFrame.from_rows(spark, rows, names=['n', 's'], nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    extracted_rows = [
        (row['n'], row['s'])
        for row in df.take(None, convert_to_dict=True)
    ]
    assert rows == extracted_rows


def test_nans_have_some_effect_in_non_float_columns(spark):
    rows_with_nans = [(1, 'spark'), (np.nan, 'b'), (3, 'hello'), (4, np.nan)]
    rows_with_nones = [(1, 'spark'), (None, 'b'), (3, 'hello'), (4, None)]

    df = FlickerDataFrame.from_rows(spark, rows_with_nans, names=['n', 's'], nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    extracted_rows = [
        (row['n'], row['s'])
        for row in df.take(None, convert_to_dict=True)
    ]
    assert rows_with_nones == extracted_rows

    with pytest.raises(Exception):
        FlickerDataFrame.from_rows(spark, rows_with_nans, names=['n', 's'], nan_to_none=False)


def test_nones_have_no_effect_in_float_columns(spark):
    rows_with_nones = [(1.0, 3.0), (None, 4.0), (3.0, 5.0), (4.0, None)]

    df = FlickerDataFrame.from_rows(spark, rows_with_nones, names=['x', 'y'], nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    extracted_rows = [
        (row['x'], row['y'])
        for row in df.take(None, convert_to_dict=True)
    ]
    assert rows_with_nones == extracted_rows

    df = FlickerDataFrame.from_rows(spark, rows_with_nones, names=['x', 'y'], nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    extracted_rows = [
        (row['x'], row['y'])
        for row in df.take(None, convert_to_dict=True)
    ]
    assert rows_with_nones == extracted_rows


def test_nans_have_some_effect_in_float_columns(spark):
    rows_with_nones = [(1.0, 3.0), (None, 4.0), (3.0, 5.0), (4.0, None)]
    rows_with_nans = [(1.0, 3.0), (np.nan, 4.0), (3.0, 5.0), (4.0, np.nan)]

    df = FlickerDataFrame.from_rows(spark, rows_with_nans, names=['x', 'y'], nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    extracted_rows = [
        (row['x'], row['y'])
        for row in df.take(None, convert_to_dict=True)
    ]
    assert rows_with_nones == extracted_rows
    assert rows_with_nans != extracted_rows

    df = FlickerDataFrame.from_rows(spark, rows_with_nans, names=['x', 'y'], nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    df['x'].take(None)
    extracted_rows = [
        (row['x'], row['y'])
        for row in df.take(None, convert_to_dict=True)
    ]
    for row in extracted_rows:
        for value in row:
            assert value is not None

    for expected_row, actual_row in zip(rows_with_nans, extracted_rows):
        for expected_value, actual_value in zip(expected_row, actual_row):
            if np.isnan(expected_value):
                assert np.isnan(actual_value)
            else:
                assert expected_value == actual_value


def test_nans_and_nones_in_float_columns(spark):
    rows_with_nans_and_nones = [(1.0, np.nan), (None, 4.0), (np.nan, 5.0), (4.0, None)]
    rows_with_all_nones = [(1.0, None), (None, 4.0), (None, 5.0), (4.0, None)]

    df = FlickerDataFrame.from_rows(spark, rows_with_nans_and_nones, names=['x', 'y'], nan_to_none=True)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    extracted_rows = [
        (row['x'], row['y'])
        for row in df.take(None, convert_to_dict=True)
    ]
    assert rows_with_all_nones == extracted_rows
    assert rows_with_nans_and_nones != extracted_rows

    df = FlickerDataFrame.from_rows(spark, rows_with_nans_and_nones, names=['x', 'y'], nan_to_none=False)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (4, 2)
    extracted_rows = [
        (row['x'], row['y'])
        for row in df.take(None, convert_to_dict=True)
    ]
    assert rows_with_all_nones != extracted_rows
    for expected_row, actual_row in zip(rows_with_nans_and_nones, extracted_rows):
        for expected_value, actual_value in zip(expected_row, actual_row):
            if expected_value is None:
                assert actual_value is None
            elif np.isnan(expected_value):
                assert np.isnan(actual_value)
            else:
                assert expected_value == actual_value
