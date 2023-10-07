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
    columns = [('Alice', 'Bob', 'Charlie'), (25, 25, 25)]
    df = FlickerDataFrame.from_columns(spark, columns)
    assert isinstance(df, FlickerDataFrame)
    assert 'name' not in df
    assert 'age' not in df
    assert df.shape == (3, 2)
    for name in df.names:
        assert isinstance(name, str)


def test_with_names(spark):
    columns = [('Alice', 'Bob', 'Charlie'), (25, 25, 25)]
    names = ['name', 'age']
    df = FlickerDataFrame.from_columns(spark, columns, names)
    assert isinstance(df, FlickerDataFrame)
    assert set(df.names) == {'name', 'age'}
    assert df.shape == (3, 2)
    assert (df['age'] == 25).all()


def test_duplicated_names_failure(spark):
    columns = [[1, 2, 3], ['hello', 'spark', 'flicker']]
    names = ['a', 'a']
    with pytest.raises(Exception):
        FlickerDataFrame.from_columns(spark, columns, names)


def test_columns_names_mismatch(spark):
    columns = [[1, 2, 3], ['hello', 'spark', 'flicker']]
    names = ['a', 'b', 'c']
    with pytest.raises(Exception):
        FlickerDataFrame.from_columns(spark, columns, names)


def test_unequal_number_of_rows(spark):
    columns = [[1, 2, 3, 5], ['hello', 'spark', 'flicker']]
    names = ['a', 'b', 'c']
    with pytest.raises(Exception):
        FlickerDataFrame.from_columns(spark, columns, names)


def test_typical_usage(spark):
    columns = [[1, 2, 3], ['hello', 'spark', 'flicker']]

    df = FlickerDataFrame.from_columns(spark, columns)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (3, 2)

    first_name = df.names[0]
    first_column = df[[first_name]].to_pandas()[first_name].to_numpy()
    assert np.all(first_column == np.array(columns[0]))

    second_name = df.names[1]
    second_column = df[[second_name]].to_pandas()[second_name].to_numpy()
    assert np.all(second_column == np.array(columns[1]))


def test_usage_with_names(spark):
    columns = [[1, 2, 3], ['hello', 'spark', 'flicker']]
    names = ['a', 'b']

    df = FlickerDataFrame.from_columns(spark, columns, names)
    assert isinstance(df, FlickerDataFrame)
    assert df.shape == (3, 2)
    assert list(df.names) == list(names)

    for i, name in enumerate(df.names):
        column = df[[name]].to_pandas()[name].to_numpy()
        expected_column = np.array(columns[i])
        assert np.all(column == expected_column)
