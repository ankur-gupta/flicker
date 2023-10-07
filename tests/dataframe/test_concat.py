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


def test_basic_usage(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    df2 = df.concat(df, ignore_names=False)
    assert isinstance(df2, FlickerDataFrame)
    assert df2.shape == (df.nrows * 2, df.ncols)

    df2 = df.concat(df, ignore_names=True)
    assert isinstance(df2, FlickerDataFrame)
    assert df2.shape == (df.nrows * 2, df.ncols)


def test_spark_dataframe(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    df2 = df.concat(df._df, ignore_names=False)
    assert isinstance(df2, FlickerDataFrame)
    assert df2.shape == (df.nrows * 2, df.ncols)

    df2 = df.concat(df._df, ignore_names=True)
    assert isinstance(df2, FlickerDataFrame)
    assert df2.shape == (df.nrows * 2, df.ncols)


def test_unequal_number_of_columns(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=2, ncols=6, names=list('abcdef'), fill='zero')
    with pytest.raises(Exception):
        df.concat(df[:3], ignore_names=True)
    with pytest.raises(Exception):
        df.concat(df[:3], ignore_names=False)


def test_same_ncols_different_order_of_names(spark):
    rows = [('Alice', 25), ('Bob', 30), ('Charlie', 35)]
    names = ['name', 'age']
    df = FlickerDataFrame.from_rows(spark, rows, names)
    df2 = df.concat(df[::-1], ignore_names=True)
    assert isinstance(df2, FlickerDataFrame)
    assert df2.shape == (df.nrows * 2, df.ncols)

    df2 = df.concat(df[::-1], ignore_names=False)
    assert isinstance(df2, FlickerDataFrame)
    assert df2.shape == (df.nrows * 2, df.ncols)


def test_same_ncols_different_names(spark):
    rows = [('Alice', 25), ('Bob', 30), ('Charlie', 35)]
    names = ['name', 'age']
    df = FlickerDataFrame.from_rows(spark, rows, names)
    df2 = df.concat(df.rename({'name': 'a', 'age': 'b'}), ignore_names=True)
    assert isinstance(df2, FlickerDataFrame)
    assert df2.shape == (df.nrows * 2, df.ncols)

    with pytest.raises(Exception):
        df.concat(df.rename({'name': 'a', 'age': 'b'}), ignore_names=False)
    with pytest.raises(Exception):
        df.concat(df.rename({'age': 'b'}), ignore_names=False)
