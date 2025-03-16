# Copyright 2025 Flicker Contributors
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
from flicker import concat, FlickerDataFrame


def test_no_extra_options(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'])
    df3 = concat([df] * 3)
    assert df3.nrows == df.nrows * 3
    assert df3.ncols == df.ncols
    assert df3.names == df.names


def test_spark_dataframes(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'])
    df2 = concat([df._df] * 2)
    assert df2.nrows == df.nrows * 2
    assert df2.ncols == df.ncols
    assert df2.names == df.names


def test_mix_dataframes(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'])
    df4 = concat([df, df._df, df, df._df])
    assert df4.nrows == df.nrows * 4
    assert df4.ncols == df.ncols
    assert df4.names == df.names


def test_one_item_in_input(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'])
    df1 = concat([df])
    assert df1.nrows == df.nrows
    assert df1.ncols == df.ncols
    assert df1.names == df.names
    assert (df1['a'] == df['a']).all()
    assert (df1['b'] == df['b']).all()


def test_empty_input(spark):
    with pytest.raises(Exception):
        concat([])


def test_ignore_names_fails(spark):
    df1 = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'], fill='zero')
    df2 = FlickerDataFrame.from_shape(spark, 3, 2, ['c', 'd'], fill='zero')
    df3 = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'd'], fill='zero')
    with pytest.raises(Exception):
        concat([df1, df2], ignore_names=False)
    with pytest.raises(Exception):
        concat([df2, df3], ignore_names=False)
    with pytest.raises(Exception):
        concat([df1, df2], ignore_names=False)


def test_ignore_names_works_1(spark):
    df1 = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'], fill='zero')
    df2 = FlickerDataFrame.from_shape(spark, 3, 2, ['c', 'd'], fill='zero')
    df = concat([df1, df2], ignore_names=True)
    assert df.nrows == df1.nrows + df2.nrows
    assert df.ncols == df1.ncols
    assert df.ncols == df2.ncols
    assert df.names == df1.names
    assert (df['a'] == 0).all()
    assert (df['b'] == 0).all()


def test_ignore_names_works_2(spark):
    df1 = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'], fill='zero')
    df2 = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'], fill='zero')
    df = concat([df1, df2], ignore_names=True)
    assert df.nrows == df1.nrows + df2.nrows
    assert df.ncols == df1.ncols
    assert df.ncols == df2.ncols
    assert df.names == df1.names
    assert (df['a'] == 0).all()
    assert (df['b'] == 0).all()


def test_same_columns_but_different_column_order(spark):
    df1 = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'], fill='zero')
    df1['a'] = 0
    df1['b'] = 1
    df2 = df1[['b', 'a']]
    df = concat([df1, df2], ignore_names=False)
    assert df.nrows == df1.nrows + df2.nrows
    assert df.ncols == df1.ncols
    assert df.ncols == df2.ncols
    assert df.names == df1.names
    assert (df['a'] == 0).all()
    assert (df['b'] == 1).all()


def test_different_number_of_columns(spark):
    df1 = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'])
    df2 = FlickerDataFrame.from_shape(spark, 3, 4, ['c', 'd', 'e', 'f'])
    with pytest.raises(Exception):
        concat([df1, df2], ignore_names=False)
    with pytest.raises(Exception):
        concat([df1, df2], ignore_names=True)
