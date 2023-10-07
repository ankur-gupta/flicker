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


def test_wrong_type_on(spark):
    df = FlickerDataFrame.from_rows(spark, [('A',), ('B',), ('C',), ], ['name'])
    assert isinstance(df, FlickerDataFrame)
    with pytest.raises(TypeError):
        df.merge(df, on=None)
    with pytest.raises(TypeError):
        df.merge(df, on={})
    with pytest.raises(TypeError):
        df.merge(df, on={'name': 1})
    with pytest.raises(TypeError):
        df.merge(df, on={'name': 'name'})
    with pytest.raises(TypeError):
        df.merge(df, on='name')
    with pytest.raises(TypeError):
        df.merge(df, on=1)
    with pytest.raises(TypeError):
        df.merge(df, on=[1])
    with pytest.raises(TypeError):
        df.merge(df, on=(1, 2))
    with pytest.raises(TypeError):
        df.merge(df, on={1, 2})
    with pytest.raises(TypeError):
        df.merge(df, on=(1, 'name'))
    with pytest.raises(TypeError):
        df.merge(df, on=('name', 1))


def test_wrong_value_on(spark):
    df = FlickerDataFrame.from_rows(spark, [('a',), ('b',), ('c',), ], ['name'])
    assert isinstance(df, FlickerDataFrame)
    with pytest.raises(ValueError):
        df.merge(df, on=[])


def test_single_column_merge(spark):
    df1 = FlickerDataFrame.from_rows(spark, [('a',), ('b',), ('c',), ], ['name'])
    df2 = FlickerDataFrame.from_rows(spark, [('a',), ('d',), ('e',), ], ['name'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    inner_df = df1.merge(df2, on=['name'], how='inner')
    assert set(inner_df.names) == {'name'}
    assert inner_df.nrows == 1
    assert inner_df.ncols == 1
    assert inner_df.shape == (1, 1)
    assert inner_df.take(1)[0] == {'name': 'a'}

    left_df = df1.merge(df2, on=['name'], how='left')
    assert set(left_df.names) == {'name'}
    assert left_df.nrows == df1.nrows
    assert left_df.ncols == 1
    left_df_as_rows = left_df.take(None, convert_to_dict=True)
    left_df_as_rows.sort(key=lambda row: row['name'])
    assert left_df_as_rows == [{'name': 'a'}, {'name': 'b'}, {'name': 'c'}]

    right_df = df1.merge(df2, on=['name'], how='right')
    assert set(right_df.names) == {'name'}
    assert right_df.nrows == df2.nrows
    assert right_df.ncols == 1
    right_df_as_rows = right_df.take(None, convert_to_dict=True)
    right_df_as_rows.sort(key=lambda row: row['name'])
    assert right_df_as_rows == [{'name': 'a'}, {'name': 'd'}, {'name': 'e'}]

    outer_df = df1.merge(df2, on=['name'], how='outer')
    assert set(outer_df.names) == {'name'}
    assert outer_df.nrows == 5
    assert outer_df.ncols == 1
    assert set(outer_df.to_dict(None)['name']) == {'a', 'b', 'c', 'd', 'e'}


def test_merge_with_pyspark_dataframe(spark):
    df1 = FlickerDataFrame.from_rows(spark, [('a',), ('b',), ('c',), ], ['name'])
    df2 = FlickerDataFrame.from_rows(spark, [('a',), ('d',), ('e',), ], ['name'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    inner_df = df1.merge(df2._df, on=['name'], how='inner')
    assert set(inner_df.names) == {'name'}
    assert inner_df.nrows == 1
    assert inner_df.ncols == 1
    assert inner_df.shape == (1, 1)
    assert inner_df.take(1)[0] == {'name': 'a'}


def test_single_column_merge_with_extra_common_columns(spark):
    df1 = FlickerDataFrame.from_rows(spark, [('a', 1), ('b', 2), ('c', 3), ], ['name', 'number'])
    df2 = FlickerDataFrame.from_rows(spark, [('a', 4), ('d', 5), ('e', 6), ], ['name', 'number'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    inner_df = df1.merge(df2._df, on=['name'], how='inner', lprefix='', lsuffix='_l', rprefix='', rsuffix='_r')
    assert set(inner_df.names) == {'name', 'number_l', 'number_r'}
    assert inner_df.nrows == 1
    assert inner_df.ncols == 3
    assert inner_df.take(1)[0] == {'name': 'a', 'number_l': 1, 'number_r': 4}

    left_df = df1.merge(df2._df, on=['name'], how='left', lprefix='l_', lsuffix='', rprefix='r_', rsuffix='')
    assert set(left_df.names) == {'name', 'l_number', 'r_number'}
    assert left_df.nrows == 3
    assert left_df.ncols == 3
    left_df_as_rows = left_df.take(None, convert_to_dict=True)
    left_df_as_rows.sort(key=lambda row: row['name'])
    assert left_df_as_rows == [
        {'name': 'a', 'l_number': 1, 'r_number': 4},
        {'name': 'b', 'l_number': 2, 'r_number': None},
        {'name': 'c', 'l_number': 3, 'r_number': None}
    ]


def test_single_column_merge_with_extra_uncommon_columns(spark):
    df1 = FlickerDataFrame.from_rows(spark, [('a', 1), ('b', 2), ('c', 3), ], ['name', 'number1'])
    df2 = FlickerDataFrame.from_rows(spark, [('a', 4), ('d', 5), ('e', 6), ], ['name', 'number2'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    inner_df = df1.merge(df2._df, on=['name'], how='inner', lprefix='', lsuffix='_l', rprefix='', rsuffix='_r')
    assert set(inner_df.names) == {'name', 'number1', 'number2'}
    assert inner_df.nrows == 1
    assert inner_df.ncols == 3
    assert inner_df.take(1)[0] == {'name': 'a', 'number1': 1, 'number2': 4}


def test_merge_result_has_duplicate_names(spark):
    df1 = FlickerDataFrame.from_rows(spark, [('a', 1), ('b', 2), ('c', 3), ], ['name', 'number'])
    df2 = FlickerDataFrame.from_rows(spark, [('a', 4), ('d', 5), ('e', 6), ], ['name', 'number'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    with pytest.raises(Exception):
        df1.merge(df2._df, on=['name'], how='inner', lprefix='', lsuffix='', rprefix='', rsuffix='')


def test_left_renamed_dataframe_has_duplicate_names(spark):
    df1 = FlickerDataFrame.from_rows(
        spark, [('a', 1, 1), ('b', 2, 2), ('c', 3, 3), ], ['name', 'number', 'number_l']
    )
    df2 = FlickerDataFrame.from_rows(spark, [('a', 4), ('d', 5), ('e', 6), ], ['name', 'number'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    with pytest.raises(Exception):
        df1.merge(df2._df, on=['name'], how='inner', lprefix='', lsuffix='_l', rprefix='', rsuffix='')


def test_right_renamed_dataframe_has_duplicate_names(spark):
    df1 = FlickerDataFrame.from_rows(spark, [('a', 1), ('b', 2), ('c', 3), ], ['name', 'number'])
    df2 = FlickerDataFrame.from_rows(
        spark, [('a', 4, 4), ('d', 5, 5), ('e', 6, 6), ], ['name', 'number', 'number_r']
    )
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    with pytest.raises(Exception):
        df1.merge(df2._df, on=['name'], how='inner', lprefix='', lsuffix='_l', rprefix='', rsuffix='_r')


def test_multi_column_merge(spark):
    rows1 = [('Alice', 'Sales', 25), ('Bob', 'Sales', 30), ('Charlie', 'Sales', 35), ('Delta', 'Marketing', 25)]
    rows2 = [('Alice', 'Sales', 25), ('Rob', 'Sales', 30), ('Romeo', 'Engineering', 35), ('Delta', 'Marketing', 25)]
    df1 = FlickerDataFrame.from_rows(spark, rows1, ['name', 'department', 'age'])
    df2 = FlickerDataFrame.from_rows(spark, rows2, ['name', 'department', 'age'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    inner_df = df1.merge(df2, on=['name', 'department', 'age'], how='inner')
    assert isinstance(inner_df, FlickerDataFrame)
    assert set(inner_df.names) == {'name', 'department', 'age'}
    assert inner_df.nrows == 2
    assert inner_df.ncols == 3

    inner_dict = inner_df.to_dict(None)
    assert set(inner_dict['department']) == {'Marketing', 'Sales'}
    assert set(inner_dict['name']) == {'Delta', 'Alice'}
    assert set(inner_dict['age']) == {25}

    inner_rows = inner_df.take(2, convert_to_dict=True)
    inner_rows.sort(key=lambda row: row['name'], reverse=False)
    assert inner_rows == [
        {'name': 'Alice', 'department': 'Sales', 'age': 25},
        {'name': 'Delta', 'department': 'Marketing', 'age': 25}
    ]
