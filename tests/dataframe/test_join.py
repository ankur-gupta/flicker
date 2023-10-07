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
    df = FlickerDataFrame.from_rows(spark, [('a',), ('b',), ('c',), ], ['name'])
    assert isinstance(df, FlickerDataFrame)
    with pytest.raises(TypeError):
        df.join(df, on=None)
    with pytest.raises(TypeError):
        df.join(df, on={'name': 1})
    with pytest.raises(TypeError):
        df.join(df, on={1: 'name'})
    with pytest.raises(TypeError):
        df.join(df, on='name')
    with pytest.raises(TypeError):
        df.join(df, on=['name'])
    with pytest.raises(TypeError):
        df.join(df, on=1)
    with pytest.raises(TypeError):
        df.join(df, on=[1])
    with pytest.raises(TypeError):
        df.join(df, on=(1, 2))
    with pytest.raises(TypeError):
        df.join(df, on={1, 2})
    with pytest.raises(TypeError):
        df.join(df, on=(1, 'name'))
    with pytest.raises(TypeError):
        df.join(df, on=('name', 1))


def test_wrong_value_on(spark):
    df = FlickerDataFrame.from_rows(spark, [('a',), ('b',), ('c',), ], ['name'])
    assert isinstance(df, FlickerDataFrame)
    with pytest.raises(ValueError):
        df.join(df, on={})


def test_basic_join(spark):
    df1 = FlickerDataFrame.from_rows(spark, [('a',), ('b',), ('c',), ], ['name1'])
    df2 = FlickerDataFrame.from_rows(spark, [('a',), ('d',), ('e',), ], ['name2'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    inner_df = df1.join(df2, on={'name1': 'name2'}, how='inner')
    assert set(inner_df.names) == {'name1', 'name2'}
    assert inner_df.nrows == 1
    assert inner_df.ncols == 2
    assert inner_df.take(1)[0] == {'name1': 'a', 'name2': 'a'}

    left_df = df1.join(df2, on={'name1': 'name2'}, how='left')
    assert set(inner_df.names) == {'name1', 'name2'}
    assert left_df.nrows == df1.nrows
    assert left_df.ncols == 2
    left_df_as_rows = left_df.take(None, convert_to_dict=True)
    left_df_as_rows.sort(key=lambda row: row['name1'])
    assert left_df_as_rows == [
        {'name1': 'a', 'name2': 'a'},
        {'name1': 'b', 'name2': None},
        {'name1': 'c', 'name2': None}
    ]

    right_df = df1.join(df2, on={'name1': 'name2'}, how='right')
    assert set(inner_df.names) == {'name1', 'name2'}
    assert right_df.nrows == df2.nrows
    assert right_df.ncols == 2
    right_df_as_rows = right_df.take(None, convert_to_dict=True)
    right_df_as_rows.sort(key=lambda row: row['name2'])
    assert right_df_as_rows == [
        {'name1': 'a', 'name2': 'a'},
        {'name1': None, 'name2': 'd'},
        {'name1': None, 'name2': 'e'}
    ]

    outer_df = df1.join(df2, on={'name1': 'name2'}, how='outer')
    assert set(outer_df.names) == {'name1', 'name2'}
    assert outer_df.nrows == 5
    assert outer_df.ncols == 2
    assert set(outer_df.to_dict(None)['name1']) == {'a', 'b', 'c', None}
    assert set(outer_df.to_dict(None)['name2']) == {'a', 'd', 'e', None}


def test_join_with_pyspark_dataframe(spark):
    df1 = FlickerDataFrame.from_rows(spark, [('a',), ('b',), ('c',), ], ['name1'])
    df2 = FlickerDataFrame.from_rows(spark, [('a',), ('d',), ('e',), ], ['name2'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    inner_df = df1.join(df2._df, on={'name1': 'name2'}, how='inner')
    assert set(inner_df.names) == {'name1', 'name2'}
    assert inner_df.nrows == 1
    assert inner_df.ncols == 2
    assert inner_df.take(1)[0] == {'name1': 'a', 'name2': 'a'}


def test_single_condition_join_with_extra_common_columns(spark):
    df1 = FlickerDataFrame.from_rows(spark, [('a', 1), ('b', 2), ('c', 3), ], ['name1', 'number'])
    df2 = FlickerDataFrame.from_rows(spark, [('a', 4), ('d', 5), ('e', 6), ], ['name2', 'number'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    inner_df = df1.join(df2, on={'name1': 'name2'}, how='inner', lprefix='', lsuffix='_l', rprefix='', rsuffix='_r')
    assert set(inner_df.names) == {'name1', 'name2', 'number_l', 'number_r'}
    assert inner_df.nrows == 1
    assert inner_df.ncols == 4
    assert inner_df.take(1)[0] == {'name1': 'a', 'name2': 'a', 'number_l': 1, 'number_r': 4}

    left_df = df1.join(df2, on={'name1': 'name2'}, how='left', lprefix='l_', lsuffix='', rprefix='', rsuffix='_r')
    assert set(left_df.names) == {'name1', 'name2', 'l_number', 'number_r'}
    assert left_df.nrows == df1.nrows
    assert left_df.ncols == 4
    left_rows = left_df.take(None, convert_to_dict=True)
    left_rows.sort(key=lambda row: row['l_number'])
    assert left_rows == [
        {'name1': 'a', 'name2': 'a', 'l_number': 1, 'number_r': 4},
        {'name1': 'b', 'name2': None, 'l_number': 2, 'number_r': None},
        {'name1': 'c', 'name2': None, 'l_number': 3, 'number_r': None},
    ]


def test_single_condition_join_with_extra_uncommon_columns(spark):
    df1 = FlickerDataFrame.from_rows(spark, [('a', 1), ('b', 2), ('c', 3), ], ['name1', 'number1'])
    df2 = FlickerDataFrame.from_rows(spark, [('a', 4), ('d', 5), ('e', 6), ], ['name2', 'number2'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    inner_df = df1.join(df2._df, on={'name1': 'name2'}, how='inner', lprefix='', lsuffix='_l', rprefix='', rsuffix='_r')
    assert set(inner_df.names) == {'name1', 'name2', 'number1', 'number2'}
    assert inner_df.nrows == 1
    assert inner_df.ncols == 4
    assert inner_df.take(1)[0] == {'name1': 'a', 'name2': 'a', 'number1': 1, 'number2': 4}


def test_single_condition_join_with_same_column_name(spark):
    df1 = FlickerDataFrame.from_rows(spark, [('a',), ('b',), ('c',), ], ['name'])
    df2 = FlickerDataFrame.from_rows(spark, [('a',), ('d',), ('e',), ], ['name'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    inner_df = df1.join(df2, on={'name': 'name'}, how='inner', lprefix='', lsuffix='_l', rprefix='', rsuffix='_r')
    assert set(inner_df.names) == {'name_l', 'name_r'}
    assert inner_df.nrows == 1
    assert inner_df.ncols == 2
    assert inner_df.take(1)[0] == {'name_l': 'a', 'name_r': 'a'}


def test_single_condition_join_with_same_column_names(spark):
    df1 = FlickerDataFrame.from_rows(spark, [('a', 1), ('b', 2), ('c', 3), ], ['name', 'number'])
    df2 = FlickerDataFrame.from_rows(spark, [('a', 4), ('d', 5), ('e', 6), ], ['name', 'number'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    inner_df = df1.join(df2, on={'name': 'name'}, how='inner', lprefix='', lsuffix='_l', rprefix='', rsuffix='_r')
    assert set(inner_df.names) == {'name_l', 'name_r', 'number_l', 'number_r'}
    assert inner_df.nrows == 1
    assert inner_df.ncols == 4
    assert inner_df.take(1)[0] == {'name_l': 'a', 'name_r': 'a', 'number_l': 1, 'number_r': 4}


def test_single_condition_join_with_same_column_name_and_extra_unnamed_column(spark):
    df1 = FlickerDataFrame.from_rows(spark, [('a', 1), ('b', 2), ('c', 3), ], ['name', 'number1'])
    df2 = FlickerDataFrame.from_rows(spark, [('a', 4), ('d', 5), ('e', 6), ], ['name', 'number2'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    inner_df = df1.join(df2, on={'name': 'name'}, how='inner', lprefix='', lsuffix='_l', rprefix='', rsuffix='_r')
    assert set(inner_df.names) == {'name_l', 'name_r', 'number1', 'number2'}
    assert inner_df.nrows == 1
    assert inner_df.ncols == 4
    assert inner_df.take(1)[0] == {'name_l': 'a', 'name_r': 'a', 'number1': 1, 'number2': 4}


def test_multi_column_join(spark):
    rows1 = [('Alice', 'Sales', 25), ('Bob', 'Sales', 30), ('Charlie', 'Sales', 35), ('Delta', 'Marketing', 25)]
    rows2 = [('Alice', 'Sales', 25), ('Rob', 'Sales', 30), ('Romeo', 'Engineering', 35), ('Delta', 'Marketing', 25)]
    df1 = FlickerDataFrame.from_rows(spark, rows1, ['name', 'department', 'age'])
    df2 = FlickerDataFrame.from_rows(spark, rows2, ['name', 'department', 'age'])
    assert isinstance(df1, FlickerDataFrame)
    assert isinstance(df2, FlickerDataFrame)

    inner_df = df1.join(df2, on={'name': 'name', 'department': 'department', 'age': 'age'}, how='inner',
                        lprefix='', lsuffix='_l', rprefix='', rsuffix='_r')
    assert isinstance(inner_df, FlickerDataFrame)
    assert set(inner_df.names) == {'name_l', 'department_l', 'age_l', 'name_r', 'department_r', 'age_r'}
    assert inner_df.nrows == 2
    assert inner_df.ncols == 6

    inner_dict = inner_df.to_dict(None)
    assert set(inner_dict['department_l']) == {'Marketing', 'Sales'}
    assert set(inner_dict['department_r']) == {'Marketing', 'Sales'}
    assert set(inner_dict['name_l']) == {'Delta', 'Alice'}
    assert set(inner_dict['name_r']) == {'Delta', 'Alice'}
    assert set(inner_dict['age_l']) == {25}
    assert set(inner_dict['age_r']) == {25}

    inner_rows = inner_df.take(2, convert_to_dict=True)
    inner_rows.sort(key=lambda row: row['name_l'], reverse=False)
    assert inner_rows == [
        {'name_l': 'Alice', 'department_l': 'Sales', 'age_l': 25,
         'name_r': 'Alice', 'department_r': 'Sales', 'age_r': 25},
        {'name_l': 'Delta', 'department_l': 'Marketing', 'age_l': 25,
         'name_r': 'Delta', 'department_r': 'Marketing', 'age_r': 25}
    ]
