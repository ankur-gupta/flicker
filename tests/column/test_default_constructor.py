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
import pytest
from pyspark.sql import Column
from pyspark.sql.functions import isnan, isnull

from flicker import FlickerDataFrame, FlickerColumn


def test_basic_usage(spark):
    df = spark.createDataFrame([(x, x, True) for x in range(5)], 'a INT, b INT, c BOOLEAN')
    assert isinstance(FlickerColumn(df, df['a']), FlickerColumn)
    assert isinstance(FlickerColumn(df, df['b']), FlickerColumn)
    assert isinstance(FlickerColumn(df, df['b'] > 0), FlickerColumn)
    assert isinstance(FlickerColumn(df, df['b'] >= 0), FlickerColumn)
    assert isinstance(FlickerColumn(df, df['b'] == 0), FlickerColumn)
    assert isinstance(FlickerColumn(df, df['b'] <= 0), FlickerColumn)
    assert isinstance(FlickerColumn(df, df['b'] < 0), FlickerColumn)
    assert isinstance(FlickerColumn(df, df['b'] % 10), FlickerColumn)
    assert isinstance(FlickerColumn(df, -df['b']), FlickerColumn)
    with pytest.raises(Exception):
        isinstance(FlickerColumn(df, ~df['b']), FlickerColumn)
    assert isinstance(FlickerColumn(df, ~df['c']), FlickerColumn)
    assert isinstance(FlickerColumn(df, df['b'].isin([5])), FlickerColumn)
    assert isinstance(FlickerColumn(df, isnan(df['b'])), FlickerColumn)
    assert isinstance(FlickerColumn(df, isnull(df['b'])), FlickerColumn)
    assert isinstance(FlickerColumn(df, df['a'] == df['b']), FlickerColumn)
    assert isinstance(FlickerColumn(df, df['a'] != df['b']), FlickerColumn)
    assert isinstance(FlickerColumn(df, df['a'] + df['b']), FlickerColumn)
    assert isinstance(FlickerColumn(df, df['a'] - df['b']), FlickerColumn)
    assert isinstance(FlickerColumn(df, df['a'] * df['b']), FlickerColumn)
    assert isinstance(FlickerColumn(df, df['a'] / df['b']), FlickerColumn)


def test_failure_wrong_type_df():
    column = Column('a')
    with pytest.raises(Exception):
        FlickerColumn('not-a-dataframe', column)
    with pytest.raises(Exception):
        FlickerColumn({}, column)
    with pytest.raises(Exception):
        FlickerColumn(None, column)
    with pytest.raises(Exception):
        FlickerColumn([], column)
    with pytest.raises(Exception):
        FlickerColumn({'a': [1, 2, 3]}, column)
    with pytest.raises(Exception):
        FlickerColumn([1, 2, 3], column)


def test_failure_wrong_type_column(spark):
    df = spark.createDataFrame([(x, x) for x in range(5)], 'a INT, b INT')
    with pytest.raises(Exception):
        FlickerColumn(df, 'not-a-column')
    with pytest.raises(Exception):
        FlickerColumn(df, None)
    with pytest.raises(Exception):
        FlickerColumn(df, {})
    with pytest.raises(Exception):
        FlickerColumn(df, {'a': [1, 2, 3]})
    with pytest.raises(Exception):
        FlickerColumn(df, [1, 2, 3])
    with pytest.raises(Exception):
        FlickerColumn(df, {1, 2, 3})
    with pytest.raises(Exception):
        FlickerColumn(df, 1)
    with pytest.raises(Exception):
        FlickerColumn(df, 1.5453)
    with pytest.raises(Exception):
        FlickerColumn(df, np.nan)


def test_duplicate_name_fails(spark):
    df = spark.createDataFrame([(x, x, x) for x in range(5)], 'a INT, a INT, b INT')
    with pytest.raises(Exception):
        FlickerColumn(df, df['b'])
    with pytest.raises(Exception):
        FlickerColumn(df, df['a'])


def test_construction_from_flicker_dataframe(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=['a', 'b', 'c', 'd', 'e', 'f'], fill='zero')
    assert isinstance(df['a'], FlickerColumn)
    assert isinstance(df['a'] > 0, FlickerColumn)
    assert isinstance(df['a'] >= 0, FlickerColumn)
    assert isinstance(df['a'] == 0, FlickerColumn)
    assert isinstance(df['a'] <= 0, FlickerColumn)
    assert isinstance(df['a'] < 0, FlickerColumn)
    assert isinstance(df['a'] % 10, FlickerColumn)
    assert isinstance(df['a'] == df['b'], FlickerColumn)
    assert isinstance(df['a'] != df['b'], FlickerColumn)
    assert isinstance(df['a'] + df['b'], FlickerColumn)
    assert isinstance(df['a'] - df['b'], FlickerColumn)
    assert isinstance(df['a'] * df['b'], FlickerColumn)
    assert isinstance(df['a'] / df['b'], FlickerColumn)
    assert isinstance(df['a'] + 1, FlickerColumn)
    assert isinstance(df['a'] - 1, FlickerColumn)
    assert isinstance(df['a'] * 0.4532, FlickerColumn)
    assert isinstance(df['a'] / 0.4532, FlickerColumn)
