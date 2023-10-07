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
from flicker import FlickerDataFrame


def test_scalar(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=3, ncols=2, names=['a', 'b'], fill='zero')
    df['c'] = 'hello'
    assert 'c' in df.names
    assert all(df[['c']].to_pandas()['c'] == 'hello')
    assert df.dtypes['c'] == 'string'

    df['d'] = 3
    assert 'd' in df.names
    assert all(df[['d']].to_pandas()['d'] == 3)
    assert df.dtypes['d'] == 'int'

    df['e'] = 4.56
    assert 'e' in df.names
    assert all(df[['e']].to_pandas()['e'] == 4.56)
    assert df.dtypes['e'] == 'double'

    df['f'] = np.nan
    assert 'f' in df.names
    assert all(pd.isna(df[['f']].to_pandas()['f']))
    assert df.dtypes['f'] == 'double'

    df['g'] = True
    assert 'g' in df.names
    assert all(df[['g']].to_pandas()['g'])
    assert df.dtypes['g'] == 'boolean'

    df['h'] = None
    assert 'h' in df.names
    assert all(pd.isna(df[['h']].to_pandas()['h']))
    assert df.dtypes['h'] == 'void'


def test_spark_column(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=2, names=['a', 'b'], fill='zero')
    df['c'] = df._df['a']
    assert 'c' in df.names
    assert (df['a'] == df['c']).all()

    df['d'] = df._df['a'] > 0
    assert 'd' in df.names
    assert df.dtypes['d'] == 'boolean'


def test_flicker_column(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=2, names=['a', 'b'], fill='zero')
    df['c'] = df['a']
    assert 'c' in df.names
    assert (df['a'] == df['c']).all()

    df['d'] = df['a'] > 0
    assert 'd' in df.names
    assert df.dtypes['d'] == 'boolean'


def test_array(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=2, ncols=2, names=['a', 'b'], fill='zero')
    df['c'] = [1, 2]
    assert 'c' in df.names
    assert df.dtypes['c'] == 'array<int>'


def test_replace_a_column(spark):
    df = FlickerDataFrame.from_shape(spark, 5, 2, names=['a', 'b'], fill='rand')
    df['a'] = True
    assert df['a'].isin([True]).all()
