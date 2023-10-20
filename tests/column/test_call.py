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
import pandas as pd
from flicker import FlickerDataFrame, FlickerColumn


def test_call_default(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    assert isinstance(df['a'], FlickerColumn)
    assert isinstance(df['a'](), pd.Series)
    assert all(df['a']() == 0)


def test_call_n_int(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    assert isinstance(df['b'](2), pd.Series)
    assert df['a'](2).shape == (2,)


def test_call_n_none(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    assert isinstance(df['c'](None), pd.Series)
    assert df['c'](None).shape == (df.nrows,)


def test_call_dont_use_pandas_dtypes(spark):
    data = {
        'double_no_null': (1.2, 3.98, 4.5, -0.58, 0.0),
        'double_with_null': (1.2, 3.98, None, -0.58, 0.0),
        'int_no_null': (1, 2, 3, 4, 5),
        'int_with_null': (1, None, 3, 4, None),
    }
    df = FlickerDataFrame.from_dict(spark, data)

    for name in data.keys():
        s = df[name](use_pandas_dtypes=False)
        assert isinstance(s, pd.Series)
        assert s.dtypes == object

    s = df['int_with_null'](n=2, use_pandas_dtypes=False)
    assert isinstance(s, pd.Series)
    assert s.shape == (2,)
    assert s.dtypes == object


def test_call_use_pandas_dtypes(spark):
    data = {
        'double_no_null': (1.2, 3.98, 4.5, -0.58, 0.0),
        'double_with_null': (1.2, 3.98, None, -0.58, 0.0),
        'int_no_null': (1, 2, 3, 4, 5),
        'int_with_null': (1, None, 3, 4, None),
    }
    df = FlickerDataFrame.from_dict(spark, data)

    for name in data.keys():
        s = df[name](use_pandas_dtypes=True)
        assert isinstance(s, pd.Series)
    assert df['double_no_null'](use_pandas_dtypes=True).dtypes == float
    assert df['double_with_null'](use_pandas_dtypes=True).dtypes == float
    assert df['int_no_null'](use_pandas_dtypes=True).dtypes == int
    assert df['int_with_null'](use_pandas_dtypes=True).dtypes == float

    s = df['int_no_null'](n=2, use_pandas_dtypes=True)
    assert isinstance(s, pd.Series)
    assert s.shape == (2,)
    assert s.dtypes == int


def test_call_with_zero_n(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'])
    df['a'](0)  # ensure that this doesn't fail
    assert isinstance(df['a'](0), pd.Series)
    assert df['a'](0).shape == (0,)
    assert df['a'](0).name == 'a'
    assert df['b'](0).name == 'b'


def test_call_empty_dataframe(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'])
    df['c'] = False

    subset_df = df[df['c']]
    assert subset_df.nrows == 0
    subset_df['a']()  # ensure that this doesn't fail
    assert isinstance(subset_df['a'](), pd.Series)
    assert subset_df['a']().shape == (0,)
    assert subset_df['b']().shape == (0,)
    assert subset_df['a']().name == 'a'
    assert subset_df['b']().name == 'b'
