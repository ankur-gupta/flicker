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
from datetime import datetime, timedelta
import numpy as np
from flicker import FlickerDataFrame
from flicker.summary import (get_column_mean, get_column_stddev, get_timestamp_column_mean, get_timestamp_column_stddev,
                             get_boolean_column_mean, get_boolean_column_stddev)


def test_none(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'], fill='zero')
    df['c'] = None
    assert get_column_mean(df._df, 'c') is None
    assert get_column_stddev(df._df, 'c') is None


def test_int(spark):
    data = {
        'int_no_null': [-10, 0, 10],
        'int_with_null_1': [-10, None, 10],
        'int_with_null_2': [-10, 0, None],
        'int_with_null_3': [None, 0, None]
    }
    df = FlickerDataFrame.from_dict(spark, data)
    df['int_with_null_4'] = None
    df['int_with_null_4'] = df['int_with_null_4'].astype(int)

    assert get_column_mean(df._df, 'int_no_null') == 0.0
    assert get_column_stddev(df._df, 'int_no_null') == 10.0
    assert get_column_mean(df._df, 'int_with_null_1') == 0.0
    assert np.allclose(get_column_stddev(df._df, 'int_with_null_1'), np.sqrt(2) * 10.0)
    assert get_column_mean(df._df, 'int_with_null_2') == -5.0
    assert np.allclose(get_column_stddev(df._df, 'int_with_null_2'), np.sqrt(2) * 5.0)
    assert get_column_mean(df._df, 'int_with_null_3') == 0.0
    assert get_column_stddev(df._df, 'int_with_null_3') is None
    assert get_column_mean(df._df, 'int_with_null_4') is None
    assert get_column_stddev(df._df, 'int_with_null_4') is None


def test_double(spark):
    data = {
        'double_no_null': [-10.0, 0.0, 10.0],
        'double_with_null_1': [-10.0, None, 10.0],
        'double_with_null_2': [-10.0, 0.0, None],
        'double_with_null_3': [None, 0.0, None],
        'double_with_nan_1': [-10.0, np.nan, 10.0],
        'double_with_nan_2': [np.nan, 0.0, np.nan]
    }
    df = FlickerDataFrame.from_dict(spark, data)
    df['double_with_null_4'] = None
    df['double_with_null_4'] = df['double_with_null_4'].astype('double')
    df['double_with_nan_3'] = np.nan
    df['double_with_nan_3'] = df['double_with_nan_3'].astype('double')

    assert get_column_mean(df._df, 'double_no_null') == 0.0
    assert get_column_stddev(df._df, 'double_no_null') == 10.0
    assert get_column_mean(df._df, 'double_with_null_1') == 0.0
    assert np.allclose(get_column_stddev(df._df, 'double_with_null_1'), np.sqrt(2) * 10.0)
    assert get_column_mean(df._df, 'double_with_null_2') == -5.0
    assert np.allclose(get_column_stddev(df._df, 'double_with_null_2'), np.sqrt(2) * 5.0)
    assert get_column_mean(df._df, 'double_with_null_3') == 0.0
    assert get_column_stddev(df._df, 'double_with_null_3') is None
    assert get_column_mean(df._df, 'double_with_null_4') is None
    assert get_column_stddev(df._df, 'double_with_null_4') is None

    assert get_column_mean(df._df, 'double_with_nan_1') == 0.0
    assert np.allclose(get_column_stddev(df._df, 'double_with_nan_1'), np.sqrt(2) * 10.0)
    assert get_column_mean(df._df, 'double_with_nan_2') == 0.0
    assert get_column_stddev(df._df, 'double_with_nan_2') is None
    assert np.isnan(get_column_mean(df._df, 'double_with_nan_3'))
    assert np.isnan(get_column_stddev(df._df, 'double_with_nan_3'))


def test_boolean(spark):
    data = {
        'bool_no_null_1': [True, True, True],
        'bool_no_null_2': [False, False, False],
        'bool_with_null_1': [True, None, True],
        'bool_with_null_2': [None, False, None]
    }
    df = FlickerDataFrame.from_dict(spark, data)
    df['bool_with_null_3'] = None
    df['bool_with_null_3'] = df['bool_with_null_3'].astype(bool)
    assert get_boolean_column_mean(df._df, 'bool_no_null_1') == 1.0
    assert get_boolean_column_stddev(df._df, 'bool_no_null_1') == 0.0
    assert get_boolean_column_mean(df._df, 'bool_no_null_2') == 0.0
    assert get_boolean_column_stddev(df._df, 'bool_no_null_2') == 0.0
    assert get_boolean_column_mean(df._df, 'bool_with_null_1') == 1.0
    assert get_boolean_column_stddev(df._df, 'bool_with_null_1') == 0.0
    assert get_boolean_column_mean(df._df, 'bool_with_null_2') == 0.0
    assert get_boolean_column_stddev(df._df, 'bool_with_null_2') is None
    assert get_boolean_column_mean(df._df, 'bool_with_null_3') is None
    assert get_boolean_column_stddev(df._df, 'bool_with_null_3') is None


def test_timestamp(spark):
    t = datetime(2023, 1, 1)
    dt = timedelta(days=1)
    data = {
        'timestamp_no_null': [t - dt, t, t + dt],
        'timestamp_with_null_1': [t, t, None],
        'timestamp_with_null_2': [None, t, None],  # stddev uses (n-1) in denominator which returns None (NULL)
    }
    df = FlickerDataFrame.from_dict(spark, data)
    df['timestamp_with_null_3'] = None
    df['timestamp_with_null_3'] = df['timestamp_with_null_3'].astype(bool)
    assert get_timestamp_column_mean(df._df, 'timestamp_no_null') == t
    assert get_timestamp_column_stddev(df._df, 'timestamp_no_null') == dt
    assert get_timestamp_column_mean(df._df, 'timestamp_with_null_1') == t
    assert get_timestamp_column_stddev(df._df, 'timestamp_with_null_1') == timedelta(0)
    assert get_timestamp_column_mean(df._df, 'timestamp_with_null_2') == t
    assert get_timestamp_column_stddev(df._df, 'timestamp_with_null_2') is None
    assert get_timestamp_column_mean(df._df, 'timestamp_with_null_3') is None
    assert get_timestamp_column_stddev(df._df, 'timestamp_with_null_3') is None
