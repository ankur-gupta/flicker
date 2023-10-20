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
import pandas as pd

from flicker import FlickerDataFrame
from flicker.summary import get_column_min, get_column_max


def test_none(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'], fill='zero')
    df['c'] = None
    assert get_column_min(df._df, 'c') is None
    assert get_column_max(df._df, 'c') is None


def test_int_double(spark):
    rows = [[-100, None, 2.0, 2.0, 2.0],
            [4, 4, 5.0, 5.0, 5.0],
            [-1, -1, -90.0, None, np.nan],
            [0, 0, 0.0, 0.0, 0.0],
            [78, 78, 1.0, 1.0, 1.0],
            [-98, -98, -10.0, -10.0, -10.0],
            [23, None, 20.0, 20.0, 20.0],
            [100, 90, 100.0, None, np.nan]]
    names = ['int_no_null', 'int_with_null', 'double_no_null', 'double_with_null', 'double_with_nan']
    df = FlickerDataFrame(spark.createDataFrame(rows, names))
    df['double_with_all_nan'] = np.nan
    df['double_with_all_nan'] = df['double_with_all_nan'].astype('double')
    df['int_with_all_null'] = None
    df['int_with_all_null'] = df['int_with_all_null'].astype('double')

    assert get_column_min(df._df, 'int_no_null') == -100
    assert get_column_max(df._df, 'int_no_null') == 100
    assert get_column_min(df._df, 'int_with_null') == -98
    assert get_column_max(df._df, 'int_with_null') == 90
    assert get_column_min(df._df, 'double_no_null') == -90.0
    assert get_column_max(df._df, 'double_no_null') == 100.0
    assert get_column_min(df._df, 'double_with_null') == -10.0
    assert get_column_max(df._df, 'double_with_null') == 20.0
    assert get_column_min(df._df, 'double_with_nan') == -10.0
    assert np.isnan(get_column_max(df._df, 'double_with_nan'))
    assert np.isnan(get_column_min(df._df, 'double_with_all_nan'))
    assert np.isnan(get_column_max(df._df, 'double_with_all_nan'))
    assert get_column_max(df._df, 'int_with_all_null') is None
    assert get_column_min(df._df, 'int_with_all_null') is None


def test_boolean(spark):
    data = {
        'bool_no_null_1': [True, False, True],
        'bool_no_null_2': [True, True, True],
        'bool_no_null_3': [False, False, False],
        'bool_with_null_1': [True, None, True],
        'bool_with_null_2': [None, False, None]
    }
    df = FlickerDataFrame.from_dict(spark, data)
    df['bool_with_null_3'] = None
    df['bool_with_null_3'] = df['bool_with_null_3'].astype(bool)
    assert get_column_min(df._df, 'bool_no_null_1') is False
    assert get_column_max(df._df, 'bool_no_null_1') is True
    assert get_column_min(df._df, 'bool_no_null_2') is True
    assert get_column_max(df._df, 'bool_no_null_2') is True
    assert get_column_min(df._df, 'bool_no_null_3') is False
    assert get_column_max(df._df, 'bool_no_null_3') is False
    assert get_column_min(df._df, 'bool_with_null_1') is True
    assert get_column_max(df._df, 'bool_with_null_1') is True
    assert get_column_min(df._df, 'bool_with_null_2') is False
    assert get_column_max(df._df, 'bool_with_null_2') is False
    assert get_column_min(df._df, 'bool_with_null_3') is None
    assert get_column_max(df._df, 'bool_with_null_3') is None


def test_timestamp(spark):
    t = datetime(2023, 1, 1)
    dt = timedelta(days=1)
    data = {
        'timestamp_no_null': [t - dt, t, t + dt],
        'timestamp_with_null_1': [t - dt, None, t + dt],
        'timestamp_with_null_2': [t - dt, t, None],
        'timestamp_with_null_3': [None, t, t + dt],
        'timestamp_with_null_4': [None, t, None]
    }
    df = FlickerDataFrame.from_dict(spark, data)
    df['timestamp_with_null_5'] = None
    df['timestamp_with_null_5'] = df['timestamp_with_null_5'].astype(bool)

    assert get_column_min(df._df, 'timestamp_no_null') == t - dt
    assert get_column_max(df._df, 'timestamp_no_null') == t + dt
    assert get_column_min(df._df, 'timestamp_with_null_1') == t - dt
    assert get_column_max(df._df, 'timestamp_with_null_1') == t + dt
    assert get_column_min(df._df, 'timestamp_with_null_2') == t - dt
    assert get_column_max(df._df, 'timestamp_with_null_2') == t
    assert get_column_min(df._df, 'timestamp_with_null_3') == t
    assert get_column_max(df._df, 'timestamp_with_null_3') == t + dt
    assert get_column_min(df._df, 'timestamp_with_null_4') == t
    assert get_column_max(df._df, 'timestamp_with_null_4') == t
    assert get_column_min(df._df, 'timestamp_with_null_5') is None
    assert get_column_max(df._df, 'timestamp_with_null_5') is None

    # Spark converts NaT into NULLs
    # rows = [(t, None), (None, t)]
    # names = ['timestamp_with_nat_1', 'timestamp_with_nat_2']
    # x = pd.DataFrame.from_records(rows, columns=names)  # contains NaT
    # spark.createDataFrame(x).show()  # NaT converted to NULLs
