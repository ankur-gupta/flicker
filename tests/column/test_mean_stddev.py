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
from flicker import FlickerDataFrame
from datetime import datetime, timedelta


def test_int_no_nulls(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, list('ab'), fill='zero')
    assert df['a'].mean() == 0.0
    assert df['a'].mean(ignore_nan=True) == 0
    assert df['a'].mean(ignore_nan=False) == 0

    assert df['a'].stddev() == 0.0
    assert df['a'].stddev(ignore_nan=True) == 0
    assert df['a'].stddev(ignore_nan=False) == 0


def test_int_with_nulls(spark):
    data = {
        'a': [-1, -1, 0, None, 0, 1, 1],
        'b': [-1, None, None, None, None, None, 1],
    }
    df = FlickerDataFrame.from_dict(spark, data)
    df['c'] = None
    df['c'] = df['c'].astype(int)

    assert df['a'].mean() == 0.0
    assert df['a'].mean(ignore_nan=True) == 0.0
    assert df['a'].mean(ignore_nan=False) == 0.0
    assert np.allclose(df['a'].stddev(), np.sqrt(4 / 5))
    assert np.allclose(df['a'].stddev(ignore_nan=True), np.sqrt(4 / 5))
    assert np.allclose(df['a'].stddev(ignore_nan=False), np.sqrt(4 / 5))

    assert df['b'].mean() == 0.0
    assert df['b'].mean(ignore_nan=True) == 0.0
    assert df['b'].mean(ignore_nan=False) == 0.0
    assert np.allclose(df['b'].stddev(), np.sqrt(2))
    assert np.allclose(df['b'].stddev(ignore_nan=True), np.sqrt(2))
    assert np.allclose(df['b'].stddev(ignore_nan=False), np.sqrt(2))

    assert df['c'].mean() is None
    assert df['c'].mean(ignore_nan=True) is None
    assert df['c'].mean(ignore_nan=False) is None
    assert df['c'].stddev() is None
    assert df['c'].stddev(ignore_nan=True) is None
    assert df['c'].stddev(ignore_nan=False) is None


def test_double_with_nans(spark):
    rows = [
        (1.0, 1.0, 1.0),
        (1.0, np.nan, np.nan),
        (1.0, 1.0, np.nan)
    ]
    names = ['double_no_nan', 'double_with_nan_1', 'double_with_nan_2']
    df = FlickerDataFrame(spark.createDataFrame(rows, names))
    df['double_with_all_nan'] = np.nan

    assert set(df.dtypes.values()) == {'double'}

    assert df['double_no_nan'].mean() == 1.0
    assert df['double_no_nan'].mean(ignore_nan=True) == 1.0
    assert df['double_no_nan'].mean(ignore_nan=False) == 1.0
    assert df['double_no_nan'].stddev() == 0.0
    assert df['double_no_nan'].stddev(ignore_nan=True) == 0.0
    assert df['double_no_nan'].stddev(ignore_nan=False) == 0.0

    assert df['double_with_nan_1'].mean(ignore_nan=True) == 1.0
    assert np.isnan(df['double_with_nan_1'].mean(ignore_nan=False))
    assert df['double_with_nan_1'].stddev(ignore_nan=True) == 0.0
    assert np.isnan(df['double_with_nan_1'].stddev(ignore_nan=False))

    assert df['double_with_nan_2'].mean(ignore_nan=True) == 1.0
    assert np.isnan(df['double_with_nan_2'].mean(ignore_nan=False))
    assert df['double_with_nan_2'].stddev(ignore_nan=True) is None
    assert np.isnan(df['double_with_nan_2'].stddev(ignore_nan=False))

    assert df['double_with_all_nan'].mean(ignore_nan=True) is None
    assert np.isnan(df['double_with_all_nan'].mean(ignore_nan=False))
    assert df['double_with_all_nan'].stddev(ignore_nan=True) is None
    assert np.isnan(df['double_with_all_nan'].stddev(ignore_nan=False))


def test_timestamp_no_nulls(spark):
    t = datetime(2023, 1, 1)
    dt = timedelta(days=1)
    df = FlickerDataFrame.from_rows(spark, [(t - dt, 1), (t, 2), (t + dt, 3)], names=['t', 'n'])
    assert df['t'].mean() == t
    assert df['t'].stddev() == dt


def test_timestamp_with_nulls(spark):
    t = datetime(2023, 1, 1)
    dt = timedelta(days=1)
    df = FlickerDataFrame.from_rows(spark, [(t - dt, 1), (t, 2), (t + dt, 3), (None, None)], names=['t', 'n'])
    assert df['t'].mean() == t
    assert df['t'].stddev() == dt


def test_boolean_no_nulls(spark):
    df = FlickerDataFrame.from_rows(spark, [(True, 1), (True, 2), (True, 3)], names=['t', 'n'])
    assert df['t'].mean() == 1.0
    assert df['t'].stddev() == 0.0


def test_boolean_with_nulls(spark):
    df = FlickerDataFrame.from_rows(spark, [(True, 1), (True, 2), (True, 3), (None, None)], names=['t', 'n'])
    assert df['t'].mean() == 1.0
    assert df['t'].stddev() == 0.0


def test_chains(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=3, ncols=1, names=['zero'], fill='zero')
    df['one'] = 1

    # This chain fails if the code is not written to handle generalities.
    assert (df['one'] - df['zero']).mean() == 1.0
    assert (df['one'] - df['zero']).stddev() == 0.0
