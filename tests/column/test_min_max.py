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
from datetime import datetime, timedelta
from flicker import FlickerDataFrame


def test_basic_usage(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, list('ab'), fill='zero')
    assert df['a'].min() == 0
    assert df['a'].min(ignore_nan=True) == 0
    assert df['a'].min(ignore_nan=False) == 0

    assert df['a'].max() == 0
    assert df['a'].max(ignore_nan=True) == 0
    assert df['a'].max(ignore_nan=False) == 0


def test_with_nulls(spark):
    data = {
        'a': (1, 3, None, -1, 0),
        'b': (1, 3, None, None, 0),
        'c': (1, None, None, None, 0)
    }
    df = FlickerDataFrame.from_dict(spark, data)
    assert df['a'].min() == -1
    assert df['a'].min(ignore_nan=True) == -1
    assert df['a'].min(ignore_nan=False) == -1

    assert df['b'].min() == 0
    assert df['b'].min(ignore_nan=True) == 0
    assert df['b'].min(ignore_nan=False) == 0

    assert df['c'].min() == 0
    assert df['c'].min(ignore_nan=True) == 0
    assert df['c'].min(ignore_nan=False) == 0

    assert df['a'].max() == 3
    assert df['a'].max(ignore_nan=True) == 3
    assert df['a'].max(ignore_nan=False) == 3

    assert df['b'].max() == 3
    assert df['b'].max(ignore_nan=True) == 3
    assert df['b'].max(ignore_nan=False) == 3

    assert df['c'].max() == 1
    assert df['c'].max(ignore_nan=True) == 1
    assert df['c'].max(ignore_nan=False) == 1


def test_with_nans(spark):
    data = [
        (1.0, 1.0, 1.0),
        (3.0, 3.0, np.nan),
        (np.nan, np.nan, np.nan),
        (-1.0, np.nan, np.nan),
        (0.0, 0.0, 0.0)
    ]
    df = FlickerDataFrame(spark.createDataFrame(data, list('abc')))
    assert set(df.dtypes.values()) == {'double'}
    assert df['a'].min(ignore_nan=True) == -1.0
    assert df['a'].min(ignore_nan=False) == -1.0
    assert df['b'].min(ignore_nan=True) == 0.0
    assert df['b'].min(ignore_nan=False) == 0.0
    assert df['c'].min(ignore_nan=True) == 0.0
    assert df['c'].min(ignore_nan=False) == 0.0

    assert df['a'].max(ignore_nan=True) == 3.0
    assert np.isnan(df['a'].max(ignore_nan=False))
    assert df['b'].max(ignore_nan=True) == 3.0
    assert np.isnan(df['b'].max(ignore_nan=False))
    assert df['c'].max(ignore_nan=True) == 1.0
    assert np.isnan(df['c'].max(ignore_nan=False))


def test_timestamp_no_nulls(spark):
    t = datetime(2023, 1, 1)
    dt = timedelta(days=1)
    df = FlickerDataFrame.from_rows(spark, [(t - dt, 1), (t, 2), (t + dt, 3)], names=['t', 'n'])
    assert df['t'].min() == t - dt
    assert df['t'].max() == t + dt


def test_timestamp_with_nulls(spark):
    t = datetime(2023, 1, 1)
    dt = timedelta(days=1)
    df = FlickerDataFrame.from_rows(spark, [(t - dt, 1), (t, 2), (t + dt, 3), (None, None)], names=['t', 'n'])
    assert df['t'].min() == t - dt
    assert df['t'].max() == t + dt


def test_boolean_no_nulls(spark):
    df = FlickerDataFrame.from_rows(spark, [(True, 1), (False, 2), (True, 3)], names=['t', 'n'])
    assert df['t'].min() is False
    assert df['t'].max() is True


def test_boolean_with_nulls(spark):
    df = FlickerDataFrame.from_rows(spark, [(True, 1), (False, 2), (True, 3), (None, None)], names=['t', 'n'])
    assert df['t'].min() is False
    assert df['t'].max() is True


def test_chains(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=3, ncols=1, names=['zero'], fill='zero')
    df['one'] = 1

    # This chain fails if the code is not written to handle generalities.
    assert (df['one'] - df['zero']).min() == 1
    assert (df['one'] - df['zero']).max() == 1
