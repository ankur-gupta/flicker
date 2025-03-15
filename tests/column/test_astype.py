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
import datetime
from string import ascii_lowercase

# https://spark.apache.org/docs/latest/sql-ref-datatypes.html
from pyspark.sql.types import DoubleType, TimestampType, StringType, BooleanType, FloatType, IntegerType

from flicker import FlickerDataFrame


def test_basic_usage(spark):
    df = FlickerDataFrame(spark.createDataFrame([(x, f'{ascii_lowercase[x]}') for x in range(5)], 'a INT, b STRING'))
    assert df['a'].dtype == 'int'
    assert df['b'].dtype == 'string'

    df['a_float'] = df['a'].astype(float)
    df['a_double'] = df['a'].astype('double')
    df['a_DoubleType'] = df['a'].astype(DoubleType())
    df['a_FloatType'] = df['a'].astype(FloatType())
    assert df['a_float'].dtype == 'double'
    assert df['a_double'].dtype == 'double'
    assert df['a_DoubleType'].dtype == 'double'
    assert df['a_FloatType'].dtype == 'float'

    df['a_str'] = df['a'].astype(str)
    df['a_string'] = df['a'].astype('string')
    df['a_StringType'] = df['a'].astype(StringType())
    assert df['a_str'].dtype == 'string'
    assert df['a_string'].dtype == 'string'
    assert df['a_StringType'].dtype == 'string'

    df['a_bool'] = df['a'].astype(bool)
    df['a_boolean'] = df['a'].astype('boolean')
    df['a_BooleanType'] = df['a'].astype(BooleanType())
    assert df['a_bool'].dtype == 'boolean'
    assert df['a_boolean'].dtype == 'boolean'
    assert df['a_BooleanType'].dtype == 'boolean'

    df['b_int'] = df['b'].astype(int)
    df['b_integer'] = df['a'].astype('bigint')
    df['b_IntegerType'] = df['a'].astype(IntegerType())
    assert df['b_int'].dtype == 'bigint'
    assert df['b_integer'].dtype == 'bigint'
    assert df['b_IntegerType'].dtype == 'int'

    df['b_float'] = df['b'].astype(float)
    df['b_double'] = df['a'].astype('double')
    assert df['b_float'].dtype == 'double'
    assert df['b_double'].dtype == 'double'

    df['b_bool'] = df['b'].astype(bool)
    df['b_boolean'] = df['a'].astype('boolean')
    assert df['b_bool'].dtype == 'boolean'
    assert df['b_boolean'].dtype == 'boolean'


def test_time_dtype(spark):
    df = FlickerDataFrame(spark.createDataFrame([(x, f'2025-01-{x + 1}') for x in range(5)], 'a INT, t STRING'))
    df['t_as_timestamp'] = df['t'].astype('timestamp')
    df['t_as_datetime'] = df['t'].astype(datetime.datetime)
    df['t_as_TimestampType'] = df['t'].astype(TimestampType())
    assert df.dtypes['t'] == 'string'
    assert df.dtypes['t_as_timestamp'] == 'timestamp'
    assert df.dtypes['t_as_datetime'] == 'timestamp'
    assert df.dtypes['t_as_TimestampType'] == 'timestamp'


def test_void(spark):
    df = FlickerDataFrame(spark.createDataFrame([(x, f'{ascii_lowercase[x]}') for x in range(5)], 'a INT, b STRING'))
    df['c'] = None
    assert df['c'].dtype == 'void'
    df['d'] = df['c'].astype('void')
    assert df['d'].dtype == 'void'

    with pytest.raises(Exception):
        # df['a'].astype('void')  # This doesn't throw an Exception
        df['e'] = df['a'].astype('void')  # This exception comes from pyspark, not flicker


def test_unsupported_type(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, names=list('ab'))
    with pytest.raises(Exception):
        df['a'].astype('some-unknown-type')
    with pytest.raises(Exception):
        df['a'].astype(type({}))
