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
from string import ascii_lowercase
from flicker import FlickerDataFrame


def test_basic_usage(spark):
    df = FlickerDataFrame(spark.createDataFrame([(x, f'{ascii_lowercase[x]}') for x in range(5)], 'a INT, b STRING'))
    assert df['a'].dtype == 'int'
    assert df['b'].dtype == 'string'

    df['a_float'] = df['a'].astype(float)
    df['a_double'] = df['a'].astype('double')
    assert df['a_float'].dtype == 'double'
    assert df['a_double'].dtype == 'double'

    df['a_str'] = df['a'].astype(str)
    df['a_string'] = df['a'].astype('string')
    assert df['a_str'].dtype == 'string'
    assert df['a_string'].dtype == 'string'

    df['a_bool'] = df['a'].astype(bool)
    df['a_boolean'] = df['a'].astype('boolean')
    assert df['a_bool'].dtype == 'boolean'
    assert df['a_boolean'].dtype == 'boolean'

    df['b_int'] = df['b'].astype(int)
    df['b_integer'] = df['a'].astype('bigint')
    assert df['b_int'].dtype == 'bigint'
    assert df['b_integer'].dtype == 'bigint'

    df['b_float'] = df['b'].astype(float)
    df['b_double'] = df['a'].astype('double')
    assert df['b_float'].dtype == 'double'
    assert df['b_double'].dtype == 'double'

    df['b_bool'] = df['b'].astype(bool)
    df['b_boolean'] = df['a'].astype('boolean')
    assert df['b_bool'].dtype == 'boolean'
    assert df['b_boolean'].dtype == 'boolean'

    # FIXME: Test datetime/timestamp


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
