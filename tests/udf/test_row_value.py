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
from datetime import datetime, timedelta
from flicker import FlickerDataFrame
from flicker.udf import (_row_value, string_row_value_udf, boolean_row_value_udf, integer_row_value_udf,
                         long_row_value_udf, double_row_value_udf, float_row_value_udf, timestamp_row_value_udf)


def test__row_value():
    assert _row_value(None) is None
    assert _row_value({'value': 1}) == 1
    assert _row_value({'value': 'a'}) == 'a'
    assert _row_value({'value': 3.14}) == 3.14
    assert _row_value({'value': [1, 2, 3]}) == [1, 2, 3]
    assert _row_value({'value': []}) == []
    assert _row_value({'value': {}}) == {}
    assert _row_value({'value': {'a': 1}}) == {'a': 1}
    assert _row_value({'value': None}) is None

    with pytest.raises(Exception):
        _row_value(1)
    with pytest.raises(Exception):
        _row_value(3.123)
    with pytest.raises(Exception):
        _row_value('abc')
    with pytest.raises(Exception):
        _row_value({})
    with pytest.raises(Exception):
        _row_value({'no-value': 1})


def test_string_row_value_udf(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [{'value': 'a'}, {'value': 'b'}, {'value': 'abc', 'd': 6}],
        'b': [{'value': 'a'}, {'no-value': 'b'}, {'value': 'abc', 'd': 6}]
    })
    df['row_value_a'] = string_row_value_udf(df['a']._column)
    pdf = df.to_pandas()
    assert pdf['row_value_a'].to_list() == ['a', 'b', 'abc']

    df['row_value_b'] = string_row_value_udf(df['b']._column)
    with pytest.raises(Exception):
        df()


def test_boolean_row_value_udf(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [{'value': True}, {'value': True}, {'value': False, 'd': 6}],
        'b': [{'value': True}, {'no-value': True}, {'value': False, 'd': 6}]
    })
    df['row_value_a'] = boolean_row_value_udf(df['a']._column)
    pdf = df.to_pandas()
    assert pdf['row_value_a'].to_list() == [True, True, False]

    df['row_value_b'] = boolean_row_value_udf(df['b']._column)
    with pytest.raises(Exception):
        df()


def test_integer_row_value_udf(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [{'value': 1}, {'value': 2}, {'value': 3, 'd': 6}],
        'b': [{'value': 1}, {'no-value': 2}, {'value': 3, 'd': 6}]
    })
    df['row_value_a'] = integer_row_value_udf(df['a']._column)
    pdf = df.to_pandas()
    assert pdf['row_value_a'].to_list() == [1, 2, 3]

    df['row_value_b'] = integer_row_value_udf(df['b']._column)
    with pytest.raises(Exception):
        df()


def test_long_row_value_udf(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [{'value': 1}, {'value': 2}, {'value': 3, 'd': 6}],
        'b': [{'value': 1}, {'no-value': 2}, {'value': 3, 'd': 6}]
    })
    df['row_value_a'] = long_row_value_udf(df['a']._column)
    pdf = df.to_pandas()
    assert pdf['row_value_a'].to_list() == [1, 2, 3]

    df['row_value_b'] = long_row_value_udf(df['b']._column)
    with pytest.raises(Exception):
        df()


def test_double_row_value_udf(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [{'value': 1.1}, {'value': 2.2}, {'value': 3.43, 'd': 6}],
        'b': [{'value': 1.1}, {'no-value': 2.44}, {'value': 3.634, 'd': 6}]
    })
    df['row_value_a'] = double_row_value_udf(df['a']._column)
    pdf = df.to_pandas()
    assert pdf['row_value_a'].to_list() == [1.1, 2.2, 3.43]

    df['row_value_b'] = double_row_value_udf(df['b']._column)
    with pytest.raises(Exception):
        df()


def test_float_row_value_udf(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [{'value': 1.0}, {'value': 2.0}, {'value': 3.0, 'd': 6}],
        'b': [{'value': 1.0}, {'no-value': 2.44}, {'value': 3.634, 'd': 6}]
    })
    df['row_value_a'] = float_row_value_udf(df['a']._column)
    pdf = df.to_pandas()
    assert pdf['row_value_a'].to_list() == [1.0, 2.0, 3.0]

    df['row_value_b'] = float_row_value_udf(df['b']._column)
    with pytest.raises(Exception):
        df()


def test_timestamp_row_value_udf_fails(spark):
    t = datetime(2023, 1, 1)
    dt = timedelta(days=1)

    # Ensure failure happens when 'value' key is not found
    data = [({'value': t - dt}, 1), ({'no-value': t}, 2), ({'value': t + dt}, 3)]
    df = FlickerDataFrame(spark.createDataFrame(data, schema=['t', 'x']))
    df['row_value_t'] = timestamp_row_value_udf(df['t']._column)
    with pytest.raises(Exception):
        df()


def test_timestamp_row_value_udf_works(spark):
    t = datetime(2023, 1, 1)
    dt = timedelta(days=1)

    # This fails because the values for the keys 'value' and 'a' have different data types.
    # data = [({'value': t - dt}, 1), ({'value': t}, 2), ({'value': t + dt, 'a': 12}, 3)]
    data = [({'value': t - dt}, 1), ({'value': t}, 2), ({'value': t + dt}, 3)]
    df = FlickerDataFrame(spark.createDataFrame(data, schema=['t', 'x']))
    df['row_value_t'] = timestamp_row_value_udf(df['t']._column)
    out = [row['row_value_t'] for row in df[['row_value_t']].take(None, convert_to_dict=True)]
    assert out == [t - dt, t, t + dt]

    # Avoid using this in Python 3.9 because of timestamp conversion error
    # pdf = df.to_pandas()
