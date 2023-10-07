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
from flicker import FlickerDataFrame
from flicker.udf import keys_udf, string_keys_udf, integer_keys_udf, _keys


def test__keys():
    assert _keys(None) == []
    assert _keys({}) == []
    assert _keys({'a': 1}) == ['a']
    assert _keys({'a': 1, 'b': 3}) == ['a', 'b']
    assert _keys({1: 1, 2: 3}) == [1, 2]

    with pytest.raises(AttributeError):
        _keys([])
    with pytest.raises(AttributeError):
        _keys(1)


def test_string_keys_works(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [{'a': 1}, {'b': 1}, {'c': 1, 'd': 6}]
    })
    df[f'keys_a'] = string_keys_udf(df['a']._column)

    pdf = df.to_pandas()
    assert pdf['keys_a'].to_list() == [['a'], ['b'], ['c', 'd']]


def test_string_keys_failure(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3]
    })
    df[f'keys_a'] = string_keys_udf(df['a']._column)

    with pytest.raises(Exception):
        df()


def test_integer_keys_works(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [{1: 1}, {2: 1}, {3: 1, 4: 6}]
    })
    df[f'keys_a'] = integer_keys_udf(df['a']._column)

    pdf = df.to_pandas()
    assert pdf['keys_a'].to_list() == [[1], [2], [3, 4]]


def test_integer_keys_failure(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3]
    })
    df[f'keys_a'] = integer_keys_udf(df['a']._column)

    with pytest.raises(Exception):
        df()


def test_keys_works(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [{'a': 1}, {'b': 1}, {'c': 1, 'd': 6}]
    })
    df[f'keys_a'] = keys_udf(df['a']._column)

    pdf = df.to_pandas()
    assert pdf['keys_a'].to_list() == [['a'], ['b'], ['c', 'd']]


def test_keys_failure(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3]
    })
    df[f'keys_a'] = keys_udf(df['a']._column)

    with pytest.raises(Exception):
        df()
