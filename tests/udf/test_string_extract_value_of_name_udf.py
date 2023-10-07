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
from flicker import FlickerDataFrame
from flicker.udf import string_extract_value_of_name_udf


def test_basic_usage(spark):
    rows = [({'name': 'a'}, 1), ({'name': 'b'}, 2), ({'name': 'c'}, 3)]
    df = FlickerDataFrame.from_rows(spark, rows, names=['x', 'n'])
    df['name'] = string_extract_value_of_name_udf(df['x']._column)
    assert [row['name'] for row in df[['name']].take(None, convert_to_dict=True)] == ['a', 'b', 'c']


def test_missing_key(spark):
    rows = [({'name': 'a'}, 1), ({'no-name': 'b'}, 2), ({'name': 'c'}, 3)]
    df = FlickerDataFrame.from_rows(spark, rows, names=['x', 'n'])
    df['name'] = string_extract_value_of_name_udf(df['x']._column)
    assert [row['name'] for row in df[['name']].take(None, convert_to_dict=True)] == ['a', None, 'c']


def test_missing_field(spark):
    rows = [({'name': 'a'}, 1), (None, 2), ({'name': 'c'}, 3)]
    df = FlickerDataFrame.from_rows(spark, rows, names=['x', 'n'])
    df['name'] = string_extract_value_of_name_udf(df['x']._column)
    assert [row['name'] for row in df[['name']].take(None, convert_to_dict=True)] == ['a', None, 'c']
