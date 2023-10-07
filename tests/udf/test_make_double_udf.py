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
from flicker.udf import make_double_udf, generate_extract_value_by_key


def test_basic_usage(spark):
    rows = [({'value': 1.0}, 1), ({'value': 2.0}, 2), ({'value': 3.0}, 3)]
    df = FlickerDataFrame.from_rows(spark, rows, names=['x', 'n'])
    double_value_udf = make_double_udf(generate_extract_value_by_key('value'))
    df['value'] = double_value_udf(df['x']._column)
    assert [row['value'] for row in df[['value']].take(None, convert_to_dict=True)] == [1.0, 2.0, 3.0]


def test_missing_key(spark):
    rows = [({'value': 1.0}, 1), ({'no-value': 2.0}, 2), ({'value': 3.0}, 3)]
    df = FlickerDataFrame.from_rows(spark, rows, names=['x', 'n'])
    double_value_udf = make_double_udf(generate_extract_value_by_key('value'))
    df['value'] = double_value_udf(df['x']._column)
    assert [row['value'] for row in df[['value']].take(None, convert_to_dict=True)] == [1.0, None, 3.0]


def test_missing_field(spark):
    rows = [({'value': 1.0}, 1), (None, 2), ({'value': 3.0}, 3)]
    df = FlickerDataFrame.from_rows(spark, rows, names=['x', 'n'])
    double_value_udf = make_double_udf(generate_extract_value_by_key('value'))
    df['value'] = double_value_udf(df['x']._column)
    assert [row['value'] for row in df[['value']].take(None, convert_to_dict=True)] == [1.0, None, 3.0]

# We cannot test failure of `make_double_udf` in action because spark won't even let us create a dataframe without
# a data types matching in every row.
# def test_failure(spark):
#     rows = [({'value': 1.0}, 1), ([], 2), ({'value': 2.0}, 3)]
#     FlickerDataFrame.from_rows(spark, rows, names=['x', 'n'])  # Fails because of spark
