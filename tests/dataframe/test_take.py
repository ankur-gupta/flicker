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
from pyspark.sql import Row
from flicker import FlickerDataFrame


def test_take_n_rows(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    rows = df.take(1, convert_to_dict=False)
    assert len(rows) == 1
    for element in rows:
        assert isinstance(element, Row)


def test_take_n_dicts(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    rows = df.take(1, convert_to_dict=True)
    assert len(rows) == 1
    for element in rows:
        assert isinstance(element, dict)


def test_take_all_rows(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    rows = df.take(None, convert_to_dict=False)
    assert len(rows) == df.nrows
    for element in rows:
        assert isinstance(element, Row)


def test_take_more_than_enough_rows(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    rows = df.take(df.nrows + 10, convert_to_dict=False)
    assert len(rows) == df.nrows
    for element in rows:
        assert isinstance(element, Row)
