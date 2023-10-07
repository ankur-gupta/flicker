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


def test_basic_usage(spark):
    records = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 25}, {'name': 'Charlie', 'age': 25}]
    df = FlickerDataFrame.from_records(spark, records)
    assert isinstance(df, FlickerDataFrame)
    assert set(df.names) == {'name', 'age'}
    assert df.shape == (3, 2)
    for name in df.names:
        assert isinstance(name, str)


def test_column_number_mismatch(spark):
    records = [{'name': 'Alice', 'age': 25}, {'name': 'Bob'}, {'name': 'Charlie', 'age': 25}]
    df = FlickerDataFrame.from_records(spark, records)
    assert isinstance(df, FlickerDataFrame)
    assert set(df.names) == {'name', 'age'}
    assert df.shape == (3, 2)
    for name in df.names:
        assert isinstance(name, str)

