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


def test_single_column_sort(spark):
    rows = [('Alice', 25), ('Bob', 30), ('Charlie', 35), ('Delta', 15)]
    names = ['name', 'age']
    df = FlickerDataFrame.from_rows(spark, rows, names)

    # Age Ascending
    age_asc = [
        tuple(row.values())
        for row in df.sort(['age'], ascending=True).take(None, convert_to_dict=True)
    ]
    rows.sort(key=lambda x: x[1], reverse=False)
    assert age_asc == rows

    # Age Descending
    age_desc = [
        tuple(row.values())
        for row in df.sort(['age'], ascending=False).take(None, convert_to_dict=True)
    ]
    rows.sort(key=lambda x: x[1], reverse=True)
    assert age_desc == rows

    # Name Ascending
    name_asc = [
        tuple(row.values())
        for row in df.sort(['name'], ascending=True).take(None, convert_to_dict=True)
    ]
    rows.sort(key=lambda x: x[0], reverse=False)
    assert name_asc == rows

    # Name Descending
    name_desc = [
        tuple(row.values())
        for row in df.sort(['name'], ascending=False).take(None, convert_to_dict=True)
    ]
    rows.sort(key=lambda x: x[0], reverse=True)
    assert name_desc == rows


def test_multi_column_sort(spark):
    rows = [('Alice', 25), ('Bob', 30), ('Alice', 45), ('Charlie', 35), ('Delta', 15), ('Bob', 45), ('Charlie', 15)]
    names = ['name', 'age']
    df = FlickerDataFrame.from_rows(spark, rows, names)

    name_age_asc = [
        tuple(row.values())
        for row in df.sort(['name', 'age'], ascending=True).take(None, convert_to_dict=True)
    ]
    rows.sort(reverse=False)
    assert name_age_asc == rows

    name_age_desc = [
        tuple(row.values())
        for row in df.sort(['name', 'age'], ascending=False).take(None, convert_to_dict=True)
    ]
    rows.sort(reverse=True)
    assert name_age_desc == rows

    age_name_asc = [
        tuple(row.values())
        for row in df.sort(['age', 'name'], ascending=True).take(None, convert_to_dict=True)
    ]
    rows.sort(key=lambda x: x[::-1], reverse=False)
    assert age_name_asc == rows

    age_name_desc = [
        tuple(row.values())
        for row in df.sort(['age', 'name'], ascending=False).take(None, convert_to_dict=True)
    ]
    rows.sort(key=lambda x: x[::-1], reverse=True)
    assert age_name_desc == rows

