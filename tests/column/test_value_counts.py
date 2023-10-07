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


def test_count_column_fails(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['count'], fill='zero')
    with pytest.raises(KeyError):
        df['count'].value_counts()


def test_correct_counts_basic(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    counts = df['zero'].value_counts(sort=True, ascending=False, drop_null=False, normalize=False, n=None)
    assert isinstance(counts, FlickerDataFrame)
    assert set(counts.names) == {'zero', 'count'}
    assert counts.take(None, convert_to_dict=True)[0] == {'zero': 0, 'count': 3}


def test_correct_counts_normalized(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    counts = df['zero'].value_counts(sort=True, ascending=False, drop_null=False, normalize=True, n=None)
    assert isinstance(counts, FlickerDataFrame)
    assert set(counts.names) == {'zero', 'count'}
    assert counts.take(None, convert_to_dict=True)[0] == {'zero': 0, 'count': 1.0}


def test_correct_counts_sort_descending(spark):
    df = FlickerDataFrame.from_columns(spark, [[1, 2, 1, 2, 2, 2, 2]], ['x'])
    counts = df['x'].value_counts(sort=True, ascending=False, drop_null=False, normalize=False, n=None)
    assert isinstance(counts, FlickerDataFrame)
    assert set(counts.names) == {'x', 'count'}
    counts_as_rows = counts.take(None, convert_to_dict=True)
    assert counts_as_rows == [
        {'x': 2, 'count': 5},
        {'x': 1, 'count': 2}
    ]


def test_correct_counts_sort_ascending(spark):
    df = FlickerDataFrame.from_columns(spark, [[1, 2, 1, 2, 2, 2, 2]], ['x'])
    counts = df['x'].value_counts(sort=True, ascending=True, drop_null=False, normalize=False, n=None)
    assert isinstance(counts, FlickerDataFrame)
    assert set(counts.names) == {'x', 'count'}
    counts_as_rows = counts.take(None, convert_to_dict=True)
    assert counts_as_rows == [
        {'x': 1, 'count': 2},
        {'x': 2, 'count': 5}
    ]


def test_correct_counts_with_dropnull_active(spark):
    df = FlickerDataFrame.from_columns(spark, [[1, None, 2, None, 1, 2, 2, 2, 2]], ['x'])
    counts = df['x'].value_counts(sort=True, ascending=False, drop_null=True, normalize=False, n=None)
    assert isinstance(counts, FlickerDataFrame)
    assert set(counts.names) == {'x', 'count'}
    counts_as_rows = counts.take(None, convert_to_dict=True)
    assert counts_as_rows == [
        {'x': 2, 'count': 5},
        {'x': 1, 'count': 2}
    ]

    norm_counts = df['x'].value_counts(sort=True, ascending=False, drop_null=True, normalize=True, n=None)
    assert isinstance(norm_counts, FlickerDataFrame)
    assert set(norm_counts.names) == {'x', 'count'}
    norm_counts_as_rows = norm_counts.take(None, convert_to_dict=True)
    assert norm_counts_as_rows == [
        {'x': 2, 'count': 5 / 9},
        {'x': 1, 'count': 2 / 9}
    ]


def test_correct_counts_with_dropnull_inactive(spark):
    df = FlickerDataFrame.from_columns(spark, [[1, None, 2, None, 1, 2, 2, 2, 2]], ['x'])
    counts = df['x'].value_counts(sort=True, ascending=False, drop_null=False, normalize=False, n=None)
    assert isinstance(counts, FlickerDataFrame)
    assert set(counts.names) == {'x', 'count'}
    counts_as_rows = counts.take(None, convert_to_dict=True)
    option1 = counts_as_rows == [
        {'x': 2, 'count': 5},
        {'x': 1, 'count': 2},
        {'x': None, 'count': 2}
    ]
    option2 = counts_as_rows == [
        {'x': 2, 'count': 5},
        {'x': None, 'count': 2},
        {'x': 1, 'count': 2}
    ]
    assert option1 or option2

    norm_counts = df['x'].value_counts(sort=True, ascending=False, drop_null=False, normalize=True, n=None)
    assert isinstance(norm_counts, FlickerDataFrame)
    assert set(norm_counts.names) == {'x', 'count'}
    norm_counts_as_rows = norm_counts.take(None, convert_to_dict=True)
    option1 = norm_counts_as_rows == [
        {'x': 2, 'count': 5 / 9},
        {'x': 1, 'count': 2 / 9},
        {'x': None, 'count': 2 / 9}
    ]
    option2 = norm_counts_as_rows == [
        {'x': 2, 'count': 5 / 9},
        {'x': None, 'count': 2 / 9},
        {'x': 1, 'count': 2 / 9}
    ]
    assert option1 or option2


def test_correct_counts_with_large_n(spark):
    values = ([1] * 10) + ([None] * 10) + ([2] * 10)
    df = FlickerDataFrame.from_columns(spark, [values], ['x'])

    counts = df['x'].value_counts(sort=True, ascending=False, drop_null=False, normalize=False, n=None)
    assert isinstance(counts, FlickerDataFrame)
    assert set(counts.names) == {'x', 'count'}
    counts_as_rows = counts.take(None, convert_to_dict=True)
    assert len(counts_as_rows) == 3
    assert {'x': 1, 'count': 10} in counts_as_rows
    assert {'x': None, 'count': 10} in counts_as_rows
    assert {'x': 2, 'count': 10} in counts_as_rows

    counts5 = df['x'].value_counts(sort=True, ascending=False, drop_null=False, normalize=False, n=5)
    assert isinstance(counts5, FlickerDataFrame)
    assert set(counts5.names) == {'x', 'count'}
    counts5_as_rows = counts5.take(None, convert_to_dict=True)
    assert counts5_as_rows == [
        {'x': 1, 'count': 5},
    ]

    norm_counts5 = df['x'].value_counts(sort=True, ascending=False, drop_null=False, normalize=True, n=5)
    assert isinstance(norm_counts5, FlickerDataFrame)
    assert set(norm_counts5.names) == {'x', 'count'}
    norm_counts5_as_rows = norm_counts5.take(None, convert_to_dict=True)
    assert norm_counts5_as_rows == [
        {'x': 1, 'count': 5 / 30},
    ]

    counts15 = df['x'].value_counts(sort=True, ascending=False, drop_null=False, normalize=False, n=15)
    assert isinstance(counts15, FlickerDataFrame)
    assert set(counts15.names) == {'x', 'count'}
    counts15_as_rows = counts15.take(None, convert_to_dict=True)
    assert len(counts15_as_rows) == 2
    assert {'x': 1, 'count': 10} in counts15_as_rows
    assert {'x': None, 'count': 5} in counts15_as_rows

    norm_counts15 = df['x'].value_counts(sort=True, ascending=False, drop_null=False, normalize=True, n=15)
    assert isinstance(norm_counts15, FlickerDataFrame)
    assert set(norm_counts15.names) == {'x', 'count'}
    norm_counts15_as_rows = norm_counts15.take(None, convert_to_dict=True)
    assert len(norm_counts15_as_rows) == 2
    assert {'x': 1, 'count': 10 / 30} in norm_counts15_as_rows
    assert {'x': None, 'count': 5 / 30} in norm_counts15_as_rows

    norm_counts15_no_nulls = df['x'].value_counts(sort=True, ascending=False, drop_null=True, normalize=True, n=15)
    assert isinstance(norm_counts15_no_nulls, FlickerDataFrame)
    assert set(norm_counts15_no_nulls.names) == {'x', 'count'}
    norm_counts15_no_nulls_as_rows = norm_counts15_no_nulls.take(None, convert_to_dict=True)
    assert len(norm_counts15_no_nulls_as_rows) == 2
    assert {'x': 1, 'count': 10 / 30} in norm_counts15_no_nulls_as_rows
    assert {'x': 2, 'count': 5 / 30} in norm_counts15_no_nulls_as_rows


def test_chains(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=3, ncols=1, names=['zero'], fill='zero')
    df['one'] = 1

    # This chain fails if the code is not written to handle generalities.
    assert isinstance((df['one'] - df['zero']).value_counts(), FlickerDataFrame)
    assert isinstance((df['one'] + df['one'] * df['zero']).value_counts(), FlickerDataFrame)
    assert isinstance((df['one'] ** df['one']).value_counts(), FlickerDataFrame)

    # Correctness
    one_minus_zero_counts = (df['one'] - df['zero']).value_counts(sort=True, ascending=False, drop_null=False,
                                                                  normalize=False, n=None)
    one_minus_zero_counts_as_rows = one_minus_zero_counts.take(None, convert_to_dict=True)
    assert one_minus_zero_counts_as_rows[0]['(one - zero)'] == 1
    assert one_minus_zero_counts_as_rows[0]['count'] == 3
