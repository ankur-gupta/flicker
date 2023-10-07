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
from flicker.udf import len_udf, _len


def test_with_nulls(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3],
        'b': ['a', 'bc', None],
        'c': [{'a': 1}, {'b': 1}, {'c': 1, 'd': 6}]
    })
    df['d'] = None
    df['e'] = 3.14

    # Compute lengths for all columns
    for name in df.names:
        df[f'len_{name}'] = len_udf(df[name]._column)

    pdf = df.to_pandas()
    assert all(pdf['len_a'] == 1)
    assert all(pdf['len_b'].to_numpy() == [1, 2, 0])
    assert all(pdf['len_c'].to_numpy() == [1, 1, 2])
    assert all(pdf['len_d'] == 0)
    assert all(pdf['len_e'] == 1)


def test__len():
    assert _len(None) == 0

    assert _len(1) == 1
    assert _len(3.14) == 1

    assert _len('') == 0
    assert _len('a') == 1
    assert _len('abc') == 3

    assert _len([]) == 0
    assert _len(tuple()) == 0
    assert _len({}) == 0
    assert _len(set()) == 0

    assert _len(['abc']) == 1
    assert _len(['abc', 1.23]) == 2
    assert _len([-0.78, 1.23]) == 2


