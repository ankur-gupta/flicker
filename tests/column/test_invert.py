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
    df = FlickerDataFrame.from_shape(spark, 5, 2, names=['a', 'b'], fill='rand')
    df['a'] = True
    df['c'] = ~df['a']
    assert df['a'].all()
    assert not df['c'].all()
    assert (df['a'] != df['c']).all()


def test_non_boolean_type_double(spark):
    df = FlickerDataFrame.from_shape(spark, 5, 2, names=['a', 'b'], fill='rand')
    with pytest.raises(Exception):
        ~df['a']


def test_non_boolean_type_int_string(spark):
    df = FlickerDataFrame(spark.createDataFrame([(x, f'{ascii_lowercase[x]}') for x in range(5)], 'a INT, b STRING'))
    with pytest.raises(Exception):
        ~df['a']
    with pytest.raises(Exception):
        ~df['b']
