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
from flicker import FlickerDataFrame, FlickerColumn


def test_left_add_column(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1

    df['one_zero'] = df['one'] + df['zero']
    df['zero_one'] = df['zero'] + df['one']._column
    assert df['one_zero'].isin([1]).all()
    assert df['zero_one'].isin([1]).all()

    # We'll test `FlickerColumn.__add__` manually
    assert df['one'].__add__(df['zero']).isin([1]).all()
    assert df['one'].__add__(df['zero']._column).isin([1]).all()


def test_right_add_column(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1

    # Fails because python uses `pyspark.sql.Column.__add__` instead of `FlickerColumn._radd__`.
    # Python only calls right-sided operations such as __radd__ when the left operand does not have an __add__.
    # https://docs.python.org/3/reference/datamodel.html#object.__radd__
    # df['zero_one'] = df['zero']._column + df['one']

    # We'll test `FlickerColumn.__radd__` manually
    assert df['one'].__radd__(df['zero']).isin([1]).all()
    assert df['one'].__radd__(df['zero']._column).isin([1]).all()


def test_left_add_literal(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1

    zero_plus_one = df['zero'] + 1
    assert isinstance(zero_plus_one, FlickerColumn)
    assert zero_plus_one.isin([1]).all()

    df['zero_plus_one'] = df['zero'] + 1
    assert df['zero_plus_one'].isin([1]).all()

    # We'll test `FlickerColumn.__add__` manually
    assert df['one'].__add__(1).isin([2]).all()


def test_right_add_literal(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1

    # Python only calls right-sided operations such as __radd__ when the left operand does not have an __add__.
    # https://docs.python.org/3/reference/datamodel.html#object.__radd__
    # Since `1` doesn't have a `__add__` method, this works.
    one_plus_zero = 1 + df['zero']
    assert isinstance(one_plus_zero, FlickerColumn)
    assert one_plus_zero.isin([1]).all()

    df['one_plus_zero'] = 1 + df['zero']
    assert df['one_plus_zero'].isin([1]).all()

    # We'll test `FlickerColumn.__radd__` manually
    assert df['one'].__radd__(1).isin([2]).all()
