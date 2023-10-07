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


def test_left_mul_column(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1

    df['one_times_zero'] = df['one'] * df['zero']
    df['zero_times_one'] = df['zero'] * df['one']._column
    assert df['one_times_zero'].isin([0]).all()
    assert df['zero_times_one'].isin([0]).all()

    # We'll test `FlickerColumn.__mul__` manually
    assert df['one'].__mul__(df['zero']).isin([0]).all()
    assert df['one'].__mul__(df['zero']._column).isin([0]).all()


def test_right_mul_column(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1

    # We'll test `FlickerColumn.__rmul__` manually
    assert df['one'].__rmul__(df['zero']).isin([0]).all()
    assert df['one'].__rmul__(df['zero']._column).isin([0]).all()


def test_left_mul_literal(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1

    zero_times_one = df['zero'] * 1
    assert isinstance(zero_times_one, FlickerColumn)
    assert zero_times_one.isin([0]).all()

    df['zero_times_one'] = df['zero'] * 1
    assert df['zero_times_one'].isin([0]).all()

    # We'll test `FlickerColumn.__mul__` manually
    assert df['one'].__mul__(1).isin([1]).all()


def test_right_mul_literal(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1

    # Python only calls right-sided operations such as __rmul__ when the left operand does not have an __mul__.
    # https://docs.python.org/3/reference/datamodel.html#object.__rmul__
    # Since `10` doesn't have a `__mul__` method, pyspark.sql.Column.__rmul__ is called.
    ten_times_zero = 10 * df['zero']
    assert isinstance(ten_times_zero, FlickerColumn)
    assert ten_times_zero.isin([0]).all()

    df['ten_times_zero'] = 10 * df['zero']
    assert df['ten_times_zero'].isin([0]).all()

    # We'll test `FlickerColumn.__rmul__` manually
    # 1 * one
    assert df['one'].__rmul__(1).isin([1]).all()
