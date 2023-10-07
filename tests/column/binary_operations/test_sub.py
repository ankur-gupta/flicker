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


def test_left_sub_column(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1

    df['one_minus_zero'] = df['one'] - df['zero']
    df['zero_minus_one'] = df['zero'] - df['one']._column
    assert df['one_minus_zero'].isin([1]).all()
    assert df['zero_minus_one'].isin([-1]).all()

    # We'll test `FlickerColumn.__sub__` manually
    assert df['one'].__sub__(df['zero']).isin([1]).all()
    assert df['one'].__sub__(df['zero']._column).isin([1]).all()


# pyspark.sql.Column.__rsub__ calls the following snippet of code and is therefore only applicable to literals:
# https://github.com/apache/spark/blob/c0d9ca3be14cb0ec8d8f9920d3ecc4aac3cf5adc/python/pyspark/sql/column.py#L184-L196
# def _reverse_op(
# ...
#     """Create a method for binary operator (this object is on right side)"""
# ...
#         jother = _create_column_from_literal(other)  <- this is how `Column.__rxxx__(other)` is processed
#         jc = getattr(jother, name)(self._jc)
# ...

# def test_right_sub_column(spark):
#     df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
#     df['one'] = 1
#
#     # We'll test `FlickerColumn.__rsub__` manually
#     # Fails!
#     assert df['one'].__rsub__(df['zero']).isin([1]).all()
#     assert df['one'].__rsub__(df['zero']._column).isin([1]).all()


def test_left_sub_literal(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1

    zero_minus_one = df['zero'] - 1
    assert isinstance(zero_minus_one, FlickerColumn)
    assert zero_minus_one.isin([-1]).all()

    df['zero_minus_one'] = df['zero'] - 1
    assert df['zero_minus_one'].isin([-1]).all()

    # We'll test `FlickerColumn.__sub__` manually
    assert df['one'].__sub__(1).isin([0]).all()


def test_right_sub_literal(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1

    # Python only calls right-sided operations such as __rsub__ when the left operand does not have an __sub__.
    # https://docs.python.org/3/reference/datamodel.html#object.__rsub__
    # Since `1` doesn't have a `__sub__` method, pyspark.sql.Column.__rsub__ is called.
    one_minus_zero = 1 - df['zero']
    assert isinstance(one_minus_zero, FlickerColumn)
    assert one_minus_zero.isin([1]).all()

    df['one_minus_zero'] = 1 - df['zero']
    assert df['one_minus_zero'].isin([1]).all()

    # We'll test `FlickerColumn.__rsub__` manually
    # 1 - one
    assert df['one'].__rsub__(1).isin([0]).all()
