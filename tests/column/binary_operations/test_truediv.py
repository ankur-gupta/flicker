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


def test_left_truediv_column(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1
    df['two'] = 2

    df['two_divided_by_one'] = df['two'] / df['one']
    assert df['two_divided_by_one'].isin([2]).all()

    df['zero_divided_by_one'] = df['zero'] / df['one']._column
    assert df['zero_divided_by_one'].isin([0]).all()

    # We'll test `FlickerColumn.__truediv__` manually
    assert df['zero'].__truediv__(df['one']).isin([0]).all()
    assert df['two'].__truediv__(df['one']._column).isin([2]).all()


# pyspark.sql.Column.__rtruediv__ calls the following snippet of code and is therefore only applicable to literals:
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
#     # We'll test `FlickerColumn.__rtruediv__` manually
#     assert df['one'].__rtruediv__(df['one']).isin([1]).all()
#     assert df['one'].__rtruediv__(df['one']._column).isin([1]).all()


def test_left_truediv_literal(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1

    zero_divided_by_one = df['zero'] / 1
    assert isinstance(zero_divided_by_one, FlickerColumn)
    assert zero_divided_by_one.isin([0]).all()

    df['zero_divided_by_one'] = df['zero'] / 1
    assert df['zero_divided_by_one'].isin([0]).all()

    # We'll test `FlickerColumn.__truediv__` manually
    assert df['one'].__truediv__(1).isin([1]).all()


def test_right_truediv_literal(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1

    # Python only calls right-sided operations such as __rtruediv__ when the left operand does not have an __truediv__.
    # https://docs.python.org/3/reference/datamodel.html#object.__rtruediv__
    # Since `10` doesn't have a `__truediv__` method, pyspark.sql.Column.__rtruediv__ is called.
    ten_divided_by_one = 10 / df['one']
    assert isinstance(ten_divided_by_one, FlickerColumn)
    assert ten_divided_by_one.isin([10]).all()

    df['ten_divided_by_one'] = 10 / df['one']
    assert df['ten_divided_by_one'].isin([10]).all()

    # We'll test `FlickerColumn.__rtruediv__` manually
    # 10 / one
    assert df['one'].__rtruediv__(10).isin([10]).all()
