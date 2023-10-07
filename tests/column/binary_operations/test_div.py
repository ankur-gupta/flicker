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


# Python no longer uses __div__ when we use the `/` operator. Instead
# `/` = __truediv__
# `//` = __floordiv__
# We still test it because pyspark.sql.Column still implements __div__ and __rdiv__.


def test_left_div_column(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1
    df['two'] = 2
    assert df['zero'].__div__(df['one']).isin([0]).all()
    assert df['two'].__div__(df['one']._column).isin([2]).all()


# pyspark.sql.Column.__rdiv__ calls the following snippet of code and is therefore only applicable to literals:
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
#     # We'll test `FlickerColumn.__rdiv__` manually
#     assert df['one'].__rdiv__(df['one']).isin([1]).all()
#     assert df['one'].__rdiv__(df['one']._column).isin([1]).all()


def test_left_div_literal(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1
    assert df['one'].__div__(1).isin([1]).all()


def test_right_div_literal(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['one'] = 1
    assert df['one'].__rdiv__(10).isin([10]).all()  # 10 / one
