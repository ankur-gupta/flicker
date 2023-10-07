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


def test_left_mod_column(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['two'] = 2
    df['five'] = 5

    df['five_mod_two'] = df['five'] % df['two']
    assert df['five_mod_two'].isin([1]).all()

    df['two_mod_five'] = df['two'] % df['five']._column
    assert df['two_mod_five'].isin([2]).all()

    # We'll test `FlickerColumn.__mod__` manually
    assert df['two'].__mod__(df['five']).isin([2]).all()
    assert df['two'].__mod__(df['five']._column).isin([2]).all()


# pyspark.sql.Column.__rmod__ calls the following snippet of code and is therefore only applicable to literals:
# https://github.com/apache/spark/blob/c0d9ca3be14cb0ec8d8f9920d3ecc4aac3cf5adc/python/pyspark/sql/column.py#L184-L196
# def _reverse_op(
# ...
#     """Create a method for binary operator (this object is on right side)"""
# ...
#         jother = _create_column_from_literal(other)  <- this is how `Column.__rxxx__(other)` is processed
#         jc = getattr(jother, name)(self._jc)
# ...

# def test_right_mod_column(spark):
#     df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
#     df['two'] = 2
#     df['five'] = 5
#
#     # We'll test `FlickerColumn.__rmod__` manually
#     # five % two
#     assert df['two'].__rmod__(df['five']).isin([1]).all()
#     assert df['two'].__rmod__(df['five']._column).isin([1]).all()


def test_left_mod_literal(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['five'] = 5

    five_mod_two = df['five'] % 2
    assert isinstance(five_mod_two, FlickerColumn)
    assert five_mod_two.isin([1]).all()

    df['five_mod_five'] = df['five'] % 5
    assert df['five_mod_five'].isin([0]).all()

    # We'll test `FlickerColumn.__mod__` manually
    assert df['five'].__mod__(2).isin([1]).all()


def test_right_mod_literal(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['five'] = 5

    # Python only calls right-sided operations such as __rmod__ when the left operand does not have an __mod__.
    # https://docs.python.org/3/reference/datamodel.html#object.__rmod__
    # Since `10` doesn't have a `__mod__` method, pyspark.sql.Column.__rmod__ is called.
    ten_mod_five = 10 % df['five']
    assert isinstance(ten_mod_five, FlickerColumn)
    assert ten_mod_five.isin([0]).all()

    df['two_mod_five'] = 2 % df['five']
    assert df['two_mod_five'].isin([2]).all()

    # We'll test `FlickerColumn.__rmod__` manually
    # 1 % five
    assert df['five'].__rmod__(1).isin([1]).all()
