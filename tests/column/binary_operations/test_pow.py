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


def test_left_pow_column(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['two'] = 2
    df['five'] = 5

    df['five_pow_two'] = df['five'] ** df['two']
    assert df['five_pow_two'].isin([25]).all()

    df['two_pow_five'] = df['two'] ** df['five']._column
    assert df['two_pow_five'].isin([32]).all()

    # We'll test `FlickerColumn.__pow__` manually
    assert df['two'].__pow__(df['five']).isin([32]).all()
    assert df['two'].__pow__(df['five']._column).isin([32]).all()


def test_right_pow_column(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['two'] = 2
    df['five'] = 5

    # We'll test `FlickerColumn.__rpow__` manually
    # five ** two
    assert df['two'].__rpow__(df['five']).isin([25]).all()
    assert df['two'].__rpow__(df['five']._column).isin([25]).all()


def test_left_pow_literal(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['five'] = 5

    five_pow_two = df['five'] ** 2
    assert isinstance(five_pow_two, FlickerColumn)
    assert five_pow_two.isin([25]).all()

    df['five_pow_one'] = df['five'] ** 1
    assert df['five_pow_one'].isin([5]).all()

    # We'll test `FlickerColumn.__pow__` manually
    assert df['five'].__pow__(2).isin([25]).all()


def test_right_pow_literal(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['five'] = 5

    # Python only calls right-sided operations such as __rpow__ when the left operand does not have an __pow__.
    # https://docs.python.org/3/reference/datamodel.html#object.__rpow__
    # Since `10` doesn't have a `__pow__` method, pyspark.sql.Column.__rpow__ is called.
    ten_pow_zero = 10 ** df['zero']
    assert isinstance(ten_pow_zero, FlickerColumn)
    assert ten_pow_zero.isin([1]).all()

    df['two_pow_five'] = 2 ** df['five']
    assert df['two_pow_five'].isin([32]).all()

    # We'll test `FlickerColumn.__rpow__` manually
    # 1 ** five
    assert df['five'].__rpow__(1).isin([1]).all()
