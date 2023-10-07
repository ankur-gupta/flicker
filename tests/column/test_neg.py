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
import numpy as np
from flicker import FlickerDataFrame


def test_basic_usage(spark):
    df = FlickerDataFrame.from_shape(spark, 5, 2, names=['a', 'b'], fill='rand')
    df['c'] = -df['a']
    df['d'] = -df['b']


def test_correctness(spark):
    df = FlickerDataFrame(spark.createDataFrame([(x, x) for x in range(5)], 'a INT, b INT'))
    df['c'] = -df['a']
    df['a + c'] = df['a'] + df['c']
    assert (df['a + c'] == 0).all()

    df['d'] = 1
    df['e'] = -df['d']
    assert (df['e'] == -1).all()


def test_nan(spark):
    df = FlickerDataFrame.from_shape(spark, 5, 2, names=['a', 'b'], fill='rand')
    df['c'] = np.nan
    df['d'] = -df['c']
    assert df['c'].is_nan().all()
    assert df['d'].is_nan().all()


def test_none(spark):
    df = FlickerDataFrame.from_shape(spark, 5, 2, names=['a', 'b'], fill='rand')
    df['c'] = None
    df['d'] = -df['c']
    assert df['c'].is_null().all()
    assert df['d'].is_null().all()
