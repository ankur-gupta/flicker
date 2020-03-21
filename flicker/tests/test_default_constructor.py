# Copyright 2020 Ankur Gupta
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from builtins import range

import pytest
from pyspark.sql.functions import lit
from flicker import FlickerDataFrame


def test_basic_usage(spark):
    df = spark.createDataFrame([(_, _) for _ in range(5)], 'a INT, b INT')
    fdf = FlickerDataFrame(df)
    assert isinstance(fdf, FlickerDataFrame)
    assert fdf.shape == (5, 2)
    assert set(fdf.columns) == {'a', 'b'}


def test_duplicate_name_fails(spark):
    df = spark.createDataFrame([(_, _) for _ in range(5)], 'a INT, a INT')
    with pytest.raises(Exception):
        FlickerDataFrame(df)


def test_from_shape(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 4)
    assert df.shape == (3, 4)


def test_from_dict(spark):
    df = FlickerDataFrame.from_dict(spark, {'a': [1, 2, 3],
                                            'b': ['v', 't', 'u']})
    assert df.shape == (3, 2)
    assert set(df.names) == set(['a', 'b'])
    assert list(df[['a']].toPandas()['a']) == [1, 2, 3]
    assert list(df[['b']].toPandas()['b']) == ['v', 't', 'u']
