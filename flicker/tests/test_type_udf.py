# Copyright 2020 Flicker Contributors
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
from flicker import FlickerDataFrame
from flicker.udf import type_udf


def test_basic_usage(spark):
    df = spark.createDataFrame([(i, str(i))
                                for i in range(3)], 'a INT, b STRING')
    fdf = FlickerDataFrame(df)
    fdf['a_type'] = type_udf(fdf['a'])
    fdf['b_type'] = type_udf(fdf['b'])
    assert set(fdf[['a_type']].toPandas()['a_type']) == {'int'}
    assert set(fdf[['b_type']].toPandas()['b_type']) == {'str'}


def test_with_nulls(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3],
        'b': ['a', 'b', None],
        'c': [{'a': 1}, {'b': 1}, {'c': 1}]
    })
    for col_name in df.names:
        type_col_name = '{}_type'.format(col_name)
        df[type_col_name] = type_udf(df[col_name])

    # Check that everything is correct
    pdf = df.toPandas()
    assert all(pdf['a_type'] == 'int')
    assert all(pdf['b_type'].to_numpy() == ['str', 'str', 'NoneType'])
    assert all(pdf['c_type'] == 'dict')
