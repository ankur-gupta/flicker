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
from flicker import FlickerDataFrame
from flicker.udf import len_udf


def test_with_nulls(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3],
        'b': ['a', 'bc', None],
        'c': [{'a': 1}, {'b': 1}, {'c': 1, 'd': 6}]
    })

    for col_name in df.columns:
        type_col_name = '{}_len'.format(col_name)
        df[type_col_name] = len_udf(df[col_name])

    # Check that everything is correct
    pdf = df.toPandas()
    assert all(pdf['a_len'] == 1)
    assert all(pdf['b_len'].to_numpy() == [1, 2, 0])
    assert all(pdf['c_len'].to_numpy() == [1, 1, 2])
