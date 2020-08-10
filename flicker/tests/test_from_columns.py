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

import numpy as np
import pytest
from flicker import FlickerDataFrame


def test_duplicated_names_failure(spark):
    columns = [[1, 2, 3], ['hello', 'spark', 'flicker']]
    names = ['a', 'a']
    with pytest.raises(Exception):
        FlickerDataFrame.from_columns(spark, columns, names)


def test_columns_names_mismatch(spark):
    columns = [[1, 2, 3], ['hello', 'spark', 'flicker']]
    names = ['a', 'b', 'c']
    with pytest.raises(Exception):
        FlickerDataFrame.from_columns(spark, columns, names)


def test_typical_usage(spark):
    columns = [[1, 2, 3], ['hello', 'spark', 'flicker']]

    df = FlickerDataFrame.from_columns(spark, columns)
    assert df.shape == (3, 2)

    first_name = df.names[0]
    first_column = df[[first_name]].to_pandas()[first_name].to_numpy()
    assert np.all(first_column == np.array(columns[0]))

    second_name = df.names[1]
    second_column = df[[second_name]].to_pandas()[second_name].to_numpy()
    assert np.all(second_column == np.array(columns[1]))


def test_usage_with_names(spark):
    columns = [[1, 2, 3], ['hello', 'spark', 'flicker']]
    names = ['a', 'b']

    df = FlickerDataFrame.from_columns(spark, columns, names)
    assert df.shape == (3, 2)
    assert list(df.names) == list(names)

    for i, name in enumerate(df.names):
        column = df[[name]].to_pandas()[name].to_numpy()
        expected_column = np.array(columns[i])
        assert np.all(column == expected_column)
