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
    with pytest.raises(Exception):
        FlickerDataFrame.from_shape(spark, 3, 5, names=list('aabcd'))


def test_zero_fill(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 5, fill='zero')

    assert isinstance(df, FlickerDataFrame)
    assert df.nrows == 3
    assert df.ncols == 5
    assert df.shape == (3, 5)
    for name in df.names:
        assert np.all(df[[name]].to_pandas()[name].to_numpy() == np.zeros(3))
    for _, dtype in df.dtypes:
        assert dtype == 'double'


def test_zero_fill_with_names(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 5, names=list('abcde'),
                                     fill='zero')
    assert isinstance(df, FlickerDataFrame)
    assert df.nrows == 3
    assert df.ncols == 5
    assert df.shape == (3, 5)
    assert df.names == list('abcde')
    for name in df.names:
        assert np.all(df[[name]].to_pandas()[name].to_numpy() == np.zeros(3))
    for _, dtype in df.dtypes:
        assert dtype == 'double'


def test_rand_fill(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 5, fill='rand')

    assert isinstance(df, FlickerDataFrame)
    assert df.nrows == 3
    assert df.ncols == 5
    assert df.shape == (3, 5)
    for _, dtype in df.dtypes:
        assert dtype == 'double'


def test_randn_fill(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 5, fill='randn')

    assert isinstance(df, FlickerDataFrame)
    assert df.nrows == 3
    assert df.ncols == 5
    assert df.shape == (3, 5)
    for _, dtype in df.dtypes:
        assert dtype == 'double'
