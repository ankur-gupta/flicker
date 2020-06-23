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
import six
import numpy as np
import pandas as pd

from flicker.utils import get_float_column_names


def test_basic_usage():
    df = pd.DataFrame({
        'a': [np.nan, 1.3, np.nan],
        'b': [True, False, True],
        'c': ['spark', np.nan, None],
        'd': [1, 2, 3]
    })
    names = get_float_column_names(df)
    assert isinstance(names, list)
    for name in names:
        assert isinstance(name, six.string_types)
    assert len(names) == 1
    assert names[0] == 'a'


def test_empty_dataframe():
    df = pd.DataFrame({})
    assert len(get_float_column_names(df)) == 0


def test_no_float_dataframe():
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': ['a', 'b', 'c']
    })
    assert len(get_float_column_names(df)) == 0


def test_all_float_dataframe():
    df = pd.DataFrame({
        'a': [1.8, 2.9, 3],
        'b': [np.nan, np.nan, np.nan],
        'c': [np.nan, 1.5, np.nan]
    })
    names = get_float_column_names(df)
    assert isinstance(names, list)
    for name in names:
        assert isinstance(name, six.string_types)
    assert len(names) == 3
    assert set(names) == {'a', 'b', 'c'}
