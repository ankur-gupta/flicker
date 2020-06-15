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


def test_empty_dataframe(spark):
    df = FlickerDataFrame(spark.createDataFrame([], ''))
    with pytest.raises(KeyError):
        df.get_dtype('a')
    with pytest.raises(KeyError):
        df.get_dtype('')
    with pytest.raises(TypeError):
        df.get_dtype(1)
    with pytest.raises(TypeError):
        df.get_dtype(1.0)
    with pytest.raises(TypeError):
        df.get_dtype(True)
    with pytest.raises(TypeError):
        df.get_dtype(None)
    with pytest.raises(TypeError):
        df.get_dtype([])


def test_general(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 4],
        'b': [1, 2, 3, np.nan],
        'c': [1.0, 2.9, None, 3.0],
        'd': ['a', 'b', 'n', 'y'],
        'e': [True, None, False, True]
    })
    assert df.get_dtype('a') == 'bigint'
    assert df.get_dtype('b') == 'double'
    assert df.get_dtype('c') == 'double'
    assert df.get_dtype('d') == 'string'
    assert df.get_dtype('e') == 'boolean'
    for name, dtype in df.dtypes:
        assert df.get_dtype(name) == dtype

    with pytest.raises(KeyError):
        df.get_dtype('non-existent-column')
    with pytest.raises(KeyError):
        df.get_dtype('')
    with pytest.raises(KeyError):
        df.get_dtype('aa')
    with pytest.raises(KeyError):
        df.get_dtype('abcde')
    with pytest.raises(TypeError):
        df.get_dtype(1)
    with pytest.raises(TypeError):
        df.get_dtype(1.0)
    with pytest.raises(TypeError):
        df.get_dtype(True)
    with pytest.raises(TypeError):
        df.get_dtype(None)
    with pytest.raises(TypeError):
        df.get_dtype([])
