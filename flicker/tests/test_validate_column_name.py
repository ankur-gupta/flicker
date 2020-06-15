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


def test_with_default_arguments(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, None],
        'b': ['a', 'v', 'r', None, 't'],
        'c': [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    for name in df.names:
        assert df._validate_column_name(name) is None
    with pytest.raises(KeyError):
        df._validate_column_name('')
    with pytest.raises(KeyError):
        df._validate_column_name('cc')
    with pytest.raises(KeyError):
        df._validate_column_name('non-existent-column')
    with pytest.raises(TypeError):
        df._validate_column_name(1)
    with pytest.raises(TypeError):
        df._validate_column_name(1.0)
    with pytest.raises(TypeError):
        df._validate_column_name(None)
    with pytest.raises(TypeError):
        df._validate_column_name(True)
    with pytest.raises(TypeError):
        df._validate_column_name([])


def test_with_specified_names(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, None],
        'b': ['a', 'v', 'r', None, 't'],
        'c': [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    assert df._validate_column_name('a', df.names) is None
    assert df._validate_column_name('a', ['a']) is None
    assert df._validate_column_name('a', ['a', 'b', 'a']) is None
    assert df._validate_column_name('b', ['a', 'b', 'a']) is None
    with pytest.raises(KeyError):
        df._validate_column_name('b', ['a', 'c', 'a'])
    with pytest.raises(KeyError):
        df._validate_column_name('cc', df.names)
    with pytest.raises(KeyError):
        df._validate_column_name('', df.names)
    with pytest.raises(TypeError):
        df._validate_column_name(1, [])
    with pytest.raises(TypeError):
        df._validate_column_name(1.0, df.names)
    with pytest.raises(TypeError):
        df._validate_column_name(None, ['a'])
    with pytest.raises(TypeError):
        df._validate_column_name(True, ['a'])
    with pytest.raises(TypeError):
        df._validate_column_name([], df.names)


def test_with_dataframe_name(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, None],
        'b': ['a', 'v', 'r', None, 't'],
        'c': [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    assert df._validate_column_name('a', dataframe_name='some name') is None
    with pytest.raises(KeyError):
        df._validate_column_name('cc', dataframe_name='some name')
    with pytest.raises(TypeError):
        df._validate_column_name(1, dataframe_name='')
    with pytest.raises(TypeError):
        df._validate_column_name(1.0, dataframe_name='left')
    with pytest.raises(TypeError):
        df._validate_column_name(None, dataframe_name='right')
    with pytest.raises(TypeError):
        df._validate_column_name(True, dataframe_name='some name')
    with pytest.raises(TypeError):
        df._validate_column_name([], dataframe_name='some name')


def test_general(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, None],
        'b': ['a', 'v', 'r', None, 't'],
        'c': [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    assert df._validate_column_name('a', df.names, dataframe_name='df') is None
    assert df._validate_column_name('a', ['a'], dataframe_name='df') is None
    assert (df._validate_column_name('b', ['a', 'b'], dataframe_name='df')
            is None)
    with pytest.raises(KeyError):
        df._validate_column_name('cc', df.names, dataframe_name='some name')
    with pytest.raises(TypeError):
        df._validate_column_name(1, ['a'], dataframe_name='')
    with pytest.raises(TypeError):
        df._validate_column_name(1.0, [], dataframe_name='left')
    with pytest.raises(TypeError):
        df._validate_column_name(None, ['a', 'b', 'a'], dataframe_name='right')
    with pytest.raises(TypeError):
        df._validate_column_name(True, df.names, dataframe_name='some name')
    with pytest.raises(TypeError):
        df._validate_column_name([], df.names, dataframe_name='some name')
