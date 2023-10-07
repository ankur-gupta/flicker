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
import pytest
from flicker import FlickerDataFrame, find_empty_columns


def test_type_failure(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, None],
        'b': ['q', None, 'w']
    })
    with pytest.raises(TypeError):
        find_empty_columns(df._df, verbose=False)


def test_basic_usage(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, None],
        'b': ['q', None, 'w']
    })
    df['c'] = None
    df['d'] = 1
    df['e'] = 'abc'
    empty_names = find_empty_columns(df, verbose=False)
    assert set(empty_names) == {'c'}
    assert set(find_empty_columns(df, verbose=True)) == {'c'}
