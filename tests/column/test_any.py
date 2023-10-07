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
from flicker import FlickerDataFrame, FlickerColumn


def test_basic_usage(spark):
    data = {'a': (False, False, None, False, False), 'b': (None, None, True, False, None)}
    df = FlickerDataFrame.from_dict(spark, data)
    df['c'] = True
    df['d'] = False
    assert not df['a'].any()
    assert df['b'].any()
    assert df['c'].any()
    assert not df['d'].any()
    df['e'] = None
    with pytest.raises(Exception):
        df['e'].any()


def test_wrong_type(spark):
    data = {'a': (1, 2, 3, 4, 5), 'b': ('a', 'b', 'c', 'd', 'e')}
    df = FlickerDataFrame.from_dict(spark, data)
    assert isinstance(df, FlickerDataFrame)
    assert isinstance(df['a'], FlickerColumn)
    assert isinstance(df['b'], FlickerColumn)
    with pytest.raises(Exception):
        df['a'].any()
    with pytest.raises(Exception):
        df['b'].any()
