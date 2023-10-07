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
    data = {
        'all_true': (True, True, True, True, True),
        'all_false': (False, False, False, False, False),
        # 'all_none': (None, None, None, None, None),  # this causes spark to fail
        'true_none': (True, None, None, True, None),
        'false_none': (None, None, None, None, False),
        'true_false_none': (True, False, None, False, False)
    }
    df = FlickerDataFrame.from_dict(spark, data)
    df['all_none'] = None  # This works!
    for name in df.names:
        df[name] = df[name].astype(bool)
    assert df['all_true'].all(ignore_null=True)
    assert df['all_true'].all(ignore_null=False)

    assert not df['all_false'].all(ignore_null=True)
    assert not df['all_false'].all(ignore_null=False)

    assert df['all_none'].all(ignore_null=True)
    assert not df['all_none'].all(ignore_null=False)

    assert df['true_none'].all(ignore_null=True)
    assert not df['true_none'].all(ignore_null=False)

    assert not df['false_none'].all(ignore_null=True)
    assert not df['false_none'].all(ignore_null=False)

    assert not df['true_false_none'].all(ignore_null=True)
    assert not df['true_false_none'].all(ignore_null=False)

    # Test with assignment
    df['c'] = True
    assert df['c'].all(ignore_null=True)
    assert df['c'].all(ignore_null=False)

    df['d'] = False
    assert not df['d'].all(ignore_null=True)
    assert not df['d'].all(ignore_null=False)

    df['e'] = None  # e is dtype='void'
    with pytest.raises(Exception):
        df['e'].all(ignore_null=True)
    with pytest.raises(Exception):
        df['e'].all(ignore_null=False)


def test_wrong_type(spark):
    data = {'a': (1, 2, 3, 4, 5), 'b': ('a', 'b', 'c', 'd', 'e')}
    df = FlickerDataFrame.from_dict(spark, data)
    assert isinstance(df, FlickerDataFrame)
    assert isinstance(df['a'], FlickerColumn)
    assert isinstance(df['b'], FlickerColumn)
    with pytest.raises(Exception):
        df['a'].all()
    with pytest.raises(Exception):
        df['b'].all()
