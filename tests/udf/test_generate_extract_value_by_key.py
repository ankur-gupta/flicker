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
from typing import Callable
from flicker.udf import generate_extract_value_by_key


def test_basic_usage():
    f = generate_extract_value_by_key('value')
    assert isinstance(f, Callable)
    assert f(None) is None
    assert f({}) is None
    assert f({'value': 1}) == 1
    assert f({'value': 1.0}) == 1.0
    assert f({'value': 'a'}) == 'a'
    assert f({'value': ['a']}) == ['a']
    assert f({'value': {'a', 'b'}}) == {'a', 'b'}

    with pytest.raises(Exception):
        f(1)
    with pytest.raises(Exception):
        f(1.0)
    with pytest.raises(Exception):
        f([])
