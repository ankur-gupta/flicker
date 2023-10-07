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
from flicker import get_length


def test_list_length():
    x = [1, 2, 3, 4, 5]
    length = get_length(x)
    assert length == len(x)


def test_tuple_length():
    x = (1, 2, 3)
    length = get_length(x)
    assert length == len(x)


def test_set_length():
    x = {1, 2, 3, 1, 2, 3, 1, 1, None, (1,)}
    length = get_length(x)
    assert length == len(x)


def test_string_length():
    x = "Hello, World!"
    length = get_length(x)
    assert length == len(x)


def test_range():
    assert get_length(range(10)) == 10


def test_custom_iterable():
    custom_iterable = (x for x in range(10))
    length = get_length(custom_iterable)
    assert length == 10


def test_empty_iterable():
    assert get_length([]) == 0
    assert get_length(tuple()) == 0
    assert get_length({}) == 0
    assert get_length(set()) == 0


def test_non_iterable_raises_error():
    with pytest.raises(TypeError):
        get_length(42)


def test_otherwise_unreachable_exception():
    class BogusLen:
        def __len__(self):
            raise TypeError('This is a bogus error to test an otherwise unreachable branch of the code.')

    with pytest.raises(TypeError):
        get_length(BogusLen())
