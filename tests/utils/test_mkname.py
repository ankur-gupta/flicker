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
from string import ascii_lowercase
from flicker import mkname


def test_default_usage():
    name = mkname()
    assert isinstance(name, str)
    assert len(name) > 1


def test_prefixes():
    assert mkname([], prefix='a_').startswith('a_')
    assert mkname([], prefix='something_').startswith('something_')
    assert mkname([], prefix='some_column_name_').startswith('some_column_name_')


def test_suffixes():
    assert mkname([], suffix='_a').endswith('_a')
    assert mkname([], suffix='_something').endswith('_something')
    assert mkname([], suffix='_some_column_name').endswith('_some_column_name')


def test_prefixes_and_suffixes():
    name = mkname([], prefix='a_', suffix='_b')
    assert name.startswith('a_')
    assert name.endswith('_b')


def test_uniqueness_works():
    names = ['name_fdas', 'name_qwpo', 'name_fngh', 'name_mlkj', 'name_qasw']
    name = mkname(names, prefix='name_')
    assert name.startswith('name_')
    assert name not in names


def test_uniqueness_impossible_situation():
    names = [f'name_{char}' for char in ascii_lowercase]
    with pytest.raises(Exception):
        mkname(names, prefix='name_', n_random_chars=1)


def test_repeated_usage():
    names = ['name_fdas', 'name_qwpo', 'name_fngh', 'name_mlkj', 'name_qasw', 'name_uyut', 'name_poiu']
    for _ in range(100):
        name = mkname(names, prefix='name_')
        names = names + [name]


def test_n_random_chars():
    n_random_chars = 10
    name = mkname(n_random_chars=n_random_chars)
    assert len(name) >= n_random_chars
