# Copyright 2020 Ankur Gupta
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
from flicker.utils import gensym


def test_empty_names():
    assert gensym([], prefix='a') == 'a'
    assert gensym([], prefix='something') == 'something'
    assert gensym([], prefix='some_column_name') == 'some_column_name'


def test_expected_name():
    names = ['name', 'age', 'account']
    assert gensym(names, prefix='name1', suffix='') == 'name1'
    assert gensym(names, prefix='new_name', suffix='') == 'new_name'
    assert gensym(names, prefix='name', suffix='_new') == 'name_new'
    assert gensym(names, prefix='', suffix='name1') == 'name1'
    assert gensym(names, prefix='', suffix='new_name') == 'new_name'


def test_basic_usage():
    names = ['name', 'age', 'account']
    for _ in range(10):
        name = gensym(names, prefix='name', suffix='')
        names = names + [name]
