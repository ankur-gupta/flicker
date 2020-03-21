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
from flicker import FlickerDataFrame
from flicker.recipes import find_empty_columns


def test_basic_usage(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, None],
        'b': ['q', None, 'w']
    })
    df['c'] = None
    df['d'] = 1
    df['e'] = 'abc'
    empty_names = find_empty_columns(df, verbose=False)
    assert set(empty_names) == set(['c'])
