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

import random

CHARACTERS = "abcdefghijklmnopqrstuvwxyz0123456789"


def gensym(names=[], prefix='col_', suffix='', n_max_tries=100,
           n_random_chars=4):
    """ Generate a new, unique column name that is different from
        all existing columns.
    """
    names = set(names)

    # Try out without any randomness first
    candidate = prefix + suffix
    if candidate in names:
        for i in range(n_max_tries):
            stub = ''.join([random.choice(CHARACTERS)
                            for _ in range(n_random_chars)])
            candidate = '{}{}{}'.format(prefix, stub, suffix)
            if candidate not in names:
                return candidate
    else:
        return candidate
    msg = 'No unique name generated in {} tries with {} random characters'
    raise KeyError(msg.format(n_max_tries, n_random_chars))
