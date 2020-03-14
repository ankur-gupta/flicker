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
    for i in range(n_max_tries):
        stub = ''.join([random.choice(CHARACTERS)
                        for _ in range(n_random_chars)])
        candidate = '{}{}{}'.format(prefix, stub, suffix)
        if candidate not in names:
            return candidate
    msg = 'No unique name generated in {} tries with {} random characters'
    raise KeyError(msg.format(n_max_tries, n_random_chars))
