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
from __future__ import annotations
from typing import Iterable

from string import ascii_lowercase
import random


def mkname(names: Iterable[str] = (), prefix: str = '', suffix: str = '',
           max_tries: int = 100, n_random_chars: int = 4) -> str:
    """ Generate a unique name by combining a given prefix and suffix with a randomly generated string.

    Parameters
    ----------
    names: Iterable[str], optional
        Existing names to check for uniqueness. Defaults to an empty iterable.
    prefix: str, optional
        Prefix to prepend to the generated name. Defaults to an empty string.
    suffix: str, optional
        Suffix to append to the generated name. Defaults to an empty string.
    max_tries: int, optional
        Maximum number of attempts to generate a unique name. Defaults to 100.
    n_random_chars: int, optional
        Number of random characters to generate for the name. Defaults to 4.

    Returns
    -------
    str
        A unique name that combines the prefix, randomly generated characters, and suffix.

    Raises
    ------
    ValueError
        If the maximum number of attempts is exceeded and a unique name cannot be generated.

    Examples
    --------
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = FlickerDataFrame.from_shape(spark, 2, 4, names=['name', 'age', 'weight_lbs', 'height'])
    >>> mkname(df.names, prefix='age_')
    'age_bzrl'
    """
    names = set(names)
    # FIXME: Check that candidate exists before adding random characters
    for i in range(max_tries):
        stub = ''.join(random.choices(ascii_lowercase, k=n_random_chars))
        candidate = f'{prefix}{stub}{suffix}'
        if candidate not in names:
            return candidate
    raise ValueError(f'Exceeded maximum {max_tries} tries to generate a unique name. '
                     f'Try increase max_tries or n_random_chars.')
