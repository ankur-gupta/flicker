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

from contextlib import contextmanager

from flicker.flicker import FlickerDataFrame
from flicker.udf import len_udf
from flicker.utils import gensym


@contextmanager
def delete_extra_columns(df):
    """ This context manager exists to provide a commonly needed functionality.
        Unlike pandas which lets you compute a temporary quantity in a
        separate Series or a numpy array, pyspark requires you to create a new
        column even for temporary quantities.

        This context manager makes sure that any new columns that you create
        within the context manager will get deleted (in-place) afterwards
        even if your code encounters an Exception. Any columns that you start
        with will not be deleted (unless of course you deliberately delete
        them yourself).

        Note that this context manager will not prevent you from overwriting
        any column (new or otherwise).

        Parameters
        ----------
        df: FlickerDataFrame

        Examples
        --------
        >>> df = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'])
        >>> df
        FlickerDataFrame[a: double, b: double]
        >>> with delete_extra_columns(df):
                df['c'] = 1
                print(df.names)
        ['a', 'b', 'c']
        >>> print(df.names)
        ['a', 'b']  # 'c' column is deleted automatically
    """
    if not isinstance(df, FlickerDataFrame):
        msg = ('df must be a FlickerDataFrame to use this context manager; '
               'you provided type(df)={}')
        raise TypeError(msg.format(str(type(df))))
    names_to_keep = list(df.names)
    yield names_to_keep
    for name in df.names:
        if name not in names_to_keep:
            del df[name]  # use "mutable" function to delete in-place


def find_empty_columns(df, verbose=True):
    if not isinstance(df, FlickerDataFrame):
        msg = 'df must be a FlickerDataFrame you provided type(df)={}'
        raise TypeError(msg.format(str(type(df))))

    # A list to store the names of empty columns
    empty_names = []

    # Since we are going to add new columns to `df` in-place, we want to store
    # the initial names so we can loop over it without confusion.
    init_names = list(df.names)
    with delete_extra_columns(df):
        for i, name in enumerate(init_names):
            if verbose:
                print('Checking {}/{}: {}'.format(i + 1, len(init_names),
                                                  name))

            # Generate a new unique name. Note that we must use the latest
            # list of names (and not `init_names`) here.
            len_name = gensym(df.names, prefix='len_{}'.format(name))

            # 1. Modify df in-place by using a "mutable" function
            # 2. len_udf returns a length 1 for "scalar"/"atomic" objects
            #    that don't have a __len__ attribute.
            df[len_name] = len_udf(df[name])
            if df[df[len_name] > 0].count() == 0:
                # "name" column is empty
                if verbose:
                    print('{} was found to be empty'.format(name))
                empty_names = empty_names + [name]
    return empty_names
