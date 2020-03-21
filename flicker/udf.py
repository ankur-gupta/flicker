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

from pyspark.sql.types import (BooleanType, IntegerType, DoubleType,
                               StringType, LongType, FloatType, TimestampType,
                               ArrayType)
from pyspark.sql.functions import udf


def make_double_udf(fun):
    return udf(fun, DoubleType())


def make_integer_udf(fun):
    return udf(fun, IntegerType())


def make_string_udf(fun):
    return udf(fun, StringType())


def make_boolean_udf(fun):
    return udf(fun, BooleanType())


# Stubs and generator functions
def _len(e):
    """ Very opinionated function that considers length of atomic objects 1"""
    if e is None:
        return 0
    elif hasattr(e, '__len__'):
        return len(e)
    else:
        return 1


def _row_value(row):
    """ A function to extract the contents of the 'value' key in a Row
        object. This function exists because this is a very common situation.
    """
    # We always assume that a row is nullable
    if row is None:
        out = None
    else:
        out = row['value']
    return out


def _keys(e):
    """ A function to extract the keys (not values) from a dict object"""
    if e is None:
        out = []
    else:
        out = list(e.keys())
    return out


def generate_extract_value_by_key(key):
    """ Generate and return a function that can extract the value of a key
        from a dict. Note that we use a closure here which can have a
        performance issue with pyspark in some cases. In such cases, a user
        can make their own function directly from scratch.

        Note that this function is opinionated: when the key doesn't exist
        it doesn't fail, it returns a None. This is because we expect that
        returning None (instead of failing) would be more commonly acceptable
        situation.
    """
    def _extract_value_by_key(e):
        if e is None:
            out = None
        elif key in e:
            out = e[key]
        else:
            # We will call e.keys() simply to make it fail. This is because
            # we want the function to fail if it's called on a type that
            # doesn't even have the `.keys` method (for example, when the
            # data type is wrong).
            e.keys()
            out = None
        return out
    return _extract_value_by_key


# Commonly used UDF functions
type_udf = make_string_udf(lambda x: type(x).__name__)
len_udf = make_integer_udf(_len)

# Row-value UDF functions
string_row_value_udf = udf(_row_value, StringType())
boolean_row_value_udf = udf(_row_value, BooleanType())
integer_row_value_udf = udf(_row_value, IntegerType())
long_row_value_udf = udf(_row_value, LongType())
double_row_value_udf = udf(_row_value, DoubleType())
float_row_value_udf = udf(_row_value, FloatType())
timestamp_row_value_udf = udf(_row_value, TimestampType())

# Extract all the keys (not values)
# Since keys are string-valued so often, we define a "default" function
# to be the same as the one for strings.
string_keys_udf = udf(_keys, ArrayType(StringType()))
integer_keys_udf = udf(_keys, ArrayType(IntegerType()))
keys_udf = string_keys_udf

# Extract the value of a specific key
# This is an example function to show you how to use
# `generate_extract_value_by_key` to create a udf.
string_extract_value_of_name_udf = udf(
    generate_extract_value_by_key('name'),
    StringType())
