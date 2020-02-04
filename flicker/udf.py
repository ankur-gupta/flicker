from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from builtins import range

from pyspark.sql.types import BooleanType, IntegerType, DoubleType, StringType
from pyspark.sql.functions import udf


def make_double_udf(fun):
    return udf(fun, DoubleType())


def make_integer_udf(fun):
    return udf(fun, IntegerType())


def make_string_udf(fun):
    return udf(fun, StringType())


def make_boolean_udf(fun):
    return udf(fun, BooleanType())


# Stubs
def _len(e):
    """ Very opinionated function that considers length of atomic objects 1"""
    if e is None:
        return 0
    elif hasattr(e, '__len__'):
        return len(e)
    else:
        return 1


# Commonly used UDF functions
type_udf = make_string_udf(lambda x: type(x).__name__)
len_udf = make_integer_udf(_len)
