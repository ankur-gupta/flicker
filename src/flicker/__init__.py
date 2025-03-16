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
import os
from flicker.version import __version__
from flicker.dataframe import FlickerDataFrame
from flicker.column import FlickerColumn
from flicker.group import FlickerGroupedData
from flicker.variables import (PYTHON_TO_SPARK_DTYPES, PYSPARK_NUMERIC_DTYPES, PYSPARK_FLOAT_DTYPES,
                               PYSPARK_INTEGER_DTYPES, PYSPARK_BOOLEAN_DTYPES, PYSPARK_TIMESTAMP_DTYPES)
from flicker.mkname import mkname
from flicker.utils import is_nan_scalar, get_length, get_names_by_dtype
from flicker.recipes import delete_extra_columns, find_empty_columns
from flicker.reshape import concat

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_path(partial_path=''):
    return os.path.join(_ROOT, partial_path)
