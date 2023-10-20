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

from datetime import datetime

PYTHON_TO_SPARK_DTYPES = {
    bool: 'boolean',
    int: 'bigint',
    float: 'double',
    str: 'string',
    datetime: 'timestamp'
}

PYSPARK_NUMERIC_DTYPES = {'double', 'float', 'int', 'bigint'}
PYSPARK_FLOAT_DTYPES = {'double', 'float'}
PYSPARK_INTEGER_DTYPES = {'int', 'bigint'}
PYSPARK_BOOLEAN_DTYPES = {'boolean'}
PYSPARK_TIMESTAMP_DTYPES = {'timestamp'}
# FIXME: Add string type, char type, decimal type