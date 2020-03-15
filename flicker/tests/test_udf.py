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
from pyspark.sql import SparkSession
from flicker import FlickerDataFrame

# Create a single spark session
from flicker.udf import type_udf, len_udf

spark = SparkSession.builder.appName('TestDefaultConstructor').getOrCreate()


class TestTypeUDF:
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3],
        'b': ['a', 'b', None],
        'c': [{'a': 1}, {'b': 1}, {'c': 1}]
    })

    def test_basic(self):
        for col_name in self.df.columns:
            type_col_name = '{}_type'.format(col_name)
            self.df[type_col_name] = type_udf(self.df[col_name])

        # Check that everything is correct
        pdf = self.df.toPandas()
        assert all(pdf['a_type'] == 'int')
        assert all(pdf['b_type'].to_numpy() == ['str', 'str', 'NoneType'])
        assert all(pdf['c_type'] == 'dict')


class TestLenUDF:
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3],
        'b': ['a', 'bc', None],
        'c': [{'a': 1}, {'b': 1}, {'c': 1, 'd': 6}]
    })

    def test_basic(self):
        for col_name in self.df.columns:
            type_col_name = '{}_len'.format(col_name)
            self.df[type_col_name] = len_udf(self.df[col_name])

        # Check that everything is correct
        pdf = self.df.toPandas()
        assert all(pdf['a_len'] == 1)
        assert all(pdf['b_len'].to_numpy() == [1, 2, 0])
        assert all(pdf['c_len'].to_numpy() == [1, 1, 2])
