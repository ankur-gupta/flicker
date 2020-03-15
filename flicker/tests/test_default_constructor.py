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


class TestDefaultConstructor:
    spark = (SparkSession.builder
             .appName('TestDefaultConstructor').getOrCreate())

    def test_basic(self):
        df = self.spark.createDataFrame([(_, _) for _ in range(5)],
                                        'a INT, b INT')
        fdf = FlickerDataFrame(df)
        assert isinstance(fdf, FlickerDataFrame)
        assert fdf.shape == (5, 2)
        assert set(fdf.columns) == {'a', 'b'}
