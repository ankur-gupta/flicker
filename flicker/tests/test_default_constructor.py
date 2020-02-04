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
