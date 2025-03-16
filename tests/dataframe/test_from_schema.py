# Copyright 2025 Flicker Contributors
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
import pytest
from pyspark.sql.types import StructType, StructField, BooleanType, IntegerType
from flicker import FlickerDataFrame


def test_empty_string_schema(spark):
    df = FlickerDataFrame.from_schema(spark, "")
    assert df.shape == (0, 0)
    assert len(df.schema) == 0


def test_none_schema(spark):
    df = FlickerDataFrame.from_schema(spark, None)
    assert df.shape == (0, 0)
    assert len(df.schema) == 0


def test_empty_struct_type_schema_1(spark):
    df = FlickerDataFrame.from_schema(spark, StructType())
    assert df.shape == (0, 0)
    assert len(df.schema) == 0


def test_empty_struct_type_schema_2(spark):
    df = FlickerDataFrame.from_schema(spark, StructType([]))
    assert df.shape == (0, 0)
    assert len(df.schema) == 0


def test_schema_recreation(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 4, names=list('abcd'), fill='zero')
    df_recreated = FlickerDataFrame.from_schema(spark, df.schema)
    assert df.ncols == df_recreated.ncols
    assert df.names == df_recreated.names
    assert df.schema == df_recreated.schema


def test_manually_constructed_schema(spark):
    schema = StructType([
        StructField('a', BooleanType(), True),
        StructField('b', IntegerType(), False)
    ])
    df = FlickerDataFrame.from_schema(spark, schema)
    assert df.ncols == len(schema)
    assert df.names == schema.fieldNames()
    assert df.schema == schema


def test_string_schema(spark):
    df = FlickerDataFrame.from_schema(spark, 'a STRING, b INT')
    assert df.ncols == 2
    assert df.names == ['a', 'b']
    assert len(df.schema) == 2
    assert df.schema.fieldNames() == ['a', 'b']
