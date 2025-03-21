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
from pyspark.sql.types import StructType

from flicker import FlickerDataFrame


def test_schema(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=2, names=list('ab'), fill='zero')
    assert isinstance(df.schema, StructType)
    assert len(df.schema) == df.ncols
    assert df.schema.fieldNames() == df.names


