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
import numpy as np
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

from flicker import FlickerDataFrame, FlickerColumn
from flicker.udf import type_udf


def test_apply_type_udf(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=4, ncols=2, names=list('ab'), fill='zero')
    type_column = df['a'].apply(type_udf)
    assert isinstance(type_column, FlickerColumn)
    assert type_column.dtype == 'string'
    type_column_values = type_column.take(None)
    assert len(type_column_values) == df.nrows
    for type_ in type_column_values:
        assert type_ in {'int', 'bigint'}

    df['type_a'] = type_column
    assert isinstance(df, FlickerDataFrame)
    assert 'type_a' in df.names
    assert isinstance(df['type_a'], FlickerColumn)
    first_row = df.take(1, convert_to_dict=True)[0]
    assert (first_row == {'a': 0, 'b': 0, 'type_a': 'int'}) or (first_row == {'a': 0, 'b': 0, 'type_a': 'bigint'})


def test_apply_lambda_udf(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=4, ncols=2, names=list('ab'), fill='rand')
    square_udf = udf(lambda x: x * x, DoubleType())
    df['a_squared'] = df['a'].apply(square_udf)
    rows = df[['a', 'a_squared']].take(None, convert_to_dict=True)
    for row in rows:
        assert np.allclose(row['a_squared'], row['a'] * row['a'])
