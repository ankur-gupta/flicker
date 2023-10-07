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
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType
from flicker import FlickerDataFrame


def normalize(pdf: pd.DataFrame) -> pd.DataFrame:
    x = pdf['x']
    x_norm = (x - x.mean()) / x.std()
    pdf['x_norm'] = x_norm
    return pdf[['id', 'x', 'x_norm']]


def test_apply_in_pandas(spark):
    # Requires PyArrow>=1.0.0
    rows = [(1, 2.0), (1, 4.0), (2, 10.0), (2, 20.0), (2, 0.0), (1, 0.0)]
    df = FlickerDataFrame.from_rows(spark, rows, ['id', 'x'])

    df_norm_per_id = df.groupby(['id']).apply(normalize, schema='id INT, x DOUBLE, x_norm DOUBLE')
    assert isinstance(df_norm_per_id, FlickerDataFrame)
    assert set(df_norm_per_id.names) == {'id', 'x', 'x_norm'}
    assert df_norm_per_id.nrows == df.nrows
    assert df_norm_per_id.ncols == 3
    assert df_norm_per_id.take(None, convert_to_dict=True) == [
        {'id': 1, 'x': 2.0, 'x_norm': 0.0},
        {'id': 1, 'x': 4.0, 'x_norm': 1.0},
        {'id': 1, 'x': 0.0, 'x_norm': -1.0},
        {'id': 2, 'x': 10.0, 'x_norm': 0.0},
        {'id': 2, 'x': 20.0, 'x_norm': 1.0},
        {'id': 2, 'x': 0.0, 'x_norm': -1.0}
    ]


def test_apply_spark(spark):
    rows = [(1, 2.0), (1, 4.0), (2, 10.0), (2, 20.0), (2, 0.0), (1, 0.0)]
    df = FlickerDataFrame.from_rows(spark, rows, ['id', 'x'])
    normalize_pandas_udf = pandas_udf('id INT, x DOUBLE, x_norm DOUBLE', PandasUDFType.GROUPED_MAP)(normalize)
    df_norm_per_id = df.groupby(['id']).apply_spark(normalize_pandas_udf)
    assert isinstance(df_norm_per_id, FlickerDataFrame)
    assert set(df_norm_per_id.names) == {'id', 'x', 'x_norm'}
    assert df_norm_per_id.nrows == df.nrows
    assert df_norm_per_id.ncols == 3
    assert df_norm_per_id.take(None, convert_to_dict=True) == [
        {'id': 1, 'x': 2.0, 'x_norm': 0.0},
        {'id': 1, 'x': 4.0, 'x_norm': 1.0},
        {'id': 1, 'x': 0.0, 'x_norm': -1.0},
        {'id': 2, 'x': 10.0, 'x_norm': 0.0},
        {'id': 2, 'x': 20.0, 'x_norm': 1.0},
        {'id': 2, 'x': 0.0, 'x_norm': -1.0}
    ]
