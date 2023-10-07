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

from flicker import FlickerDataFrame
from flicker.udf import type_udf


def test_basic_usage(spark):
    df = FlickerDataFrame(spark.createDataFrame([(i, str(i)) for i in range(3)], 'a INT, b STRING'))
    df['type_a'] = type_udf(df['a']._column)
    df['type_b'] = type_udf(df['b']._column)
    assert df['type_a'].isin(['int']).all()
    assert df['type_b'].isin(['str']).all()


def test_with_nulls(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3],
        'b': ['a', 'b', None],
        'c': [{'a': 1}, {'b': 1}, {'c': 1}]
    })
    df['d'] = None
    df['e'] = 3.14

    # Compute lengths for all columns
    for name in df.names:
        df[f'type_{name}'] = type_udf(df[name]._column)

    pdf = df.to_pandas()
    assert all(pdf['type_a'] == 'int')
    assert all(pdf['type_b'].to_numpy() == ['str', 'str', 'NoneType'])
    assert all(pdf['type_c'] == 'dict')
    assert all(pdf['type_d'] == 'NoneType')
    assert all(pdf['type_e'] == 'float')
