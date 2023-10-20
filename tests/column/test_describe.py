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
from string import ascii_lowercase
import numpy as np
import pandas as pd
from flicker import FlickerDataFrame


def test_basic_usage(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    for name in df.names:
        desc = df[name].describe()
        assert isinstance(desc, pd.Series)
        assert desc.name == name
        assert desc.loc['count'] == df.nrows
        assert isinstance(desc.loc['mean'], float)
        assert isinstance(desc.loc['stddev'], float)


def test_multiple_dtypes_together(spark):
    data = [(x, f'{ascii_lowercase[x]}', True, 3.45 if x == 1 else np.nan) for x in range(5)]
    schema = 'a INT, b STRING, c BOOLEAN, d DOUBLE'
    df = FlickerDataFrame(spark.createDataFrame(data, schema))
    df['e'] = False

    # Integer column
    desc = df['a'].describe()
    assert isinstance(desc, pd.Series)
    assert desc.name == 'a'
    assert desc.loc['count'] == df.nrows
    assert isinstance(desc.loc['mean'], float)
    assert isinstance(desc.loc['stddev'], float)
    assert isinstance(desc.loc['min'], int)
    assert isinstance(desc.loc['max'], int)

    # String column
    desc = df['b'].describe()
    assert isinstance(desc, pd.Series)
    assert desc.name == 'b'
    assert desc.loc['count'] == df.nrows
    assert isinstance(desc.loc['min'], str)
    assert isinstance(desc.loc['max'], str)

    # Boolean column
    desc = df['c'].describe()
    assert isinstance(desc, pd.Series)
    assert desc.name == 'c'
    assert desc.loc['count'] == df.nrows
    assert isinstance(desc.loc['mean'], float)
    assert isinstance(desc.loc['stddev'], float)
    assert isinstance(desc.loc['min'], bool)
    assert isinstance(desc.loc['max'], bool)
    assert desc.loc['min'] is True
    assert desc.loc['max'] is True

    # Float column
    desc = df['d'].describe()
    assert isinstance(desc, pd.Series)
    assert desc.name == 'd'
    assert desc.loc['count'] == df.nrows
    assert isinstance(desc.loc['mean'], float)
    assert isinstance(desc.loc['stddev'], float)
    assert isinstance(desc.loc['min'], float)
    assert isinstance(desc.loc['max'], float)

    # Boolean column
    desc = df['e'].describe()
    assert isinstance(desc, pd.Series)
    assert desc.name == 'e'
    assert desc.loc['count'] == df.nrows
    assert isinstance(desc.loc['mean'], float)
    assert isinstance(desc.loc['stddev'], float)
    assert isinstance(desc.loc['min'], bool)
    assert isinstance(desc.loc['max'], bool)
    assert desc.loc['mean'] == 0.0
    assert desc.loc['stddev'] == 0.0
    assert desc.loc['min'] is False
    assert desc.loc['max'] is False
