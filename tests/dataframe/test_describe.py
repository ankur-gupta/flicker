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
    desc = df.describe()
    assert isinstance(desc, pd.DataFrame)
    assert set(desc.columns) == set(df.names)
    assert all(desc.loc['count'] == df.nrows)
    for stat in ['mean', 'stddev']:
        for value in desc.loc[stat]:
            assert isinstance(value, float)


def test_multiple_dtypes_together(spark):
    data = [(x, f'{ascii_lowercase[x]}', True, 3.45 if x == 1 else np.nan) for x in range(5)]
    schema = 'a INT, b STRING, c BOOLEAN, d DOUBLE'
    df = FlickerDataFrame(spark.createDataFrame(data, schema))
    df['e'] = False
    desc = df.describe()
    assert isinstance(desc, pd.DataFrame)
    assert set(desc.columns) <= set(df.names)
    assert all(desc.loc['count'] == df.nrows)
    for stat in ['mean', 'stddev']:
        for value in desc.loc[stat]:
            assert isinstance(value, float)

    for stat in ['min', 'max']:
        assert isinstance(desc['a'].loc[stat], int)
        assert isinstance(desc['b'].loc[stat], str)
        assert isinstance(desc['c'].loc[stat], bool)
        assert isinstance(desc['d'].loc[stat], float)
        assert isinstance(desc['e'].loc[stat], bool)

    assert desc['c'].loc['min'] is True
    assert desc['c'].loc['max'] is True
    assert desc['e'].loc['min'] is False
    assert desc['e'].loc['max'] is False
