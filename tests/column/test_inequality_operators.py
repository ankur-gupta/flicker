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
from flicker import FlickerDataFrame, FlickerColumn


def test_le_lt(spark):
    data = [(x, x + 1, float(x), float(x + 1)) for x in range(5)]
    schema = 'a INT, b INT, c DOUBLE, d DOUBLE'
    df = FlickerDataFrame(spark.createDataFrame(data, schema))
    assert isinstance(df['a'] < df['b'], FlickerColumn)

    # Int
    assert (df['a'] < df['b']).all()
    assert (df['a'] <= df['b']).all()
    assert (df['a'] <= df['a']).all()
    assert not (df['a'] < df['a']).all()

    # Double
    assert (df['c'] < df['d']).all()
    assert (df['c'] <= df['d']).all()
    assert (df['c'] <= df['c']).all()
    assert not (df['c'] < df['c']).all()

    # Int-Double mix
    assert (df['a'] < df['d']).all()
    assert (df['a'] <= df['d']).all()
    assert not (df['a'] < df['c']).all()


def test_ge_gt(spark):
    data = [(x, x + 1, float(x), float(x + 1)) for x in range(5)]
    schema = 'a INT, b INT, c DOUBLE, d DOUBLE'
    df = FlickerDataFrame(spark.createDataFrame(data, schema))
    assert isinstance(df['a'] < df['b'], FlickerColumn)

    # Int
    assert (df['b'] > df['a']).all()
    assert (df['b'] >= df['a']).all()
    assert (df['a'] >= df['a']).all()
    assert not (df['a'] > df['a']).all()

    # Double
    assert (df['d'] > df['c']).all()
    assert (df['d'] >= df['c']).all()
    assert (df['c'] >= df['c']).all()
    assert not (df['c'] > df['c']).all()

    # Int-Double mix
    assert (df['d'] > df['a']).all()
    assert (df['d'] >= df['a']).all()
    assert not (df['c'] > df['a']).all()


def test_eq_ne(spark):
    data = [(x, x + 1, float(x), float(x + 1)) for x in range(5)]
    schema = 'a INT, b INT, c DOUBLE, d DOUBLE'
    df = FlickerDataFrame(spark.createDataFrame(data, schema))
    assert isinstance(df['a'] < df['b'], FlickerColumn)

    # Int
    assert (df['a'] == df['a']).all()
    assert (df['a'] != df['b']).all()

    # Double
    assert (df['c'] == df['c']).all()
    assert (df['c'] != df['d']).all()

    # Int-Double mix
    assert (df['a'] == df['c']).all()
    assert (df['a'] != df['d']).all()
