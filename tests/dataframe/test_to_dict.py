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
from flicker import FlickerDataFrame


def test_basic_usage(spark):
    data = {'a': [0, 1], 'b': [3.4, 5.6], 'c': ['a', 'b']}
    df = FlickerDataFrame.from_dict(spark, data)
    out = df.to_dict(n=1)
    assert isinstance(out, dict)
    assert set(out.keys()) == set(data.keys())
    for name, column in out.items():
        assert len(column) == 1


def test_n_none(spark):
    data = {'a': [0, 1, 0, 1], 'b': [3.4, 5.6, np.nan, 89.90], 'c': ['a', 'b', None, None]}
    df = FlickerDataFrame.from_dict(spark, data)
    out = df.to_dict(n=None)
    assert isinstance(out, dict)
    assert set(out.keys()) == set(data.keys())
    for name, column in out.items():
        assert len(column) == df.nrows


def test_n_zero(spark):
    data = {'a': [0, 1, 0, 1], 'b': [3.4, 5.6, np.nan, 89.90], 'c': ['a', 'b', None, None]}
    df = FlickerDataFrame.from_dict(spark, data)
    out = df.to_dict(n=0)
    assert isinstance(out, dict)
    assert set(out.keys()) == set(data.keys())
    for name, column in out.items():
        assert len(column) == 0


def test_empty_dataframe(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, ['a', 'b'])
    df['c'] = False

    subset_df = df[df['c']]
    out = subset_df.to_dict()
    assert isinstance(out, dict)
    assert set(out.keys()) == {'a', 'b', 'c'}
    for name, column in out.items():
        assert len(column) == 0
