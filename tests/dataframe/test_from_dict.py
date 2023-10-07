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
import pytest
import numpy as np
from flicker import FlickerDataFrame


def test_basic_usage(spark):
    data = {'a': [0, 1], 'b': [3.4, 5.6], 'c': ['a', 'b']}
    df = FlickerDataFrame.from_dict(spark, data)
    assert isinstance(df, FlickerDataFrame)
    assert df.ncols == len(data)
    assert df.nrows == 2
    assert df.shape == (2, len(data))
    assert set(df.names) == set(data.keys())
    for name in df.names:
        actual = df[[name]].to_pandas()[name].to_numpy()
        expected = np.array(data[name])
        assert np.all(actual == expected)


def test_unequal_rows(spark):
    data = {'a': [0, 1], 'b': [3.4, 5.6, 6.7], 'c': ['a', 'b']}
    with pytest.raises(Exception):
        FlickerDataFrame.from_dict(spark, data)