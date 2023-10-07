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

from flicker import FlickerDataFrame, FlickerGroupedData


def test_works(spark):
    df = FlickerDataFrame.from_dict(spark, {'a': [1, 2, 1, 2], 'b': [1.0, 2.0, 3.0, 4.0]})
    g = FlickerGroupedData(df._df, df._df.groupBy('a'))
    assert isinstance(g, FlickerGroupedData)


def test_wrong_types(spark):
    df = FlickerDataFrame.from_dict(spark, {'a': [1, 2, 1, 2], 'b': [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(TypeError):
        FlickerGroupedData(None, df._df.groupBy('a'))
    with pytest.raises(TypeError):
        FlickerGroupedData(df, df._df.groupBy('a'))
    with pytest.raises(TypeError):
        FlickerGroupedData(df._df, None)
    with pytest.raises(TypeError):
        FlickerGroupedData(df._df, df.groupby(['a']))


def test_duplicate_column_fails(spark):
    df = spark.createDataFrame([(_, _, _) for _ in range(5)], 'x INT, x INT, y INT')
    with pytest.raises(ValueError):
        FlickerGroupedData(df, df.groupBy('y'))
