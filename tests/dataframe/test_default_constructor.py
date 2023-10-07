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
from flicker import FlickerDataFrame


def test_basic_usage(spark):
    df = spark.createDataFrame([(_, _) for _ in range(5)], 'a INT, b INT')
    fdf = FlickerDataFrame(df)
    assert isinstance(fdf, FlickerDataFrame)
    assert fdf.shape == (5, 2)
    assert set(fdf.names) == {'a', 'b'}


def test_duplicate_name_fails(spark):
    df = spark.createDataFrame([(_, _) for _ in range(5)], 'a INT, a INT')
    with pytest.raises(Exception):
        FlickerDataFrame(df)


def test_wrong_type_input(spark):
    with pytest.raises(Exception):
        FlickerDataFrame()
    with pytest.raises(Exception):
        FlickerDataFrame(None)
    with pytest.raises(Exception):
        FlickerDataFrame(1)
    with pytest.raises(Exception):
        FlickerDataFrame(5.67)
    with pytest.raises(Exception):
        FlickerDataFrame([1, 2, 3])
    with pytest.raises(Exception):
        FlickerDataFrame([{'name': 1}])
