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

from flicker import FlickerDataFrame, FlickerColumn


def test_basic_usage(spark):
    df = FlickerDataFrame(spark.createDataFrame([(x, x) for x in range(5)], 'a INT, b INT'))
    df['c'] = True
    df['d'] = False
    assert isinstance(df['c'], FlickerColumn)

    with pytest.raises(Exception):
        df['c'].__bool__()
    with pytest.raises(Exception):
        bool(df['c'])
    with pytest.raises(Exception):
        x = 1 if df['c'] else 0
