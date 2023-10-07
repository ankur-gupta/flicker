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
from flicker import FlickerDataFrame


def test_basic_usage(spark):
    data = [(x, f'{ascii_lowercase[x]}', True, 3.45) for x in range(5)]
    schema = 'a INT, b STRING, c BOOLEAN, d DOUBLE'
    df = FlickerDataFrame(spark.createDataFrame(data, schema))
    assert df['a'].dtype == 'int'
    assert df['b'].dtype == 'string'
    assert df['c'].dtype == 'boolean'
    assert df['d'].dtype == 'double'

    df['e'] = None
    assert df['e'].dtype == 'void'
    # FIXME: Add datetime
