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
from flicker import FlickerDataFrame, FlickerGroupedData


def test_str_repr(spark):
    df = FlickerDataFrame.from_dict(spark, {'a': [1, 2, 1, 2], 'b': [1.0, 2.0, 3.0, 4.0]})
    g = df.groupby(['a'])
    assert isinstance(g, FlickerGroupedData)
    assert isinstance(repr(g), str)
    assert isinstance(str(g), str)
