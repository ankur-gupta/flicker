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
from flicker import FlickerDataFrame


def test_take_default(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=4, ncols=2, names=list('ab'), fill='zero')
    for n in range(0, df.nrows):
        out = df['a'].take(n)
        assert isinstance(out, list)
        assert len(out) == n
    assert all([value == 0 for value in df['a'].take()])
    assert df['a'].take(1) == [0]
    assert df['a'].take(2) == [0, 0]
    assert df['a'].take(3) == [0, 0, 0]
    assert df['a'].take(4) == [0, 0, 0, 0]
    assert df['a'].take(5) == [0, 0, 0, 0]
    assert df['a'].take(6) == [0, 0, 0, 0]
    assert df['a'].take(10) == [0, 0, 0, 0]
    assert df['a'].take(None) == [0] * df.nrows
