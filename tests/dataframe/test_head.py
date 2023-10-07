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


def test_head_default(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    assert isinstance(df.head(), FlickerDataFrame)
    assert df.head().nrows <= df.nrows


def test_head_n(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    assert isinstance(df.head(1), FlickerDataFrame)
    assert df.head(1).shape == (1, df.ncols)

    assert isinstance(df.head(2), FlickerDataFrame)
    assert df.head(2).shape == (2, df.ncols)

    if df.nrows < 100:
        assert isinstance(df.head(100), FlickerDataFrame)
        assert df.head(100).shape == df.shape


def test_head_n_none(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    assert isinstance(df.head(None), FlickerDataFrame)
    assert df.head(None).shape == df.shape
