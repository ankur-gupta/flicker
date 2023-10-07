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


def test_delitem_valid(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    assert set(df.names) == set(list('abcdef'))
    assert df.ncols == len('abcdef')

    del df['a']
    assert set(df.names) == set(list('bcdef'))
    assert df.ncols == len('bcdef')

    del df['e']
    assert set(df.names) == set(list('bcdf'))
    assert df.ncols == len('bcdf')


def test_delitem_failure(spark):
    df = FlickerDataFrame.from_shape(spark, nrows=10, ncols=6, names=list('abcdef'), fill='zero')
    with pytest.raises(Exception):
        del df['unknown-column-name']
