# Copyright 2020 Flicker Contributors
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from builtins import range

import pytest
from flicker import FlickerDataFrame


def test_basic_usage(spark):
    df = FlickerDataFrame.from_dict(spark, {
        'a': [1, 2, 3, 1, None],
        'b': ['a', 'v', 'r', None, 't'],
        'c': [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    df.write.mode('overwrite').parquet('./data-as-parquet')
    df_reread = FlickerDataFrame(spark.read.parquet('./data-as-parquet'))
    assert df.shape == df_reread.shape
