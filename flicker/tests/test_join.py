# Copyright 2020 Ankur Gupta
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


def test_join_on_dict(spark):
    x = FlickerDataFrame.from_shape(spark, 2, 4,
                                    ["year", "age", "name", "case"])
    y = FlickerDataFrame.from_shape(spark, 2, 4,
                                    ["new_year", "new_age", "new_name",
                                     "new_case"])
    z = x.join(y, on={'year': 'new_year'}, how='inner')
    assert z.shape == (4, 8)
    assert set(z.names) == set(["year", "age", "name", "case",
                                "new_year", "new_age", "new_name", "new_case"])


def test_join_with_lsuffix(spark):
    x = FlickerDataFrame.from_shape(spark, 2, 3,
                                    ["year", "balance", "account"])
    with pytest.raises(Exception):
        x.join(x, on='year', how='inner')

    y = x.join(x, on='year', how='inner', lsuffix='_left')
    assert y.shape == (4, 6)
    expected_names = list(x.names) + [name + '_left' for name in x.names]
    assert set(y.names) == set(expected_names)
