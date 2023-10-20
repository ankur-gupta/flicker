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
from datetime import datetime, timedelta

import pandas as pd

from flicker import FlickerDataFrame
from flicker.summary import get_summary_timestamp_columns


def test_timestamp(spark):
    t = datetime(2023, 1, 1)
    dt = timedelta(days=1)
    data = {
        't': [t - dt, t, t + dt],
        'n': [1, 2, 3]
    }
    df = FlickerDataFrame.from_dict(spark, data)
    summary = get_summary_timestamp_columns(df._df)
    assert isinstance(summary, pd.DataFrame)
    assert set(summary.columns) == {'t'}
    assert summary['t'].loc['count'] == df.nrows
    assert summary['t'].loc['min'] == t - dt
    assert summary['t'].loc['max'] == t + dt
    assert summary['t'].loc['stddev'] == dt
