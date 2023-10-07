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
import pandas as pd
from flicker import FlickerDataFrame


def test_basic_usage(spark):
    records = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 25}, {'name': 'Charlie', 'age': 25}]
    pdf = pd.DataFrame(records)
    df = FlickerDataFrame.from_pandas(spark, pdf)
    assert df.shape == pdf.shape
    assert set(df.names) == set(pdf.columns)
    bool_df = df.to_pandas() == pdf
    for name in bool_df.columns:
        assert all(bool_df[name])


def test_duplicated_names_failure(spark):
    pdf = pd.DataFrame({'a': [0, 1], 'b': [3.4, 5.6]})
    pdf_duplicate_names = pd.concat([pdf, pdf], axis=1)
    with pytest.raises(Exception):
        FlickerDataFrame.from_pandas(spark, pdf_duplicate_names)