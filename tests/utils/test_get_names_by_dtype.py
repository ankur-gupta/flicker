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
from flicker import FlickerDataFrame, get_names_by_dtype


def test_basic_usage(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 1, ['zero'], fill='zero')
    df['approx_pi'] = 3.14
    df['approx_e'] = 2.718
    df['name'] = 'Alice'
    df['full_name'] = 'John Doe'

    assert set(get_names_by_dtype(df._df, 'bigint')) == {'zero'}
    assert set(get_names_by_dtype(df._df, 'double')) == {'approx_pi', 'approx_e'}
    assert set(get_names_by_dtype(df._df, 'string')) == {'name', 'full_name'}
