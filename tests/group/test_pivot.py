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


def test_pivot(spark):
    rows = [
        ('python', 2020, 2),
        ('rust', 2020, 3),
        ('python', 2020, 4),
        ('python', 2023, 5),
        ('rust', 2023, 6),
    ]
    df = FlickerDataFrame.from_rows(spark, rows, names=['language', 'year', 'weeks'])
    g = df.groupby(['year'])
    assert isinstance(g, FlickerGroupedData)
    p = g.pivot('language')
    assert isinstance(p, FlickerGroupedData)
    df_pivot_by_year = p.mean(['weeks'])
    assert isinstance(df_pivot_by_year, FlickerDataFrame)
    assert set(df_pivot_by_year.names) == {'year', 'python', 'rust'}
    assert df_pivot_by_year.nrows == df['year'].value_counts().nrows
    assert df_pivot_by_year.ncols == 3
    assert df_pivot_by_year.take(None, convert_to_dict=True) == [
        {'year': 2020, 'python': 3.0, 'rust': 3.0},
        {'year': 2023, 'python': 5.0, 'rust': 6.0}
    ]
