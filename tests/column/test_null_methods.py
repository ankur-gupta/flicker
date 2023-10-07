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


def test_basic_usage(spark):
    data = {
        'double_no_null': (1.2, 3.63634, 4.5, -0.58, 0.9809),
        'double_with_null': (1.2, 3.90808, None, -0.58, 0.768)
    }
    df = FlickerDataFrame.from_dict(spark, data)
    assert not df['double_no_null'].is_null().any()
    assert not df['double_no_null'].is_null().all()
    assert df['double_no_null'].is_not_null().any()
    assert df['double_no_null'].is_not_null().all()

    assert df['double_with_null'].is_null().any()
    assert not df['double_with_null'].is_null().all()
    assert df['double_with_null'].is_not_null().any()
    assert not df['double_with_null'].is_not_null().all()


def test_void(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, list('ab'))
    df['c'] = None
    assert df['c'].dtype == 'void'
    assert df['c'].is_null().all()
    assert df['c'].is_null().any()
    assert not df['c'].is_not_null().any()


def test_all_null_bool(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, list('ab'))
    df['c'] = None
    df['c'] = df['c'].astype(bool)
    assert df['c'].is_null().all()
    assert df['c'].is_null().any()
    assert not df['c'].is_not_null().any()


def test_all_null_float(spark):
    df = FlickerDataFrame.from_shape(spark, 3, 2, list('ab'))
    df['c'] = None
    df['c'] = df['c'].astype(float)
    assert df['c'].is_null().all()
    assert df['c'].is_null().any()
    assert not df['c'].is_not_null().any()
