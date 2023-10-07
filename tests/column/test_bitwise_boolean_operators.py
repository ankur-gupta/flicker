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


def test_left_and(spark):
    data = {
        'all_true': (True, True, True, True, True),
        'all_false': (False, False, False, False, False),
    }
    df = FlickerDataFrame.from_dict(spark, data)

    # All of these use left-sided __and__.
    assert df[df['all_true'] & df['all_false']].nrows == 0
    assert df[df['all_false'] & df['all_true']].nrows == 0
    assert df[df['all_true'] & df['all_false']._column].nrows == 0
    assert df[df['all_false'] & df['all_true']._column].nrows == 0

    # We'll also test __and__ manually
    assert df[df['all_true'].__and__(df['all_false'])].nrows == 0
    assert df[df['all_true'].__and__(df['all_false']._column)].nrows == 0


def test_right_and(spark):
    data = {
        'all_true': (True, True, True, True, True),
        'all_false': (False, False, False, False, False),
    }
    df = FlickerDataFrame.from_dict(spark, data)

    # These will fail because python uses `pyspark.sql.Column.__and__` instead of `FlickerColumn._rand__`.
    # Python only calls right-sided operations such as __rand__ when the left operand does not have an __and__.
    # https://docs.python.org/3/reference/datamodel.html#object.__rand__
    # assert df[df['all_true']._column & df['all_false']].nrows == 0
    # assert df[df['all_false']._column & df['all_true']].nrows == 0

    # We'll test __rand__ manually
    assert df[df['all_true'].__rand__(df['all_false'])].nrows == 0
    assert df[df['all_true'].__rand__(df['all_false']._column)].nrows == 0


def test_left_or(spark):
    data = {
        'all_true': (True, True, True, True, True),
        'all_false': (False, False, False, False, False),
    }
    df = FlickerDataFrame.from_dict(spark, data)

    # All of these use left-sided __or__.
    assert df[df['all_true'] | df['all_false']].nrows == df.nrows
    assert df[df['all_false'] | df['all_true']].nrows == df.nrows
    assert df[df['all_true'] | df['all_false']._column].nrows == df.nrows
    assert df[df['all_false'] | df['all_true']._column].nrows == df.nrows

    # We'll also test __ror__ manually
    assert df[df['all_true'].__or__(df['all_false'])].nrows == df.nrows
    assert df[df['all_true'].__or__(df['all_false']._column)].nrows == df.nrows


def test_right_or(spark):
    data = {
        'all_true': (True, True, True, True, True),
        'all_false': (False, False, False, False, False),
    }
    df = FlickerDataFrame.from_dict(spark, data)

    # Python only calls right-sided operations such as __ror__ when the left operand does not have an __or__.
    # https://docs.python.org/3/reference/datamodel.html#object.__ror__

    # We'll test __ror__ manually
    assert df[df['all_true'].__ror__(df['all_false'])].nrows == df.nrows
    assert df[df['all_true'].__ror__(df['all_false']._column)].nrows == df.nrows
