# 🔥flicker
[![PyPI Latest Release](https://img.shields.io/pypi/v/flicker.svg)](https://pypi.org/project/flicker/)
![build](https://github.com/ankur-gupta/flicker/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/ankur-gupta/flicker/branch/master/graph/badge.svg)](https://codecov.io/gh/ankur-gupta/flicker)

This python package provides a `FlickerDataFrame` object. `FlickerDataFrame`
is a thin wrapper over `pyspark.sql.DataFrame`. The aim of `FlickerDataFrame`
is to provide a more Pandas-like dataframe API. Flicker is like
[Koalas](https://github.com/databricks/koalas)
in that Flicker attempts to provide a pandas-like API. But there are strong
differences in design. We will release a Design Principles guide for `flicker`
soon.

One way to understand `flicker`'s position is via the following analogy:

> _**keras** is to **tensorflow** as **flicker** is to **pyspark**_

`flicker` aims to provides a more intuitive, pythonic API over a `pyspark`
backend. `flicker` relies completely on `pyspark` for all distributed
computing work.


# Getting Started
## Install
`flicker` is intended to be run with Python 3. You can install `flicker`
from [PyPI](https://pypi.org/project/flicker/):
```bash
pip install --user flicker
```

`flicker` does not use Python 3 features just yet. This means that `flicker`
may work with Python 2 (though it has not been tested and is highly
discouraged). For use with Python 2, try installing `flicker` with `pip2` or
build from source. Please note that `flicker` would _very soon_ become
incompatible with Python 2 as we start using Python 3 features.

As of now, `flicker` is compatible with `pyspark 2.x`. Compatibility with
`pyspark 3.x` is not supported just yet.

## Quick Example
`flicker` aims to simplify some of the common and tedious aspects of a PySpark
dataframe without compromising performance. The following example shows some
of the features of `flicker`.

 ```python
from pyspark.sql import SparkSession
from flicker import FlickerDataFrame

# Get a spark session, if needed.
spark = SparkSession.builder.appName('PySparkShell').getOrCreate()

# Set case sensitivity to true.
# https://stackoverflow.com/a/62455158/4383754
spark.conf.set('spark.sql.caseSensitive', True)

# Create a dummy Flicker DataFrame using normally distributed random data of shape (100, 3)
df = FlickerDataFrame.from_shape(spark, nrows=100, ncols=3, names=['a', 'b', 'c'], fill='randn')

# Print the object to see the column names and types
df
# FlickerDataFrame[a: double, b: double, c: double]

# You can get pandas-like API to inspect a FlickerDataFrame
df.shape
# (100, 3)

df.names
# ['a', 'b', 'c']

df.dtypes
# [('a', 'double'), ('b', 'double'), ('c', 'double')]

# One of the main features of flicker is the following handy shortcut to view the data.
# Calling a FlickerDataFrame object, returns the first 5 rows as a pandas DataFrame.
# See ?df for more examples on how you can use this to quickly and interactively perform analysis.
df()
#           a         b         c
# 0 -0.488747 -0.378013  0.350972
# 1  0.224332  0.322416 -0.943630
# 2  0.249755 -0.738754 -0.060325
# 3  1.108189  1.657239 -0.114664
# 4  1.768242 -2.422804 -1.012876

# Another cool feature of flicker is pandas-like assignment API. Instead of having to
# use .withColumn(), you can simply assign. For example, if we wanted to create a new
# column that indicates if df['a'] is positive or not, we can do it like this:
df['is_a_positive'] = df['a'] > 0

df
# FlickerDataFrame[a: double, b: double, c: double, is_a_positive: boolean]

# We can now 'call' df to view the first 5 rows.
df()
#           a         b         c  is_a_positive
# 0 -0.488747 -0.378013  0.350972          False
# 1  0.224332  0.322416 -0.943630           True
# 2  0.249755 -0.738754 -0.060325           True
# 3  1.108189  1.657239 -0.114664           True
# 4  1.768242 -2.422804 -1.012876           True

# These features can intermixed in nearly every imaginable way. Here are some quick examples.
# Example 1: show the first 5 rows of the dataframe that has only 'a' and 'c' names selected.
df[['a', 'c']]()

# Example 2: Filter the data to select only the rows that have a positive value in column 'a' and
# show the first 3 rows of the filtered dataframe.
df[df['is_a_positive']](3)
#           a         b         c  is_a_positive
# 0  0.224332  0.322416 -0.943630           True
# 1  0.249755 -0.738754 -0.060325           True
# 2  1.108189  1.657239 -0.114664           True

# Example 3: Show first 2 rows that have a positive product of 'a' and 'b'
df[(df['a'] * df['b']) > 0][['a', 'b']](2)
#           a         b
# 0 -0.488747 -0.378013
# 1  0.224332  0.322416
 ```

 ## Additional functions
`flicker` aims to provide commonly used recipes as general-purpose functions
that you can immediatelty use out-of-the-box. These are a few quick examples.
 ```python
import numpy as np
from pyspark.sql import SparkSession
from flicker import FlickerDataFrame
from flicker.udf import len_udf, type_udf

# Get a spark session, if needed.
spark = SparkSession.builder.appName('PySparkShell').getOrCreate()

# Create a more complicated dataframe using one of the factory constructor
data = [(1, 'spark', 2.4, {}), (2, 'flicker', np.nan, {'key': 1})]
column_names = ['a', 'b', 'c', 'd']
df = FlickerDataFrame.from_rows(spark, rows=data, names=column_names)
df
# FlickerDataFrame[a: bigint, b: string, c: double, d: map<string,bigint>]

df()
#    a        b    c           d
# 0  1    spark  2.4          {}
# 1  2  flicker  NaN  {'key': 1}

# Get the type of column 'd' and store it in a new column 'd_type'
df['d_type'] = type_udf(df['d'])

# The new column 'd_type' gets added without you having to worry about making a udf.
df
# FlickerDataFrame[a: bigint, b: string, c: double, d: map<string,bigint>, d_type: string]

# Show the first 5 rows of the dataframe
df()
#    a        b    c           d d_type
# 0  1    spark  2.4          {}   dict
# 1  2  flicker  NaN  {'key': 1}   dict

# Get the lengths of elements in the columns 'a' and 'd'
df['a_len'] = len_udf(df['a'])
df['d_len'] = len_udf(df['d'])
df
# FlickerDataFrame[a: bigint, b: string, c: double, d: map<string,bigint>, d_type: string, d_len: int, a_len: int]

df()
#    a        b    c           d d_type  d_len  a_len
# 0  1    spark  2.4          {}   dict      0      1
# 1  2  flicker  NaN  {'key': 1}   dict      1      1

# Filter out rows that have an empty dict in column 'd'
df[df['d_len'] > 0]()
#    a        b   c           d d_type  d_len  a_len
# 0  2  flicker NaN  {'key': 1}   dict      1      1

# Finally, you can always perform an operation on a dataframe and store it as a new dataframe
new_df = df[df['d_len'] > 0]
```

## Use the underlying PySpark DataFrame
If `flicker` isn't enough, you can always use the underlying PySpark DataFrame.
Here are a few examples.
```python
# Continued from the above example.

# `._df` contains the underlying PySpark DataFrame
type(df._df)
# pyspark.sql.dataframe.DataFrame

# Use PySpark functions to compute the frequency table based on type of column 'd'
df._df.groupBy(['d_type']).count().show()
# +------+-----+
# |d_type|count|
# +------+-----+
# |  dict|    2|
# +------+-----+

# You can always convert a PySpark DataFrame into a FlickerDataFrame
# after you've performed the native PySpark operations. This way, you can
# continue to enjoy the benefits of FlickerDataFrame. Converting a
# PySpark DataFrame into a FlickerDataFrame is always fast irrespective of
# dataframe size.
df_freq_table = FlickerDataFrame(df._df.groupBy(['d_type']).count())
df_freq_table()
#   d_type  count
# 0   dict      2
```

 # Status
`flicker` is actively being developed. While `flicker` is immediately useful
for data analysis, it may not be ready for production use just yet. It is very
likely that you will need a function that has not yet written in `flicker`.
In such cases, you can always use the underlying PySpark DataFrame to do
every operation that PySpark supports. Please consider filing an issue for
missing functions, bugs, or unintuitive API. Happy sparking!

# License
`flicker` is available under [Apache License 2.0](https://github.com/ankur-gupta/flicker/blob/master/LICENSE).

`flicker` depends on other python packages listed in
[requirements.txt](https://github.com/ankur-gupta/flicker/blob/master/requirements.txt)
which have their own licenses. `flicker` releases do not bundle any code from
the dependencies directly.

The documentation is made using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
theme which has [MIT License](https://squidfunk.github.io/mkdocs-material/license/).
