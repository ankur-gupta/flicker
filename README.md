# 🔥flicker
![GitHub](https://img.shields.io/github/license/ankur-gupta/flicker?link=https%3A%2F%2Fgithub.com%2Fankur-gupta%2Fflicker%2Fblob%2Fmain%2FLICENSE)
![package build](https://img.shields.io/github/actions/workflow/status/ankur-gupta/flicker/build-package.yml)
![mkdocs dev build](https://img.shields.io/github/actions/workflow/status/ankur-gupta/flicker/mkdocs-dev.yml?label=mkdocs)
[![codecov](https://codecov.io/gh/ankur-gupta/flicker/graph/badge.svg?token=iYwe8TbPrn)](https://codecov.io/gh/ankur-gupta/flicker)
![PyPI - Version](https://img.shields.io/pypi/v/flicker?link=https%3A%2F%2Fpypi.org%2Fproject%2Fflicker%2F)
![PyPI - Downloads](https://img.shields.io/pypi/dm/flicker)

<!-- TOC -->
* [🔥flicker](#flicker)
  * [What is `flicker`?](#what-is-flicker)
  * [How is `flicker` different than `pyspark.pandas` (formerly `koalas`)?](#how-is-flicker-different-than-pysparkpandas-formerly-koalas)
  * [Installation](#installation)
  * [Quick Example](#quick-example)
  * [Use the underlying PySpark DataFrame or Column](#use-the-underlying-pyspark-dataframe-or-column)
  * [Use UDFs](#use-udfs-)
  * [Why not use `pyspark.pandas` (formerly `koalas`)?](#why-not-use-pysparkpandas-formerly-koalas)
  * [Status](#status)
  * [License](#license)
<!-- TOC -->

## What is `flicker`?
This python package provides a `FlickerDataFrame` object. `FlickerDataFrame`
is a thin wrapper over `pyspark.sql.DataFrame`. The aim of `FlickerDataFrame`
is to provide a more [Polars](https://www.pola.rs/)-like (not pandas-like) dataframe API. 
One way to understand `flicker`'s position is via the following analogy:

> _**keras** is to **tensorflow** as **flicker** is to **pyspark**_

`flicker` provides a modern, clean, intuitive, pythonic API over a `pyspark`
backend. `flicker` relies completely on `pyspark` for all distributed
computing work.

## How is `flicker` different than `pyspark.pandas` (formerly `koalas`)?
`flicker` is indeed just an alternative to `pyspark.pandas` (formerly `koalas`).
Theoretically, `pyspark.pandas` can provide similar functionality as `flicker` but, in practice,
`pyspark.pandas` suffers from severe performance and usability issues. You can see a detailed example 
[here](#why-not-use-pysparkpandas-formerly-koalas).

Flicker, on the other hand, is designed to provide a modern dataframe API. 
In terms for API design, Flicker is more similar to [Polars](https://www.pola.rs/) than to 
[Pandas](https://pandas.pydata.org/). Flicker is designed to be just as performant as PySpark itself.
And, finally, flicker considers interactive usage (such as exploratory data analysis) as the most important use case. 

## Installation
`flicker` is intended to be run with Python>=3.9 and PySpark>=3.4.1. We recommend Python 3.11 and PySpark 3.5.0.
You can install `flicker` from [PyPI](https://pypi.org/project/flicker/):
```bash
pip install flicker
```
If you need to set up Spark on your machine, see [pyspark-starter](https://github.com/ankur-gupta/pyspark-starter).

Alternatively, you can also build from source.
```bash
# Brief instructions. Modify to your requirements.
pip install hatch
git clone https://github.com/ankur-gupta/flicker 
cd $REPO_ROOT  # typically, ./flicker
hatch build 
pip install ./dist/flicker-1.0.0-py3-none-any.whl
```

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

# Flicker provides handy factory methods that you can use to create dataframes. 
# These factory methods are typically mostly useful to perform quick experiments.
# The easiest one is the `.from_shape` method.
df = FlickerDataFrame.from_shape(spark, nrows=100, ncols=3, names=['a', 'b', 'c'], fill='randn')

# Print the object to see the column names and types
# This returns immediately but doesn't print data.
df
# FlickerDataFrame[a: double, b: double, c: double]

# Best way to get started is to check the shape of the dataframe.
# Spark (and Flicker) already knows the number of columns but the number of rows needs to be
# computed, which can take some time.
df.shape
# (100, 3)

# You can also get the number of rows or columns directly.
df.nrows  # returns immediately because nrows is cached
# 100

df.ncols
# 3

# Instead of `df.columns`, use df.names for column names 
df.names
# ['a', 'b', 'c']

# dtypes is an OrderedDict
df.dtypes
# OrderedDict([('a', 'double'), ('b', 'double'), ('c', 'double')])

# You can get the dtype for a column name.
df.dtypes['b']
# 'double'

# One of the main features of flicker is the following handy shortcut to view the data.
# Calling a FlickerDataFrame object, returns the first 5 rows as a pandas DataFrame.
# See ?df for more examples on how you can use this to quickly and interactively perform analysis.
df()
#           a         b         c
# 0 -0.593432  0.768301 -0.302519
# 1  -0.11001  0.414888  0.075711
# 2 -0.997298  0.082082  1.080696
# 3  0.299431 -0.072324 -0.055095
# 4  -0.17833 -0.655759  0.252901

# Another cool feature of flicker is pandas-like assignment API. Instead of having to
# use .withColumn(), you can simply assign. For example, if we wanted to create a new
# column that indicates if df['a'] is positive or not, we can do it like this:
df['is_a_positive'] = df['a'] > 0

# See the new column 'is_a_positive'
df  # returns immediately
# FlickerDataFrame[a: double, b: double, c: double, is_a_positive: boolean]

# We can now 'call' df to view the first 5 rows.
df()
#           a         b         c is_a_positive
# 0 -0.593432  0.768301 -0.302519         False
# 1  -0.11001  0.414888  0.075711         False
# 2 -0.997298  0.082082  1.080696         False
# 3  0.299431 -0.072324 -0.055095          True
# 4  -0.17833 -0.655759  0.252901         False

# These features can intermixed in nearly every imaginable way. Here are some quick examples.
# Example 1: show the first 2 rows of the dataframe that has only 'a' and 'c' names selected.
df[['a', 'c']](2)
#           a         c
# 0 -0.593432 -0.302519
# 1  -0.11001  0.075711

# Example 2: Filter the data to select only the rows that have a positive value in column 'a' and
# show the first 3 rows of the filtered dataframe.
df[df['is_a_positive']](3)
#           a         b         c is_a_positive
# 0  0.299431 -0.072324 -0.055095          True
# 1  0.338228  -0.48378 -1.168131          True
# 2  0.578432 -1.223312 -0.546291          True

# Example 3: Show first 2 rows that have a positive product of 'a' and 'b'
df[(df['a'] * df['b']) > 0][['a', 'b']](2)
#           a         b
# 0  -0.17833 -0.655759
# 1 -0.054472  -0.82237

# You can also get some basic column operations done
df['a']  # returns immediately
# FlickerColumn<'a'>

# FIXME: FILL ME

# Show first 2 values of column 'a'
df[['a']](2)
#           a
# 0 -0.593432
# 1  -0.11001

# Describe the distribution of column 'a'
df[['a']].describe()
#                 a
# summary
# count         100
# mean    -0.024628
# stddev   0.980973
# min     -2.752549
# max      2.477625

# Get the value counts for 'is_a_positive' column 
df['is_a_positive'].value_counts()  # returns immediately
# FlickerDataFrame[is_a_positive: boolean, count: bigint]

# See the first 5 rows of the above dataframe by 'calling' it
df['is_a_positive'].value_counts()()
#   is_a_positive count
# 0         False    57
# 1          True    43

# Normalize the counts
df['is_a_positive'].value_counts(normalize=True)()
#   is_a_positive count
# 0         False  0.57
# 1          True  0.43
```

## Use the underlying PySpark DataFrame or Column
If `flicker` isn't enough, you can always use the underlying PySpark DataFrame.
Here are a few examples.
```python
from pyspark.sql import SparkSession
from flicker import FlickerDataFrame
from flicker.udf import type_udf

# Get a spark session, if needed.
spark = SparkSession.builder.appName('PySparkShell').getOrCreate()
spark.conf.set('spark.sql.caseSensitive', True)

# Create a more complicated dataframe using one of the factory methods
data = [(1, 'spark', 2.4, {}), (2, 'flicker', None, {'key': 1})]
column_names = ['a', 'b', 'c', 'd']
df = FlickerDataFrame.from_rows(spark, rows=data, names=column_names)
df
# FlickerDataFrame[a: bigint, b: string, c: double, d: map<string,bigint>]

# Typically, NaNs get converted into None(s) but be careful about None vs NaN issues. 
df()
#    a        b     c           d
# 0  1    spark   2.4          {}
# 1  2  flicker  None  {'key': 1}

# Always best to extract the FlickerDataFrame into pure python to verify if something is a NaN
# or None.
df.take()
# [{'a': 1, 'b': 'spark', 'c': 2.4, 'd': {}},
#  {'a': 2, 'b': 'flicker', 'c': None, 'd': {'key': 1}}]

# `._df` contains the underlying PySpark DataFrame.
# Note that `df._df` is immutable but `df` is mutable.
type(df._df)  # pyspark.sql.dataframe.DataFrame

# You can use call any of the underlying methods of `df._df`. Since, `df._df` is immutable,
# you don't have to worry about any `df._df.method()` call modifying `df`. 
df._df.show()
# +---+-------+----+----------+
# |  a|      b|   c|         d|
# +---+-------+----+----------+
# |  1|  spark| 2.4|        {}|
# |  2|flicker|NULL|{key -> 1}|
# +---+-------+----+----------+

# You can destructure a FlickerDataFrame by accessing the underlying columns
d = df['d']
d  # FlickerColumn<'d'>
type(d)  # flicker.column.FlickerColumn

# As before, you can access the underlying pyspark.sql.Column 
d._column  # Column<'d'>
type(d._column)  # pyspark.sql.column.Column

# You can always convert a PySpark DataFrame into a FlickerDataFrame
# after you've performed the native PySpark operations. This way, you can
# continue to enjoy the benefits of FlickerDataFrame. Converting a
# PySpark DataFrame into a FlickerDataFrame is always fast irrespective of
# dataframe size.
df['d_type'] = type_udf(df['d']._column)
df_freq_table = FlickerDataFrame(df._df.groupBy(['d_type']).count())
df_freq_table()
#   d_type  count
# 0   dict      2
```

## Use UDFs 
`flicker.udf` comes with some useful UDFs. You easily use UDFs with `FlickerColumn` as shown below.

```python
from pyspark.sql import SparkSession
from flicker import FlickerDataFrame
from flicker.udf import type_udf, len_udf

# Get a spark session, if needed.
spark = SparkSession.builder.appName('PySparkShell').getOrCreate()
spark.conf.set('spark.sql.caseSensitive', True)

# Create a more complicated dataframe using one of the factory methods
data = [(1, 'spark', 2.4, {}), (2, 'flicker', None, {'key': 1})]
column_names = ['a', 'b', 'c', 'd']
df = FlickerDataFrame.from_rows(spark, rows=data, names=column_names)
# FlickerDataFrame[a: bigint, b: string, c: double, d: map<string,bigint>]

# FlickerColumn is quite powerful too
type(df['d'])  # flicker.column.FlickerColumn

# You can apply a UDF to a FlickerColumn
df['d_type'] = df['d'].apply(type_udf)

# Note the new column 'd_type'
df
# FlickerDataFrame[a: bigint, b: string, c: double, d: map<string,bigint>, d_type: string]

# Use PySpark functions to compute the frequency table based on type of column 'd'
df.groupby(['d_type']).count()()
#   d_type count
# 0   dict     2

# You can, of course, just use the value_counts method.
df['d_type'].value_counts()()
#   d_type count
# 0   dict     2

# You don't have to assign the result of .apply() method. Result of .apply() method is still a FlickerColumn 
# object, which lets you use any FlickerColumn method.
df['d'].apply(len_udf)()
#   _len(d)
# 0       0
# 1       1

df['d'].apply(len_udf).describe()
#           _len(d)
# summary
# count           2
# mean          0.5
# stddev   0.707107
# min             0
# max             1
```

## Why not use `pyspark.pandas` (formerly `koalas`)?
[Koalas](https://koalas.readthedocs.io/en/latest/index.html) was a pandas API over Apache Spark, 
which was [officially included](https://issues.apache.org/jira/browse/SPARK-34849) in PySpark as 
[`pyspark.pandas`](https://spark.apache.org/docs/latest/api/python/migration_guide/koalas_to_pyspark.html).
Koalas is now deprecated and directs users towards `pyspark.pandas`.
You can see the documentation for `pyspark.pandas` [here](https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/index.html).

While `pyspark.pandas` is the official, user-friendly dataframe API, there are three noticeable issues when working with 
it.
1. `pyspark.pandas` crashes with frequent `java.lang.OutOfMemoryError` errors even when `pyspark.sql.DataFrame` is 
capable of handling the same data size
2. `pyspark.pandas`'s design runs unnecessary spark queries even for non-data-dependent tasks (such as getting the 
documentation), making interactive use too cumbersome for real-life use 
3. `pyspark.pandas` inherits all the design choices, whether good or bad, from pandas

We demonstrate the above issues with a real-life example. You may need to set up your system for PySpark before 
you can run the example yourself; see [pyspark-starter](https://github.com/ankur-gupta/pyspark-starter).

For this example, we will use the publicly available 
[Speedtest by Ookla](https://registry.opendata.aws/speedtest-global-performance/) dataset on AWS S3.
This dataset has ~186M rows. The following code snippet was run on an Apple Macbook Pro (M2 Max, 32 GB).

```python
# pyspark.pandas.DataFrame: Frequent OOM errors
import pyspark.pandas as ps
pdf = ps.read_parquet('s3a://ookla-open-data/parquet/performance/type=*/year=*/quarter=*/*.parquet')
type(pdf)  # pyspark.pandas.frame.DataFrame
print(f'{pdf.shape[0] / 1e6: .1f}M')  # 185.8M

# Just getting one row fails
pdf.head(1)
# [Stage 5:=======>                           (9 + 12) / 70]
# java.lang.OutOfMemoryError: Java heap space
# ConnectionRefusedError: [Errno 61] Connection refused

# Python session is now unusable: we cannot run any other query now. 
pdf.head(1)  # fails immediately
# ConnectionRefusedError: [Errno 61] Connection refused

# We cannot even get the shape
pdf.shape  # fails immediately
# ConnectionRefusedError: [Errno 61] Connection refused

# We cannot reload the data either. Python session needs to be restarted. 
pdf = ps.read_parquet('s3a://ookla-open-data/parquet/performance/type=*/year=*/quarter=*/*.parquet')
# ConnectionRefusedError: [Errno 61] Connection refused
```

After encountering the OOM error, the entire python session is useless. We now need to restart the python session if 
we want to run more code. These OOM errors are too frequent and require frequent python restarts.   

While one might assume that this data is simply too big for our machine, that's not the case. 
We can easily load the same data as a Flicker (or `pyspark.sql.DataFrame`) dataframe, as shown below.

```python
# flicker.FlickerDataFrame or pyspark.sql.DataFrame: No OOM errors
# See https://github.com/ankur-gupta/pyspark-starter to set up your system for Pyspark and AWS S3. 
from pyspark.sql import SparkSession
from flicker import FlickerDataFrame
spark = SparkSession.builder.appName('PySparkShell').getOrCreate()

df = FlickerDataFrame(spark.read.parquet('s3a://ookla-open-data/parquet/performance/type=*/year=*/quarter=*/*.parquet'))
print(f'{df.shape[0] / 1e6: .1f}M')  # 185.8M

type(df)  # flicker.dataframe.FlickerDataFrame
df()  # no OOM error
# [Stage 5:>                                    (0 + 1) / 1]
#             quadkey                                               tile avg_d_kbps avg_u_kbps avg_lat_ms tests devices
# 0  0022133222312322  POLYGON((-160.02685546875 70.6435894914449, -1...       8763       3646         45     1       1
# 1  0022133222330013  POLYGON((-160.032348632812 70.6399478155463, -...       9195       3347         43     1       1
# 2  0022133222330023  POLYGON((-160.043334960938 70.6363054807905, -...       6833       3788         42     1       1
# 3  0022133222330100  POLYGON((-160.02685546875 70.6417687358462, -1...       8895       3429         43     2       2
# 4  0022320121121332  POLYGON((-166.739501953125 68.3526207780586, -...       4877        935         45     3       2

df.shape  # returns immediately because nrows is cached
# (185832935, 7)
```

Let's give `pyspark.pandas.DataFrame` another shot. In a new python session, let's try some basic computation.
```python
# pyspark.pandas.DataFrame: Runs unnecessary spark queries
import pyspark.pandas as ps
pdf = ps.read_parquet('s3a://ookla-open-data/parquet/performance/type=*/year=*/quarter=*/*.parquet')

pdf.columns  # returns immediately
# Index(['quadkey', 'tile', 'avg_d_kbps', 'avg_u_kbps', 'avg_lat_ms', 'tests', 'devices'],
#        dtype='object')

# Just printing a column errors out. This is because pyspark.pandas tries to load too much 
# data at every operation. 
pdf['tests']  # OOM error
# [Stage 2:=======>                           (9 + 12) / 70]
# java.lang.OutOfMemoryError: Java heap space
# ConnectionRefusedError: [Errno 61] Connection refused

# As before, this python session is now useless. 
pdf['tests']
# ConnectionRefusedError: [Errno 61] Connection refused
```
As the above snippet shows, `pyspark.pandas` runs a spark query in order to mimic pandas-like behavior. 
This causes `pyspark.pandas` to load too much data at nearly every operation, which 
exacerbates the OOM error problem. Even if there is no OOM error, the repeated loading of too much data slows down 
everything, even when we don't actually want to perform a data-heavy computation. Here is an example.

```python
# In yet another new python session
# pyspark.pandas.DataFrame: Runs unnecessary spark queries
import pyspark.pandas as ps
pdf = ps.read_parquet('s3a://ookla-open-data/parquet/performance/type=*/year=*/quarter=*/*.parquet')

# Note that we didn't actually call the `.value_counts` method yet.
pdf['tests'].value_counts
# [Stage 2:=======>                           (9 + 12) / 70]
# java.lang.OutOfMemoryError: Java heap space
# ConnectionRefusedError: [Errno 61] Connection refused
```
We have to restart our python session yet again. This is what makes `pyspark.pandas` impractical for interactive use.
Interactive use is arguably the most important reason for the existence of a pandas-like API on Spark.

Sometimes, we are able to execute some operations before we eventually run out of memory.
```python
# In yet another new python session
import pyspark.pandas as ps
pdf = ps.read_parquet('s3a://ookla-open-data/parquet/performance/type=*/year=*/quarter=*/*.parquet')

type(pdf['tests'])  # pyspark.pandas.series.Series
pdf['tests'].value_counts()
# [Stage 2:=======>                           (9 + 12) / 71]
# 1       59630189
# 2       25653143
# 3       15311171
# 4       10511920
# 5        7820975
# 6        6146086
# 7        4972223
# ...
# 982          205
# 986          204
# Name: tests, dtype: int64
# Showing only the first 1000

pdf['tests'] == 0
# [Stage 5:===============>                  (19 + 12) / 70]
# java.lang.OutOfMemoryError: Java heap space
# ConnectionRefusedError: [Errno 61] Connection refused
```

The OOM error can happen even when the printable output is known to be small.
```python
# In yet another new python session
import pyspark.pandas as ps
pdf = ps.read_parquet('s3a://ookla-open-data/parquet/performance/type=*/year=*/quarter=*/*.parquet')

pdf['devices'].head(1)  # no OOM error
# 0    1
# Name: devices, dtype: int64

# Getting the documentation requires executes a spark query, which may or may not fail.
# Even without a failure, we must wait for the spark query to run just to see the documentation.
# This kills interactivity.
?pdf.apply
# [Stage 2:=======>                           (9 + 12) / 70]
# java.lang.OutOfMemoryError: Java heap space
# py4j.protocol.Py4JNetworkError: Error while sending or receiving

# Your machine may not see an error yet, but it'll still take a long time before you can see the documentation.
# Even if you encounter, you should still see the documentation after the long time.
# However, you'll now have to reload your python session.
pdf['devices'].head(1)
# ConnectionRefusedError: [Errno 61] Connection refused
```
`pyspark.pandas` is designed in a way that requires the execution of a spark query even if all you want to do is get
documentation of a method. This "feature" kills interactivity. You'll have to wait a long time for the query to execute
before you'll see the documentation. If the query finishes successfully, you'll just have had to wait a long time, which
can be annoying. If the query runs into an OOM error, you will still be able to see the documentation (after the long 
wait) but your python session will now need to be restarted.

Flicker and PySpark's original interface (`pyspark.sql.DataFrame`) does not suffer from these problems.
Obviously, both `flicker.FlickerDataFrame` and `pyspark.sql.DataFrame` can encounter OOM errors, but they don't suffer 
from design-induced OOM that are frequently with `pyspark.pandas`. 
Neither `pyspark.pandas` nor `pyspark.sql.DataFrame` run a spark query just to access documentation.

```python
# flicker.FlickerDataFrame or pyspark.sql.DataFrame: No spark query to access documentation
# See https://github.com/ankur-gupta/pyspark-starter to set up your system for Pyspark and AWS S3. 
from pyspark.sql import SparkSession
from flicker import FlickerDataFrame
spark = SparkSession.builder.appName('PySparkShell').getOrCreate()

df = FlickerDataFrame(spark.read.parquet('s3a://ookla-open-data/parquet/performance/type=*/year=*/quarter=*/*.parquet'))
df  # returns immediately
# FlickerDataFrame[quadkey: string, tile: string, 
#                  avg_d_kbps: bigint, avg_u_kbps: bigint, avg_lat_ms: bigint, 
#                  tests: bigint, devices: bigint]

df.names
# ['quadkey', 'tile', 'avg_d_kbps', 'avg_u_kbps', 'avg_lat_ms', 'tests', 'devices']

df['tests']  # returns immediately
# FlickerColumn<'tests'>

df[['tests']]()  # no OOM error
#   tests
# 0     1
# 1     1
# 2     1
# 3     2
# 4     3

df['tests'].value_counts()  # returns immediately
# FlickerDataFrame[tests: bigint, count: bigint]

df['tests'].value_counts()()  # no OOM error
# [Stage 7:=======>                           (9 + 12) / 71]
#   tests     count
# 0     1  59630189
# 1     2  25653143
# 2     3  15311171
# 3     4  10511920
# 4     5   7820975

df['tests'] == 0  # returns immediately
# FlickerColumn<'(tests = 0)'>

(df['tests'] == 0).value_counts()  # returns immediately
# FlickerDataFrame[(tests = 0): boolean, count: bigint]

(df['tests'] == 0).value_counts()()  # no OOM error
# [Stage 14:=======>                          (9 + 12) / 71]
#   (tests = 0)      count
# 0       False  185832935

?df.head  # no spark query, no OOM error
# Signature: df.head(n: 'int | None' = 5) -> 'FlickerDataFrame'
# Docstring: <no docstring>
# File:      .../flicker/src/flicker/dataframe.py
# Type:      method

df[['devices']](1)  # no OOM error
#   devices
# 0       1

df[['devices']].value_counts()  # still no OOM error
# [Stage 11:=======>                         (10 + 12) / 71]
#   devices     count
# 0       1  92155855
# 1       2  30108598
# 2       3  15262519
# 3       4   9404369
# 4       5   6455647

# This is a more time-consuming query, but if the original `spark.sql.DataFrame` can run it,
# then so can `flicker.FlickerDataFrame`.
(df['quadkey'].astype('Long') % 5).value_counts()()
# [Stage 31:=======>                          (9 + 12) / 71]
#   (CAST(quadkey AS BIGINT) % 5)     count
# 0                             0  46468834
# 1                             2  46456313
# 2                             1  46454483
# 3                             3  46453305
# Weird! No 'quadkey' % 5 == 4. 
# We can keep going. No OOM error. No need to restart our python session.
```
We can keep going. There is always a chance that we will encounter an OOM for some operation. 
For example, we will get an expected OOM error if we tried to convert the entire ~186M-row `flicker.DataFrame` into 
a `pandas.DataFrame`. But, OOM errors are much less frequent than with `pyspark.pandas`. 
In our experiments, it is more likely that your AWS S3 token will expire before you encounter an OOM error. And, 
unlike, `pyspark.pandas`, `flicker` does not run unnecessary spark queries. This makes interactive use very efficient and predictable. 

## Status
`flicker` is actively being developed. While `flicker` is ready for use, please note that 
some API may be changed in the future. Also, it is very likely that you will need a function that has not yet 
written in `flicker`. In such cases, you can always use the underlying PySpark DataFrame to do
every operation that PySpark supports. Please consider filing an issue for
missing functions, bugs, or unintuitive API. Happy sparking!

## License
`flicker` is available under [Apache License 2.0](https://github.com/ankur-gupta/flicker/blob/master/LICENSE).

`flicker` depends on other python packages listed in [requirements.txt](https://github.com/ankur-gupta/flicker/blob/master/requirements.txt)
which have their own licenses. `flicker` releases do not bundle any code from
the dependencies directly.

The documentation is made using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
theme which has [MIT License](https://squidfunk.github.io/mkdocs-material/license/).
