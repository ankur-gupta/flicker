# Overview
A quick example is provided [here](https://github.com/ankur-gupta/flicker).

!!! Analogy
    `keras` is to `tensorflow` as `flicker` is to `pyspark`

## Quick Example

```python linenums="1"
from pyspark.sql import SparkSession
from flicker import FlickerDataFrame

# (Optional) Create a spark session, if needed.
spark = SparkSession.builder.appName('PySparkShell').getOrCreate()

# Create a dummy Flicker DataFrame using normally distributed random data of shape (100, 3)
df = FlickerDataFrame.from_shape(spark, nrows=100, ncols=3,
                                 names=['a', 'b', 'c'], fill='randn')

# See the nice printed dataframe in ipython/jupyter
df

# Pandas-like API to inspect a FlickerDataFrame
print(df.shape)
# (100, 3)

print(df.names)
# ['a', 'b', 'c']

print(df.dtypes)
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
df()
#           a         b         c  is_a_positive
# 0 -0.488747 -0.378013  0.350972          False
# 1  0.224332  0.322416 -0.943630           True
# 2  0.249755 -0.738754 -0.060325           True
# 3  1.108189  1.657239 -0.114664           True
# 4  1.768242 -2.422804 -1.012876           True

# These features can intermixed in nearly every imaginable way. Here are some quick examples.
# Example 1: show the first 5 rows of the dataframe that has only 'a' and 'c' columns selected.
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

## Motivation
PySpark DataFrame API is a copy of the Scala API.

## What's next?
Try out the [Tutorial](getting-started/installation.md) or the Examples.
