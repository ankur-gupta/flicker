# Overview
Flicker is a pure-python package that aims to provide a pythonic API to
PySpark dataframes.

A PySpark dataframe (`pyspark.sql.DataFrame`) is a distributed dataframe that
lets you process data that is too large to fit on a single machine. PySpark
dataframe is arguably the most popular distributed dataframe implementation
at the time of writing. It has been used extensively in real-world production
systems. But, PySpark is based on Spark which is written in Scala[^1].
PySpark dataframe API is almost entirely the same as the Spark (Scala)
dataframe API. As a result, many python programmers feel that writing
PySpark code is like  _Writing Scala code in Python syntax_.

Consider the following snippet that adds a new column to a PySpark dataframe.
```python
pyspark_df = pyspark_df.withColumn('new_column', lit(1))
```
In contrast, the same operation can be done for a pandas dataframe like this
```python
pandas_df['new_column'] = 1
```

The difference PySpark and pandas dataframe APIs may not seem significant
based on the simple example above. In real-life, the verbosity and lack of
pythonic patterns makes working with PySpark unappealing. This is especially
true when performing ad-hoc, interactive data analysis (instead of writing
durable, well-tested production code). For example, it is tiring to type out
`df.show()` after every snippet of code to see the first few rows of the
dataframe. Common operations such as renaming multiple columns at once or
joining two PySpark dataframes requires a lot of boiler plate code that is
essentially the same thing over and over again.

Flicker provides `FlickerDataFrame` which wraps the common operations into
an intuitive, pythonic, well-tested API that does not compromise
performance at all.
<!-- [FIXME: see design notes, pyspark vs flicker tutorial] -->
Flicker reduces boiler-plate code by providing you a well-tested method
to perform common operations with less verbosity.

The best analogy is the following:

!!! Analogy
    `keras` is to `tensorflow` as `flicker` is to `pyspark`

Some of the reasons why people found TensorFlow 1.x API tedious are the same
reasons why python programmers find PySpark dataframe API tedious.
Tediousness of using TensorFlow 1.x
[led to the invention](https://blog.keras.io/author/francois-chollet.html)
of Keras.
[Keras API](https://keras.io/why_keras/)
did not handle the mathematical and algorithmic complexities
(such as graph construction and automatic differentiation) that are required
of a self-sufficient ML library such as TensorFlow. Instead, Keras simply
provided a better API to TensorFlow by using TensorFlow as a backend[^2].

Flicker is similar to Keras, in spirit. Flicker does not handle the
complexities of distributed data processing. Instead, Flicker provides a
better API to use PySpark as a backend.

## What's next?
Check out the [Quick Example](quick-example.md) to see how Flicker compares
with PySpark. See installation instructions in
[Getting Started](getting-started/installation/).

[^1]: [Spark SQL: Relational Data Processing in Spark](http://people.csail.mit.edu/matei/papers/2015/sigmod_spark_sql.pdf). Michael Armbrust, Reynold S. Xin, Cheng Lian, Yin Huai, Davies Liu, Joseph K. Bradley, Xiangrui Meng, Tomer Kaftan, Michael J. Franklin, Ali Ghodsi, Matei Zaharia. SIGMOD 2015. June 2015.
[^2]: Keras API was eventually [adopted](https://github.com/keras-team/keras/issues/5050) by TenorFlow. Keras API (`tf.keras`) is the default API for TensorFlow 2.x.
