# Quick Example
This example describes how to use `flicker`. This also provides a good 
comparison between `flicker` and `pyspark` dataframe APIs.  
<!-- [FIXME: tutorial, flicker vs pyspark] -->
<!-- [FIXME: link to this example in jupyter notebook] -->

The example assumes that you have already installed `flicker`. Please see 
[Getting Started](/getting-started/installation/)
for installation instructions. This example uses `flicker 0.0.16` and 
`pyspark 2.4.5`.

You can follow along without having to install or setup a spark environment
by using the `flicker-playground` 
[docker image](https://hub.docker.com/r/ankurio/flicker-playground), 
as described in [Getting Started](/getting-started/installation/).

## Create the DataFrame
Let's create a Spark session and a PySpark dataframe to begin. This is the 
same as with PySpark. In some cases, you may already have a 
`spark: SparkSession` object defined for you (such as when running the 
`pyspark` executable or on [AWS EMR](https://aws.amazon.com/emr/)).
 
```python linenums="1"
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('PySparkShell').getOrCreate()
pyspark_df = spark.createDataFrame(
    [(1, 'Turing', 41), (2, 'Laplace', 77), (3, 'Kolmogorov', 84)], 
    'id INT, name STRING, age INT')
```

To use the benefits of `flicker`, let's create `FlickerDataFrame` from 
the PySpark dataframe. This is easy &ndash; just call the default constructor.
```python linenums="6"
from flicker import FlickerDataFrame
df = FlickerDataFrame(pyspark_df)
```
If you're following along this example in your own interactive python terminal,
you'll notice that the above step is pretty fast. A `FlickerDataFrame` simply 
wraps the PySpark dataframe within itself[^1] (you can access it at `df._df`).
Flicker does not copy any data from `pyspark_df` to `df` in the above 
code snippet. This means that no matter how big your PySpark dataframe is, 
creating a Flicker dataframe is always quick.

In the rest of this example, we show code for both Flicker and PySpark 
side-by-side.

## Print the DataFrame
Printing a distributed dataframe may require pulling data 
in from worker nodes, which in turn, may need to perform any un-executed 
operations before they can send the data. This makes printing PySpark 
or Flicker dataframes slow operations, unlike pandas. This is why neither 
PySpark nor Flicker shows you the contents when you print `df` or `pyspark_df`.

=== "Flicker"
    ```python
    df
    # FlickerDataFrame[id: int, name: string, age: int]
    ```

=== "PySpark"
    ```python
    pyspark_df
    # DataFrame[id: int, name: string, age: int]
    ```

In order to see the contents (actually, just the first few rows) of the 
dataframe, we can invoke the `.show()` method. Since we have to print 
dataframes very often (such as when performing interactive analysis), Flicker 
lets you "print" the first few rows by just calling the dataframe.

=== "Flicker"
    ```python
    df()
    #    id        name  age
    # 0   1      Turing   41
    # 1   2     Laplace   77
    # 2   3  Kolmogorov   84
    ```

=== "PySpark"
    ```python
    pyspark_df.show()
    # +---+----------+---+
    # | id|      name|age|
    # +---+----------+---+
    # |  1|    Turing| 41|
    # |  2|   Laplace| 77|
    # |  3|Kolmogorov| 84|
    # +---+----------+---+
    ```

If you're running the commands in a terminal, you will see the output like 
the one shown above. For this small example, the Flicker version of printed 
content looks unimpressive against the PySpark version but Flicker-printed 
content looks much better for bigger dataframes. If you're running the same 
commands in a Jupyter notebook, the Flicker-printed content you appear as 
a pretty, mildly-interactive HTML dataframe but the PySpark-printed content 
would just be text. 

Under the hood, Flicker and PySpark have very different behaviors. 
PySpark's `pyspark_df.show()` uses side-effects &ndash; it truly prints the 
formatted string to `stdout` and then returns `None`. Flicker's `df()`, on the 
other hand, returns a small pandas dataframe which then gets printed 
appropriately depending on the interactive tool 
(such as Jupyter or IPython)[^2]. This also means that if you wanted to 
inspect the printed dataframe, you could simply do this:

=== "Flicker"
    ```python
    pandas_df_sample = df()
    pandas_df_sample['name'].values
    # array(['Turing', 'Laplace', 'Kolmogorov'], dtype=object)
    ```

=== "PySpark"
    ```python
    pandas_df_sample = pyspark_df.limit(5).toPandas()
    pandas_df_sample['name'].values
    # array(['Turing', 'Laplace', 'Kolmogorov'], dtype=object)
    ```

Obviously, PySpark lets you do this too but with more verbosity.

## Inspect shape and columns
Flicker provides a pandas-like API. The same result may be obtained using 
PySpark with a little bit more verbosity.

=== "Flicker"
    ```python
    df.shape
    # (3, 3)
    ```

=== "PySpark"
    ```python
    (pyspark_df.count(), len(pyspark_df.columns))
    # (3, 3)
    ```
Note that Flicker still uses PySpark's `.count()` method under the hood to get 
the number of rows. This means that both Flicker and PySpark snippets above
may be slow the first time we run them. However, `FlickerDataFrame` "stores" 
the row count which means that invoking `df.shape` the second time should be 
instantaneous, as long as `df` is not modified since the first invocation. 

Getting the column names is also easy. Flicker differentiates between 
a column (`pyspark.sql.Column` object) and a column name (a `str` object). 
This is why we named the property `df.names` instead of `df.columns`.
Similar to PySpark dataframe, we can get the data types for all the columns.

=== "Flicker"
    ```python
    df.names
    # ['id', 'name', 'age']
    df.dtypes
    # [('id', 'int'), ('name', 'string'), ('age', 'int')]
    ```

=== "PySpark"
    ```python
    pyspark_df.columns
    # ['id', 'name', 'age']
    pyspark_df.dtypes
    # [('id', 'int'), ('name', 'string'), ('age', 'int')]
    ```

## Extracting a column
Unlike a `pandas.Series` object, the `pyspark.sql.Column` object does not 
materialize the column for us. Since Flicker is just an API over PySpark,
Flicker does not materialize the column either. 

=== "Flicker"
    ```python
    df['name']  # not a FickerDataFrame object
    # Column<b'name'>
    ```

=== "PySpark"
    ```python
    pyspark_df['name']  # not a pyspark.sql.DataFrame objecy
    # Column<b'name'>
    ```

PySpark does not provide a proper equivalent to the pandas' Series object. 
If we wanted to perform an operation on a column, we may still be able to it 
albeit in a round-about way. For example, we can count the number of 
distinct `name`s like this:

=== "Flicker"
    ```python
    df[['name']].distinct().nrows
    # 3
    ```

=== "PySpark"
    ```python
    pyspark_df[['name']].distinct().count()
    # 3
    ```

## Extracting multiple columns
Luckily, this is the same in Flicker, PySpark, and pandas. 
As previously mentioned, the contents don't get printed unless we specifically
ask for it.

=== "Flicker"
    ```python
    df[['name', 'age']]
    # FlickerDataFrame[name: string, age: int]
    ```

=== "PySpark"
    ```python
    pyspark_df[['name', 'age']]
    # DataFrame[name: string, age: int]
    ```
 
## Creating a new column
This is where Flicker shines &ndash; you can use pandas-like assignment API.
Observant readers may notice that the following makes `FlickerDataFrame`
objects mutable, unlike a `pyspark.sql.DataFrame` object which is immutable.
This is by design.  

=== "Flicker"
    ```python
    df['is_age_more_than_fifty'] = df['age'] > 50
    df()  # Must print to see the output  
    #    id        name  age  is_age_more_than_fifty
    # 0   1      Turing   41                   False
    # 1   2     Laplace   77                    True
    # 2   3  Kolmogorov   84                    True
    ```

=== "PySpark"
    ```python
    pyspark_df = pyspark_df.withColumn('is_age_more_than_fifty', pyspark_df['age'] > 50)
    pyspark_df.show()  # Must print to see the output  
    # +---+----------+---+----------------------+
    # | id|      name|age|is_age_more_than_fifty|
    # +---+----------+---+----------------------+
    # |  1|    Turing| 41|                 false|
    # |  2|   Laplace| 77|                  true|
    # |  3|Kolmogorov| 84|                  true|
    # +---+----------+---+----------------------+
    ```

The above combination of _perform an operation_ and then _print_ is a common 
pattern when performing interactive analysis. This is because simply 
executing `df['is_age_more_than_fifty'] = df['age'] > 50` does not actually
perform the computation. It's only when you print (or count or take any other 
[action](https://spark.apache.org/docs/latest/rdd-programming-guide.html#actions)), 
that the previously specified computation is actually performed. By printing 
immediately after specifying an operation helps catch errors early.  

## Filtering
This is also the same in Flicker, PySpark, and pandas.   

=== "Flicker"
    ```python
    # Use boolean column to filter
    df[df['is_age_more_than_fifty']]  
    # FlickerDataFrame[id: int, name: string, age: int, is_age_more_than_fifty: boolean]
    
    # Filter and print in one-line
    df[df['age'] < 50]()    
    #    id    name  age  is_age_more_than_fifty
    # 0   1  Turing   41                   False
    ```

=== "PySpark"
    ```python
    # Use boolean column to filter
    pyspark_df[pyspark_df['is_age_more_than_fifty']]  
    # DataFrame[id: int, name: string, age: int, is_age_more_than_fifty: boolean]
    
    # Filter and print in one-line
    pyspark_df[pyspark_df['age'] < 50].show()  
    # +---+------+---+----------------------+
    # | id|  name|age|is_age_more_than_fifty|
    # +---+------+---+----------------------+
    # |  1|Turing| 41|                 false|
    # +---+------+---+----------------------+
    ```

## Common operations
Flicker comes loaded with methods that perform common operations. A prime 
example is generating value counts, typically done in pandas via 
`.value_counts()` method. Flicker also provides this method with only 
minor (but sensible) modifications to the method arguments.

=== "Flicker"
    ```python
    df.value_counts('name')  
    # FlickerDataFrame[name: string, count: bigint]
    
    df.value_counts('name')()
    #          name  count
    # 0      Turing      1
    # 1     Laplace      1
    # 2  Kolmogorov      1
    ```

=== "PySpark"
    ```python
    pyspark_df.groupby('name').count()  
    # DataFrame[name: string, count: bigint]
    
    pyspark_df.groupby('name').count().show()  
    # +----------+-----+
    # |      name|count|
    # +----------+-----+
    # |    Turing|    1|
    # |Kolmogorov|    1|
    # |   Laplace|    1|
    # +----------+-----+
    ```

Even though the PySpark snippet above looks simple enough, it requires the 
programmer to know that they have to use `.groupby()` method to generate
value counts (much like in SQL). This additional cognitive load on the 
programmer is a hallmark of PySpark dataframe API. But, Flicker can do more
than that. 

=== "Flicker"
    ```python
    df.value_counts('is_age_more_than_fifty', normalize=True, 
                    sort=True, ascending=True)()  
    #    is_age_more_than_fifty     count
    # 0                   False  0.333333
    # 1                    True  0.666667
    ```

=== "PySpark"
    ```python
    nrows = pyspark_df.count()
    count_df = (pyspark_df.groupBy('is_age_more_than_fifty')
                .count()
                .orderBy('count', ascending=True))
    count_df.withColumn('count', count_df['count'] / nrows).show()
    # +----------------------+------------------+
    # |is_age_more_than_fifty|             count|
    # +----------------------+------------------+
    # |                 false|0.3333333333333333|
    # |                  true|0.6666666666666666|
    # +----------------------+------------------+
    ```

PySpark requires defining more variables to normalize the counts. We need to 
generate value counts for a lot of dataframes in order to simply inspect the 
data. The obvious solution is to wrap the PySpark code snippet into a function
and then re-use it. That's exactly what Flicker does! 

Generating value counts is only one such example. See other methods such as 
`any`, `all`, `min`, `rows_with_max` for other common examples. Even more 
useful are methods `rename` and `join` which do a lot more than the 
corresponding PySpark methods.

## Chain everything together
You can chain everything together into complex operations. Flicker can 
often do a sequence of operations in one line without having to define any 
temporary variables. 

=== "Flicker"
    ```python
    df[df['age'] < 50].rows_with_max('age')[['name']]()['name'][0]  
    # 'Turing'
    ```

=== "PySpark"
    ```python
    filtered_df = pyspark_df[pyspark_df['age'] < 50]
    age_max = filtered_df.agg({'age': 'max'}).collect()[0][0]
    filtered_df[filtered_df['age'].isin([age_max])][['name']].toPandas()['name'][0]
    # 'Turing'
    ```
It may appear that the Flicker expression above is too complicated to be 
meaningful. However, while performing interactive analysis, the above 
expression naturally arises as your mind sequentially searches for 
increasingly specific information. This is better experienced than can be 
described.

## Get the PySpark dataframe
If you have to use PySpark dataframe for some operations, you can easily 
get the underlying PySpark dataframe stored in the `._df` attribute.
This may be useful when there is no Flicker method available to perform 
an operation that can easily be performed with PySpark[^3]. You can also 
mix and match &ndash; perform some computation with Flicker and the rest with 
PySpark.   

=== "Flicker"
    ```python
    pyspark_df = df._df
    processed_pyspark_df = df[df['age'] < 50].rows_with_max('age')._df
    ``` 

## There is more
This example only describes the basic Flicker dataframe API. We note some 
advantages in this section:

* `FlickerDataFrame` does not allow duplicate column names and does not create 
duplicate column names (which PySpark dataframe does and then fails awkwardly).
* `FlickerDataFrame.rename` method lets you rename multiple columns at once
* `FlickerDataFrame.join` lets you specify the join condition using a `dict` 
and lets you add a suffix/prefix in one line of code. 
* `FlickerDataFrame` comes with many factory constructors such as 
`from_rows`, `from_columns`, and even a `from_shape` that lets you create 
`FlickerDataFrame` quickly.
* `flicker.udf` contains some commonly needed UDF functions such as `type_udf`
and `len_udf`
* `flicker.recipes` contains some more useful tools that are needed for 
real-world data analysis 

## Footnotes
[^1]: [Composition](https://en.wikipedia.org/wiki/Object_composition) is the fancy term for it.
[^2]: Conversion to a pandas dataframe can sometimes convert `np.nan` into `None`.
[^3]: If possible, please contribute by [filing a GitHub issue](https://github.com/ankur-gupta/flicker/issues/new) and/or sending a PR. 
