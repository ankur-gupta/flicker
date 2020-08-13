# Quick Example (https://flicker.perfectlyrandom.org/quick-example/)
from pyspark.sql import SparkSession
from flicker import FlickerDataFrame

# Start Spark
spark = SparkSession.builder.appName('PySparkShell').getOrCreate()
pyspark_df = spark.createDataFrame(
    [(1, 'Turing', 41), (2, 'Laplace', 77), (3, 'Kolmogorov', 84)],
    'id INT, name STRING, age INT')

# ----------------------
# Create the DataFrame
# ----------------------
df = FlickerDataFrame(pyspark_df)

# ----------------------
# Print the DataFrame
# ----------------------
df
# FlickerDataFrame[id: int, name: string, age: int]
pyspark_df
# DataFrame[id: int, name: string, age: int]

df()
#    id        name  age
# 0   1      Turing   41
# 1   2     Laplace   77
# 2   3  Kolmogorov   84

pyspark_df.show()
# +---+----------+---+
# | id|      name|age|
# +---+----------+---+
# |  1|    Turing| 41|
# |  2|   Laplace| 77|
# |  3|Kolmogorov| 84|
# +---+----------+---+

pandas_df_sample = df()
pandas_df_sample['name'].values
# array(['Turing', 'Laplace', 'Kolmogorov'], dtype=object)

pandas_df_sample = pyspark_df.limit(5).toPandas()
pandas_df_sample['name'].values
# array(['Turing', 'Laplace', 'Kolmogorov'], dtype=object)

# -------------------------
# Inspect shape and columns
# -------------------------
df.shape
# (3, 3)

(pyspark_df.count(), len(pyspark_df.columns))
# (3, 3)

df.names
# ['id', 'name', 'age']
df.dtypes
# [('id', 'int'), ('name', 'string'), ('age', 'int')]

pyspark_df.columns
# ['id', 'name', 'age']
pyspark_df.dtypes
# [('id', 'int'), ('name', 'string'), ('age', 'int')]

# ---------------------------
# Extracting a column
# ---------------------------
df['name']  # not a FickerDataFrame object
# Column<b'name'>

pyspark_df['name']  # not a pyspark.sql.DataFrame object
# Column<b'name'>

df[['name']].distinct().nrows
# 3

pyspark_df[['name']].distinct().count()
# 3

# ---------------------------
# Extracting multiple columns
# ---------------------------
df[['name', 'age']]
# FlickerDataFrame[name: string, age: int]

pyspark_df[['name', 'age']]
# DataFrame[name: string, age: int]

# ---------------------------
# Creating a new column
# ---------------------------
df['is_age_more_than_fifty'] = df['age'] > 50
df()  # Must print to see the output
#    id        name  age  is_age_more_than_fifty
# 0   1      Turing   41                   False
# 1   2     Laplace   77                    True
# 2   3  Kolmogorov   84                    True

pyspark_df = pyspark_df.withColumn('is_age_more_than_fifty', pyspark_df['age'] > 50)
pyspark_df.show()  # Must print to see the output
# +---+----------+---+----------------------+
# | id|      name|age|is_age_more_than_fifty|
# +---+----------+---+----------------------+
# |  1|    Turing| 41|                 false|
# |  2|   Laplace| 77|                  true|
# |  3|Kolmogorov| 84|                  true|
# +---+----------+---+----------------------+

# ---------------------------
# Filtering
# ---------------------------
# Use boolean column to filter
df[df['is_age_more_than_fifty']]
# FlickerDataFrame[id: int, name: string, age: int, is_age_more_than_fifty: boolean]

# Filter and print in one-line
df[df['age'] < 50]()
#    id    name  age  is_age_more_than_fifty
# 0   1  Turing   41                   False

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

# ---------------------------
# Common operations
# ---------------------------
df.value_counts('name')
# FlickerDataFrame[name: string, count: bigint]

df.value_counts('name')()
#          name  count
# 0      Turing      1
# 1     Laplace      1
# 2  Kolmogorov      1

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

df.value_counts('is_age_more_than_fifty', normalize=True,
                sort=True, ascending=True)()
#    is_age_more_than_fifty     count
# 0                   False  0.333333
# 1                    True  0.666667

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

# ---------------------------
# Chain everything together
# ---------------------------
df[df['age'] < 50].rows_with_max('age')[['name']]()['name'][0]
# 'Turing'

filtered_df = pyspark_df[pyspark_df['age'] < 50]
age_max = filtered_df.agg({'age': 'max'}).collect()[0][0]
filtered_df[filtered_df['age'].isin([age_max])][['name']].toPandas()['name'][0]
# 'Turing'

# ---------------------------
# Get the PySpark dataframe
# ---------------------------
pyspark_df = df._df
processed_pyspark_df = df[df['age'] < 50].rows_with_max('age')._df
