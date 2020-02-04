# Flicker
![build](https://github.com/ankur-gupta/flicker/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/ankur-gupta/flicker/branch/master/graph/badge.svg)](https://codecov.io/gh/ankur-gupta/flicker)
[![PyPI version](https://badge.fury.io/py/flicker.svg)](https://badge.fury.io/py/flicker)

This python package provides a `FlickerDataFrame` object. `FlickerDataFrame` 
is a wrapper over `pyspark.sql.DataFrame`. The aim of `FlickerDataFrame` is to 
provide a more Pandas-like dataframe API.

Flicker is like [Koalas](https://github.com/databricks/koalas) in that Flicker attempts to provide a pandas-like API. But there are strong differences in design. 
