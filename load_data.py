import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import  col, sum, desc, row_number, asc, max, month, dayofmonth, hour
from pyspark.sql.types import StructType, StringType, IntegerType
import os
import sys
import time
print(os.environ["PYSPARK_PYTHON"] )
import numpy

conf = SparkConf().setAppName("SVD example")
conf.set("spark.pyspark.mllib.linalg.distributed.SingularValueDecomposition.ncv", "500")
conf.set("spark.sql.pivotMaxValues", 50000)
conf.set("spark.driver.memory", "8g") 
conf.set("spark.pyspark.python.computeSVD.maxIter", "200")

spark = SparkSession.builder.config(conf=conf).getOrCreate()

start= time.time()
print("spark session created")
schema = StructType().add("index", IntegerType(), True).add("user_id",StringType(), True).add("isbn", StringType(), True).add("book_ratings", IntegerType(), True)
data_df =spark.read.option("header",True).format("csv").schema(schema).load("train1.csv")

data_df = data_df.drop("index")
data_df = data_df.groupBy("user_id").pivot("isbn").sum("book_ratings")

data_df.write.option("header",True).csv("data_df.csv", mode="overwrite")
end = time.time()
print("Query 1 took {} sec".format(start-end))

