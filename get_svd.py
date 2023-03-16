import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import  col, sum, desc, row_number, asc, max, month, dayofmonth, hour
import os
import sys
import time
import numpy
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql import functions as F

conf = SparkConf().setAppName("SVD example")
conf.set("spark.pyspark.mllib.linalg.distributed.SingularValueDecomposition.ncv", "500")
conf.set("spark.sql.pivotMaxValues", 50000)
conf.set("spark.pyspark.python.computeSVD.maxIter", "200")
spark = SparkSession.builder.config(conf=conf).getOrCreate()

start= time.time()
data_df =spark.read.option("header",True).csv("data_df.csv")
user_ids = data_df["user_id"]
data_df = data_df.drop("user_id")
k = 10 #dimensions
data_df = data_df.select(*( F.col(col).cast("double").alias(col) for col in data_df.columns))
data_df_na = data_df.fillna(value=0)
mean_cols = [(mean(col(c)).over(Window.partitionBy())) for c in data_df_na.columns]
data_df_na = data_df_na.select(*[(col(c) - mean_cols[i]).alias(c) for i, c in enumerate(data_df_na.columns)])
your_rdd = data_df_na.rdd.map(list)
rowmatrix=RowMatrix(your_rdd)
svd = rowmatrix.computeSVD(k, computeU=True)
res = svd.U.rows.map(lambda x: (x,)).toDF()
res = res.withColumn("_1",vector_to_array("_1"))
res = res.select([res._1[x] for x in range(k)])
res.write.option("header",True).csv("U"+str(k) +".csv", mode="overwrite")
user_ids.write.option("header",True).csv("userids.csv", mode="overwrite")

end = time.time()
print("Query 1 took {} sec".format(start-end))

