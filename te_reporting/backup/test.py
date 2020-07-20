from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Python Spark SQL basic example").config("spark.some.config.option", "some-value").getOrCreate()
dataset = [['event1','3','1','2'],['event2','12','10','11'], ['ctu','1','1','1']]
df = spark.createDataFrame(dataset,['column','max','min','mean'])
