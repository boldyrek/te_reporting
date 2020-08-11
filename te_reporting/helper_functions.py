from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def start_spark_session():
    """
    Starting spark session
    """

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    return spark
spark = start_spark_session()   

def get_imputed_df( IMPUTATION_TRAIN_PATH ):
    """
    Load rack files: genereal preprocessing, preprocessing, imputed
    """
    imputed_df = load_df( IMPUTATION_TRAIN_PATH )

    return imputed_df

def load_df( df_path ):
    """
    load any df
    """
    df = spark.read.format("csv").option("header", "true").load( df_path )
    return df


def suffix_and_join_dfs(df1, df2, on_column ):
    """
    give suffix 1 to first df and suffix 2 to last df
    """
    suffix2 = '2'
    df2 = attach_suffix_to_columns(df2, suffix2)
    on_column2 = on_column + '_' + suffix2
    joined_df = df1.join(df2, col(on_column) == col(on_column2) )
    
    return joined_df


def attach_suffix_to_columns( df, prefix ):    
    """
    """
    columns = df.columns
    df = df.select(*[col(x).alias(x + '_' + prefix) for x in columns])
    return df