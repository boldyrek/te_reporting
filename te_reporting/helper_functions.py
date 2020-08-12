from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from xlsxwriter.workbook import Workbook
import pandas as pd
from openpyxl import load_workbook
import importlib.util
from pyspark.sql.functions import lit

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


def get_imputed_df( IMPUTATION_TRAIN_PATH, IMPUTATION_PREDICT_PATH ):
    """
    Load rack files: genereal preprocessing, preprocessing, imputed
    """
  
    imputed_train_df = load_df( IMPUTATION_TRAIN_PATH )
    imputed_predict_df = load_df( IMPUTATION_PREDICT_PATH )
    imputed_train_df = imputed_train_df.withColumn("normal_2", lit(None)).select("*")
    imputed_predict_df = imputed_predict_df.withColumn("normal", lit(None)).select("*")
    result = imputed_train_df.union(imputed_predict_df)
    
    return result

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


def write_to_excel(df, sheet_name):
    """
    How to write to excel
    https://stackoverflow.com/questions/42370977/how-to-save-a-new-sheet-in-an-existing-excel-file-using-pandas
    """
    path = 'xls/output.xlsx'
    book = load_workbook(path)
    writer = pd.ExcelWriter(path, engine = 'openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df.toPandas().to_excel(writer, sheet_name = sheet_name)
    writer.save()
    writer.close()

   
def get_module_from_path( module_name , module_path ):

    spec = importlib.util.spec_from_file_location(  module_name , module_path )
    my_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_module)
    
    return my_module
    
  