from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col
import glob
def get_df_with_descriptive_stats_for_columns( spark , df):

    """
    Get the columns names and for every column create a tupel col, maximum , minumium to make it sutable to create a datafrma out out tuples
    (event1, 3, 1) 
    """

    columns = df.schema.names
    columns_with_stats = []  # append tuples to a list, later to create a spark df
    for col in columns: # for each column calculate stat values
        print(col)
        one_column_df = df.select( col )
        minimum = calc_column_min( one_column_df )
        maximum = calc_column_max( one_column_df )
        mean = calc_column_avg( one_column_df )
        stddev = calc_column_stddev( one_column_df )
        median = calc_column_median( one_column_df )
        columns_with_stats.append((col,maximum, minimum, mean, stddev, median))
    print(columns_with_stats)
    columns_with_stats_df  = spark.createDataFrame( columns_with_stats, ['column_name','min','max','mean',  'stddev','median'] )

    return columns_with_stats_df 


def calc_column_max( one_col_df ):
    
    column = one_col_df.columns[0]
    summary_value = one_col_df.agg({column : 'max'}).collect()[0][0]
    try:
        summary_value = round( summary_value, 2)
    except Exception as e:
        print(e)
    
    return summary_value


def calc_column_min( one_col_df ):
    column = one_col_df.columns[0]
    summary_value = one_col_df.agg({column : 'min'}).collect()[0][0]
    try:
        summary_value = round( summary_value, 2)
    except Exception as e:
        print(e)
    
    return summary_value


def calc_column_avg( one_col_df ):
    
    column = one_col_df.columns[0]
    summary_value = one_col_df.agg({column : 'avg'}).collect()[0][0]
    try:
        summary_value = round( summary_value, 2)
    except Exception as e:
        print(e)
    
    return summary_value


def calc_column_stddev( one_col_df ):
    column = one_col_df.columns[0]
    summary_value = one_col_df.agg({column : 'stddev'}).collect()[0][0]
    try:
        summary_value = round( summary_value, 2)
    except Exception as e:
        print(e)
    
    return summary_value

def calc_column_count( one_col_df ):
    column = one_col_df.columns[0]
    summary_value = one_col_df.agg({column : 'stddev'}).collect()[0][0]

    return summary_value


def calc_column_median( one_col_df ):
    """
    """ 
    column = one_col_df.columns[0]
    changed_type_df = change_to_int_type(one_col_df ) # for calculating median we need to change datatype
    summary_value = changed_type_df.approxQuantile(column, [0.5], 0.25)
    if len(summary_value) == 0: 
        summary_value = 0.0 # if there is no value has to return null otherwide when creating df it throws an error
    else: summary_value = summary_value[0]
    try:
        summary_value = round( summary_value, 2)
    except Exception as e:
        print(e)
    
    return summary_value



def change_to_int_type(one_col_df):
    """
    For quantile calculations I had to change a datatype because of the follogin error
    Quantile calculation for column CTU with data type StringType is not supported.
    """
    column = one_col_df.columns[0]
    changed_type_df = one_col_df.withColumn(column, one_col_df[column].cast(DoubleType()))
    return changed_type_df



def get_delta_descriptive_stats_df(joined_df, suffix ):
    """
    Suffix deferentiate simmilar columns  between 2 dataframes
    Get delta stats between 2 dfs
    instead of doing it a long way 
    delta_descriptive_stats_df_min = joined_df.withColumn("delta_min", col("min_" + suffix) - col("min"))
    delta_descriptive_stats_df_min_max = joined_df_min.withColumn("delta_max", col("max_" + suffix) - col("max"))
    ....
    i made it short in a loop
    """
    delta_descriptive_stats_df = joined_df
    summary_stats = ['min', 'max', 'mean', 'stddev', 'median']
    for summary_stat in summary_stats:
        delta_descriptive_stats_df = delta_descriptive_stats_df .withColumn("delta_" + summary_stat, 
                                        col(summary_stat + "_" + suffix) - col(summary_stat))
    return delta_descriptive_stats_df.select(["column_name","delta_min", "delta_max", "delta_mean", "delta_stddev",  "delta_median" ])
