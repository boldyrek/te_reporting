from pyspark.sql.types import IntegerType, DoubleType

def calc_column_max( one_col_df ):
    
    column = one_col_df.columns[0]
    summary_value = one_col_df.agg({column : 'max'}).collect()[0][0]

    return summary_value


def calc_column_min( one_col_df ):
    column = one_col_df.columns[0]
    summary_value = one_col_df.agg({column : 'min'}).collect()[0][0]

    return summary_value


def calc_column_avg( one_col_df ):
    column = one_col_df.columns[0]
    summary_value = one_col_df.agg({column : 'avg'}).collect()[0][0]

    return round( summary_value, 2)


def calc_column_stddev( one_col_df ):
    column = one_col_df.columns[0]
    summary_value = one_col_df.agg({column : 'stddev'}).collect()[0][0]

    return round( summary_value, 2)

def calc_column_count( one_col_df ):
    column = one_col_df.columns[0]
    summary_value = one_col_df.agg({column : 'stddev'}).collect()[0][0]

    return summary_value


def calc_column_median( one_col_df ):
    """
    """ 
    column = one_col_df.columns[0]
    changed_type_df = change_to_int_type(one_col_df ) # for calculating median we need to change datatype
    median = changed_type_df.approxQuantile(column, [0.5], 0.25)
    if len(median) == 0: 
        median = 0.0 # if there is no value has to return null otherwide when creating df it throws an error
    else: median = median[0]
    
    return round(median,2)



def change_to_int_type(one_col_df):
    """
    For quantile calculations I had to change a datatype because of the follogin error
    Quantile calculation for column CTU with data type StringType is not supported.
    """
    column = one_col_df.columns[0]
    changed_type_df = one_col_df.withColumn(column, one_col_df[column].cast(DoubleType()))
    return changed_type_df