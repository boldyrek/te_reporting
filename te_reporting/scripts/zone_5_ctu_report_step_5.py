"""
Zone 5 , step 5: ctu report
For every CTU we get min, max, mean, stddev ... summary statistics

	Create a table with row-wise listing of:			
1	CTU			Table 1
2	Min and max event dates in each CTU			Table 1
3	Length of each CTU in days			Table 2
4	Event descriptive stats across CTUs and all batches			Table 1
	Minimum			
	Maximum			
	Mean			
5	Target event stats across CTUs and all batches:			Table 1
	Minimum			
	Maximum			
	Mean	
    
    
"""

import pyspark
from pyspark.sql.functions import concat, col, lit, to_date
from col_stats import *
from helper_functions import get_imputed_df, start_spark_session, load_df, write_to_excel, get_module_from_path
import config as cfg



# def transform_event_stats_list_to_df(event_stats_list, CTU_num):
#     """
#     creata a df:
#     colmn_name stats_summary_name stats_summary_value
#     event 1       min             1
#     event 1       max             5
#     event 2       min             101
#     event 2       max             199
#     ......
#     based on input of list of typles like [(event1, min, 1), (event1, max, 5)]
#     """
    
#     cols_stats_df = spark.createDataFrame(\
#                                           event_stats_list,\
#                                           [ 'event_plus_summary',  str(CTU_num) ] )
    
#     return cols_stats_df 








def get_list_of_ctu_event_stats( single_ctu_df ):
    """
    Input a dataframe
    Output a list with stats for every columns [(column1, min, 1), (column1, max, 5), (column2, min,     105), ... ]
    """
    
    all_columns_summary_stats = []
    columns = single_ctu_df.schema.names
    for col in columns: # for each column calculate stat values
        df_one_col = single_ctu_df.select(col)
        maximum = calc_column_max( df_one_col )
        maximum_row = (col+'-max', str(maximum))
        all_columns_summary_stats.append( maximum_row )
        
        minimum =  calc_column_min( df_one_col )
        minimum_row = (col +'-min', str(minimum))
        all_columns_summary_stats.append( minimum_row )
        
        mean = calc_column_avg( df_one_col )
        mean_row = (col + '-mean', str(mean))
        all_columns_summary_stats.append( mean_row )
        
        stddev = calc_column_stddev( df_one_col )
        stddev_row = (col + '-stddev', str(mean))
        all_columns_summary_stats.append( stddev_row )
    length_of_ctu_df = get_length_of_ctu( single_ctu_df )
    all_columns_summary_stats.append(('length_of_ctu', str(length_of_ctu_df)))
    target_frequency = calc_target_frequency(  single_ctu_df )
    all_columns_summary_stats.append(('target_event_frequency', str(target_frequency)))
    return all_columns_summary_stats

def get_length_of_ctu(single_CTU_df):

    minimum_date = single_CTU_df.select(
        to_date(col('event_date')).alias('event_date')).agg({"event_date":'min'}).collect()[0][0]
    maximum_date  = single_CTU_df.select(
    to_date(col('event_date')).alias('event_date')).agg({"event_date":'max'}).collect()[0][0]
    length_of_days = maximum_date - minimum_date

    return length_of_days.days

def calc_target_frequency(  single_ctu_df ):
    te_constants = get_module_from_path( 'te_constants_master.py', cfg.TE_CONSTATS )
    ref_event_col = te_constants.REF_EVENT_COL
    where_clause = ref_event_col + '> 0'
    count_positive_target = single_CTU_df.where(where_clause).count()
    num_rows_total = single_CTU_df.count()
    frequency = num_rows_total / count_positive_target
    
    return round(frequency,2)

def get_df_with_dropped_garbage_cols(df):
    """
    Drop some of the unnesessary columns
    """
    columns_to_drop = ['level_0', 'index', 'Unnamed: 0', '_c0']
    df_to_drop = df.select('*')
    df_to_drop = df_to_drop.drop(*columns_to_drop)
    
    return df_to_drop



def get_joined_df( event_stats_all_ctus ):
    """
    Input: getting a list of dataframes for every CTU: 0,1,2,3 .. max_CTU 
    need to join them together in a loop along the axis 1 
    return a df with columns
    column_name stats_summary_name stats_summary_value
    We need to join like  this looking dataframes into one with all the CTUs:
      +-------------------+----+
        |event_plus_summary|   1|
        +-------------------+----+
        |         event1_max|   3|
        |         event1_min|   1|
        |        event1_mean| 2.0|
        |      event1_stddev| 2.0|
    """
    ctu_summary_joined_df = event_stats_all_ctus[0].select("*") # get first ctu df as a starting poin for joining
    for ctu_num , ctu_summary_stats in enumerate(event_stats_all_ctus[1:]):
        ctu_summary_joined_df = ctu_summary_joined_df.join(
                ctu_summary_stats, ctu_summary_joined_df.event_plus_summary \
                == ctu_summary_stats.event_plus_summary).\
                drop(ctu_summary_stats.event_plus_summary)
    #ctu_summary_joined_df.show().column_name == ctu_summary_stats.column_name)
    return ctu_summary_joined_df

def get_event_stats_df( single_CTU_clean_df, CTU_num ):
        """
        Output df looks like this:
        +-------------------+----+
        |event_plus_summary|   1|
        +-------------------+----+
        |         event1_max|   3|
        |         event1_min|   1|
        |        event1_mean| 2.0|
        |      event1_stddev| 2.0|
        """
        event_stats =  get_list_of_ctu_event_stats(  single_CTU_clean_df ) # getting a list summary stats
        cols_stats_df = spark.createDataFrame(\
                                          event_stats,\
                                          [ 'event_plus_summary',  str(CTU_num) ] )
    
        return cols_stats_df
        

def print_df (joined_ctu_event_stats_df ):
    joined_ctu_event_stats_df.select(
        ['event_plus_summary', '0', '1', '2']).show(200,truncate =False)

def get_max_CTU_num(df):
    
    df_CTU = df.select('CTU')
    max_CTU = calc_column_max( df_CTU ) # get the maximum values of a CTU, to determin num of CTUS
    
    max_CTU_int = int(float( max_CTU )) # convert string values like 9.0 into int 
    max_CTU_int = max_CTU_int + 1
    return max_CTU_int

def split_column_by_underscore(df):
    """
    Split columns_plus_summary column by undescore and 
    remove splited column 
    """
    split_col = pyspark.sql.functions.split(df['event_plus_summary'], '-')
    df = df.withColumn('event', split_col.getItem(0))
    df = df.withColumn('event_stats', split_col.getItem(1))
    df = df.drop('event_plus_summary')
    return df 

def get_event_cols_first ( joined_ctu_event_stats_df ):
    """
    put event event columns first 
    """
    cols = joined_ctu_event_stats_df.columns
    new_order_cols =  cols[-2:] + cols[:-2]
    return new_order_cols


    
"""
***** MAIN *******
"""



#file_name = "../data/example.csv"

spark = start_spark_session(  )
df = load_df(cfg.PREPROCESS_PATH)
#df = load_df(file_name)
#df = get_imputed_df(cfg.IMPUTATION_TRAIN_PATH)
event_stats_ctus_dfs = []
max_CTU_num = get_max_CTU_num(df)
for CTU_num in range(max_CTU_num):
    print(CTU_num)
    single_CTU_df = df.filter(f"CTU == { CTU_num }") # get a df with one CTU
    single_CTU_clean_df = get_df_with_dropped_garbage_cols( single_CTU_df )
    ctu_event_stats_df = get_event_stats_df( single_CTU_clean_df, CTU_num )
    event_stats_ctus_dfs.append( ctu_event_stats_df )

joined_ctu_event_stats_df = get_joined_df( event_stats_ctus_dfs )
#print_df (joined_ctu_event_stats_df)
joined_ctu_event_stats_df = split_column_by_underscore( joined_ctu_event_stats_df )
new_order_cols = get_event_cols_first( joined_ctu_event_stats_df)
joined_ctu_event_stats_df = joined_ctu_event_stats_df.select(new_order_cols).orderBy('event')
write_to_excel(joined_ctu_event_stats_df, "zone_5_ctu_report_step_5")