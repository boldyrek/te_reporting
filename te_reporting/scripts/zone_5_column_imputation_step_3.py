"""
Te version 2.7
Zone 5, step 3:  column imputation
							
	Task description						Automation
	After imputing at the raw column level, create table with row-wise listing of:						
1	Column name						Table 1
2	Imputation approach (Te version 2.7 implements only zero of forward fill)						Table 1
3	Column descriptive statistics:						Table 1
	minimum						
	maximum						
	mean						
	standard deviation						
	Median							
				
Input:  General preprocessing dataframe
Output: of the script should be in excell format and should looks like table bellow:
         minimum	maximum	mean	standard deviation	median	imputation approach
Column 1	1	     13	     7	      4	       8	     5            ffill
Column 2	1	     6	     4	      2	       4	     3            bfill
Column 3	101	     112	 106	  4	       107       7            bfill 	
…	…	…	…	…	…	
"""
import sys
sys.path.append("/home/boldyrek/mysoft/te/te_reporting/")
from sys import argv
from helper_functions import get_imputed_df, start_spark_session, load_df, write_to_excel, get_module_from_path
from col_stats import *
import config as cfg
import importlib.util
import  pyspark.sql.functions as spark_funcs





def get_summary_stats_for_every_column(df):
    """
    Input: Input df, and columns of that dataframe
    Calculate summary statistics for every column 
    Output: 
    
    """
    columns = df.columns
    columns_summary_stats = [] # append tuples to a list, later to create a spark df
    for col in columns: # for each column calculate stat values
        one_col_df = df.select(col) # select only nesessary colum
        maximum = calc_column_max(one_col_df)
#         print(maximum)
#         raise SystemExit
        minimum = calc_column_min(one_col_df)
        mean = calc_column_avg(one_col_df)
        stddev = calc_column_stddev(one_col_df)
        median = calc_column_median(one_col_df)
        columns_summary_stats.append((col, maximum, minimum, mean, stddev, median))
        columns_summary_stats_df = spark.createDataFrame( columns_summary_stats,\
                                       ['column', 'max', 'min', 'mean', 'stddev', 'median' ] )


    return columns_summary_stats_df

def get_column_imputation_approach_df( columns):
    te_constants = get_module_from_path('te_constants_master.py', cfg.TE_CONSTATS )
    imputation_cols = []
    for col in columns:
        if col in te_constants.FFILL_COLS:
            imputation_cols.append((col, 'ffil'))
        if col in te_constants.BFILL_COLS:
            imputation_cols.append((col, 'bfill'))
        if col in te_constants.CUMSUM_COLS:
            imputation_cols.append((col, 'cumsum'))
    imputation_cols_df = spark.createDataFrame(imputation_cols,\
                                           ['column_name2', 'imputation_approach' ] )  
    return imputation_cols_df


"""
****** MAIN ******
1. Create spark session 
2. Read the file into a dataframe
4. Calculate statistical summary for every column in a dataframe
5. Get imputation approach from the te_constants.py 
6. Join dfs from 4 and 5
7. Save it as an excel tab 
"""

file_name = "../data/example.csv"
# Step 1 create sparj session
spark = start_spark_session()
# Step 2 Read file into df
gen_pre_df = load_df(cfg.GEN_PREPROCESS_PATH)
# Step 4 Calculate statistical summary for every column in a dataframe
columns_summary_stats_df = get_summary_stats_for_every_column(gen_pre_df)
# Step 5 Get imputation approach from the te_constants.py 
columns = gen_pre_df.columns
imputation_cols_df = get_column_imputation_approach_df(columns)
# Step 6 Join dfs from 4 and 5
excel_ready_df = columns_summary_stats_df.join(imputation_cols_df, 
                     spark_funcs.col('column') == spark_funcs.col('column_name2'), "left_outer")
# Step 7 Save it as an excel tab 
excel_ready_df = excel_ready_df.drop("column_name2")
write_to_excel(excel_ready_df, "zone_5_col_imputation_step_3")
