"""
Zone 5 stage I aggregation step 2
Task description	Automation
After aggregating data to the day-level, create a table with row-wise listing of:	
1. Event column name	Table1
2. Total number of events for column over total number of events for ALL columns	Table1
	
Rationale: Proportion of events for a particular event column. Expected to be similar to zone 4 proportions	

Table 1	
	
	Num events / Total num events 
Column 1	2%
Column 2	5%
Column 3	3%
Column 4	5%
Column 5	1%
…	…
Example 
Date  	Purchase	Calls	Money spend
1/1/00	0	3	50
1/2/00	1	0	51
1/3/00	0	4	52
1/4/00	0	0	32
1/5/00	1	1	0
	   0.33	0.6	0.80

"""

from sys import argv
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from helper_functions import get_imputed_df, start_spark_session, load_df, write_to_excel, get_module_from_path
import config as cfg

def drop_garbage_cols(df):
    """
    Drop some of the unnesessary columns
    """
    columns_to_drop = ['level_0', 'index', 'Unnamed: 0', '_c0', 'party_id', 'event_date', 'CTU', 'event_id']
    df_to_drop = df.select('*')
    df_to_drop = df_to_drop.drop(*columns_to_drop)
    
    return df_to_drop

"""
*** MAIN ***
"""


spark = start_spark_session()
prepro_df  = load_df(cfg.PREPROCESS_PATH)
num_rows = prepro_df.count()
event_rate_df = prepro_df.select([(F.count(F.when(prepro_df[c] != 0, c))/num_rows).alias(c) for c in prepro_df.columns])
event_rate_df_clean =  drop_garbage_cols( event_rate_df)
event_rate_df_clean_pd = event_rate_df_clean.toPandas().transpose().reset_index().rename(columns={0:'Column event rate ', 'index' : 'Column names'})
event_rate_df_clean_spark = spark.createDataFrame(event_rate_df_clean_pd)
write_to_excel(event_rate_df_clean_spark, "zone_5_stage_I_aggrega_step_2")