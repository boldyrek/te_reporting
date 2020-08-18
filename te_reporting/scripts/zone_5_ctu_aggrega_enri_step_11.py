import sys
sys.path.append("/home/boldyrek/mysoft/te/te_reporting/")
from helper_functions import get_imputed_df, start_spark_session, write_to_excel
import config as cfg

"""
Te version 2.7
Zone 5, step 11, Ctu aggregation enrichment

Task description	Automation			
After CTU aggregation (get one row per acct-CTU pair) on entire dataset and column enrichment (add rolling, trend, lag, etc columns), create table with row-wise listing of:				
Column name	Table1			
Min aggregation look-back window	user has to define			
Max aggregation look-back window	user has to define			
Look-back window used	Table1			
Is column aggregated (y/n)	Table1			
Feature aggregation logic	Table1			
Is column enriched (y/n)	Table1			
Feature enrichment logic	Table1			
Column descriptive statistics:	Table1			
Minimum				
Maximum				
Mean				
Standard deviation				
Median				
				
Rationale: Evaluate reasonable look-back window options, get distribution descriptive statistics on baseline CTU level data, this is the first time data is persited at the CTU level	

PSEUDO CODE:
1. Get TE_WINDOW from /home/boldyrek/mysoft/te/te_code_v_2_7/terminal/ATLAS_1.2/config
get_module_from_path( module_name , module_path ): 15 min
2. if colum is aggeragted or not to be speicifed in config file:
    cfg.AGGREGATIONS
    col_and_its_aggreagation =[]
    for list_of_aggregations in cfg.AGGREGATIONS:            
        for col in cols: 
            if col in list:
                col_and_its_aggregation.append( col, list_of_aggregations )
            else: 
                col_and_its_aggregation.append( col, 'there is no aggregation' )
            
    AGGREGATIONS = ['CUMSUM_COLS','CUMMAX_COLS']
    Feature aggregation logic
30 mins
3. Feature enrichment logic: load_pipeline_config.py
60 mins
4. Feature descriptive statistics 
    def get_df_with_descriptive_stats_for_columns( spark , df):
50 mins

"""
os.path.split('/home/boldyrek/mysoft/te/te_code_v_2_7/terminal/ATLAS_1.2/config/config.py')
#get_module_from_path( module_name , module_path ):
#start_spark_session():
#Efor col in columns:
    
    
    