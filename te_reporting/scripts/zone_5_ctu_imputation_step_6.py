###################################################################################################
# Cerebri AI CONFIDENTIAL
# Copyright (c) 2017-2020 Cerebri AI Inc., All Rights Reserved.
#
# NOTICE: All information contained herein is, and remains the property of Cerebri AI Inc.
# and its subsidiaries, including Cerebri AI Corporation (together “Cerebri AI”).
# The intellectual and technical concepts contained herein are proprietary to Cerebri AI
# and may be covered by U.S., Canadian and Foreign Patents, patents in process, and are
# protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material is strictly
# forbidden unless prior written permission is obtained from Cerebri AI. Access to the
# source code contained herein is hereby forbidden to anyone except current Cerebri AI
# employees or contractors who have executed Confidentiality and Non-disclosure agreements
# explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended publication or
# disclosure of this source code, which includes information that is confidential and/or
# proprietary, and is a trade secret, of Cerebri AI. ANY REPRODUCTION, MODIFICATION,
# DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE OF THIS SOURCE
# CODE WITHOUT THE EXPRESS WRITTEN CONSENT OF CEREBRI AI IS STRICTLY PROHIBITED, AND IN
# VIOLATION OF APPLICABLE LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF
# THIS SOURCE CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR SELL
# ANYTHING THAT IT MAY DESCRIBE, IN WHOLE OR IN PART.
###################################################################################################
#!/home/dev/.conda/envs/py365/bin/python3.6
##########################################################################################################
## LOCATION
##########################################################################################################

"""
Te version 2.7
Zone 5, step 6: CTU imputations

After imputing rows for missing CTUs, create table with row-wise listing:		
Column name		Table 1
Imputation approach (min, max, median, cumsum, zero, FFill, etc.)		Table 1
Proportion of accounts that have more than:		Table 1
99% missing		
75% missing		
50% missing		
25% missing		
		
Differences in descriptive statistics between steps 6 and 3		Table 2
Delta min		
Delta max		
Delta mean		
Delta std		
Delta median	

Input files: 
imputed_train_ and  preprocessing_
		
        
TO DO:
Sort by delta max desc
"""
import sys
sys.path.append("/home/boldyrek/mysoft/te/te_reporting/")
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from helper_functions import get_imputed_df, start_spark_session, load_df, write_to_excel, get_module_from_path
import config as cfg


def calc_column_func(df, column, func):

    """
    for a column, calculate a statistical value
    """

    return df.agg({column : func}).collect()[0][0]


def get_descriptive_statistics_for_columns(df):

    """
    Get the columns names and for every column create a tupel col, maximum , minumium to make it sutable to create a datafrma out out tuples
    (event1, 3, 1) 
    """

    columns = preprocessing_df.schema.names
    columns_with_stats = []  # append tuples to a list, later to create a spark df
    for col in columns: # for each column calculate stat values
        maximum = calc_column_func(df, col, 'max')
        minimum = calc_column_func(df, col, 'min')
        mean = calc_column_func(df, col, 'avg')
        columns_with_stats.append((col,maximum, minimum, mean))
    return columns_with_stats 


def drop_garbage_cols(df):
    """
    Drop some of the unnesessary columns
    """
    columns_to_drop = ['level_0', 'index', 'Unnamed: 0', '_c0']
    df_to_drop = df.select('*')
    df_to_drop = df_to_drop.drop(*columns_to_drop)
    
    return df_to_drop


def get_delta_columns_df(joined_df):
    """
    Substract simmilar summary columns (like min, max, mean .. ) for preprocessing df and imputed df 
    """
    joined_df_min = joined_df.withColumn("delta_min", col("min_pre") - col("min"))
    joined_df_min_max = joined_df_min.withColumn("delta_max", col("max_pre") - col("max"))
    joined_df_min_max_mean = joined_df_min_max.withColumn("delta_mean", col("mean_pre") - col("mean"))
    
    return joined_df_min_max_mean
    
 
"""
**** MAIN *****
"""


spark = start_spark_session()


preprocessing_df = load_df(cfg.PREPROCESS_PATH)
preprocessing_columns_with_stats = get_descriptive_statistics_for_columns(preprocessing_df)
preprocessing_cols_stats_df = spark.createDataFrame( preprocessing_columns_with_stats, ['column','max','min','mean'] )

imputed_df = get_imputed_df( cfg.IMPUTATION_TRAIN_PATH, cfg.IMPUTATION_PREDICT_PATH )
imputed_columns_with_stats = get_descriptive_statistics_for_columns(imputed_df)
imputed_cols_stats_df = spark.createDataFrame( imputed_columns_with_stats, ['column','max','min','mean'] )


preprocessing_cols_stats_df_re = preprocessing_cols_stats_df.\
select(*(col(x).alias(x + '_pre') for x in preprocessing_cols_stats_df.columns))
joined_df = preprocessing_cols_stats_df_re.join(imputed_cols_stats_df, preprocessing_cols_stats_df_re.column_pre == imputed_cols_stats_df.column)

delta_columns_df = get_delta_columns_df(joined_df)
elta_columns_df = delta_columns_df.select('column','delta_min', 'delta_max', 'delta_mean')
write_to_excel(delta_columns_df, "zone_5_ctu_imputation_step_6")