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
Zone 5, step 9,  splitting feature statistic

Difference in descriptive statistics between train and test							
							
	Delta minimum	Delta maximum	Delta mean	Delta standard deviation	Delta median	Kl divergence	KS test
Column 1							
Column 2							
Column 3							
Column 4							
Column 5							
Column 6							
Column 7							

High level pseudo code:
1. load split_train
   load split_pred
2. calc min, max, sttdev, mean for train : get_descriptive_statistics_for_columns
 calc min, max, sttdev, mean for test : get_descriptive_statistics_for_columns
3.join dfs :
join select(*(col(x).alias(x + '_pre') for x in preprocessing_cols_stats_df.columns))
joined_df = preprocessing_cols_stats_df_re.join(imputed_cols_stats_df, preprocessing_cols_stats_df_re.column_pre == imputed_cols_stats_df.column)

4. diffrence delts between columns : get_delta_columns_df(joined_df):
5. Caclulate KS divergence and kl divergence
6. K test
7. join delta_df, k_test, kl_divergence
"""

import sys
sys.path.append("/home/boldyrek/mysoft/te/te_reporting/")
from col_stats import *
import config as cfg
from helper_functions import *
from scipy import stats
from helper_functions import start_spark_session, get_imputed_df, suffix_and_join_dfs, write_to_excel, load_df


spark = start_spark_session()

# step 1 loading dfs
imputed_train = load_df( cfg.SPLIT_TRAIN_PATH )
imputed_predict = load_df( cfg.SPLIT_PRED_PATH )

#imputed_train = test_df
#imputed_predict = test_df

# step 2 getting descriptive statistics
imputed_train_descriptive_stats = get_df_with_descriptive_stats_for_columns ( spark,  imputed_train )
imputed_predict_descriptive_stats = get_df_with_descriptive_stats_for_columns ( spark, imputed_predict )

#Step 3 join dfs
joined_descriptive_stats= suffix_and_join_dfs(
    imputed_train_descriptive_stats, imputed_predict_descriptive_stats, 'column_name' )

delta_df = get_delta_descriptive_stats_df(joined_descriptive_stats, '2' )

# Step 5 ks stats
def get_df_with_ks_stats( imputed_train, imputed_predict ):
    columns = imputed_train.schema.names
    col_ks = []
    for col in columns:
        imputed_train_col = imputed_train.select(col).toPandas()[col].tolist()
        imputed_predict_col = imputed_predict.select(col).toPandas()[col].tolist()
        try:
            ks = stats.ks_2samp(imputed_train_col, imputed_predict_col)
            p_value = str(round(ks[0], 2))
            
            kd = str(round(ks[1], 2))

        except Exception as e:
            #print('col ',col ,e)
            p_value = ''
            kd = ''      
        col_ks.append((col,p_value, kd))   
    ks_stats_df = spark.createDataFrame(col_ks, ['column_name_ks', 'p_value', 'kd'])
    return ks_stats_df

ks_stats_df = get_df_with_ks_stats (imputed_train, imputed_predict )
#ks_stats_df.show()
# Step 7 Join 
delta_df = delta_df.join(ks_stats_df, col('column_name') == col('column_name_ks')).\
    select('column_name','delta_min','delta_max','delta_mean','delta_stddev','delta_median',
           'p_value','kd')
write_to_excel( delta_df, 'zone_5_split_fea_stats_ste_9')
