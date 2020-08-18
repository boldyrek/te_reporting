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
Zone 5, step 7, target definition 

Descriptive statistics on the number of positive labels for across all customers

Minimum    Maximum    Mean    Standard deviation    Median        
1    2    1    1.5    1        
"""
import sys
sys.path.append("/home/boldyrek/mysoft/te/te_reporting/")
from helper_functions import start_spark_session, get_imputed_df
from col_stats import *
from pyspark.sql.functions import col
from helper_functions import get_imputed_df, start_spark_session, load_df, write_to_excel 
import config as cfg

spark = start_spark_session()
imputed_df = get_imputed_df( cfg.IMPUTATION_TRAIN_PATH, cfg.IMPUTATION_PREDICT_PATH )

"""
take the targets where target is equal 1, group by party id, count how many 1 targets
"""
imputed_df_count = imputed_df.where("te_2month = 1").groupBy('party_id').agg({'te_2month' : 'count'})
imputed_df_count_te_2month = imputed_df_count.select("count(te_2month)")
minimum = calc_column_min(imputed_df_count_te_2month)
maximum = calc_column_max(imputed_df_count_te_2month)
mean = calc_column_avg(imputed_df_count_te_2month)
stdev = calc_column_stddev(imputed_df_count_te_2month)
median = calc_column_median(imputed_df_count_te_2month)
positive_label_stats_across_customers_df = spark.createDataFrame([[minimum, maximum, mean, stdev, median]],\
                      ['minimum', 'maximum','mean','stdev','median'])
write_to_excel( positive_label_stats_across_customers_df , "zone_5_target_definitio_step_7")

