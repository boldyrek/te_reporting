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
Zone 5, step 7, target defintion 
Proportion of customers with at least one positive target
"""
import sys
sys.path.append("/home/boldyrek/mysoft/te/te_reporting/")
from helper_functions import start_spark_session, get_imputed_df, suffix_and_join_dfs, write_to_excel
from col_stats import *
from pyspark.sql.functions import col, round,  countDistinct, asc
import  pyspark.sql.functions
from pyspark.sql.types import IntegerType
import config as cfg

spark = start_spark_session()

def main(imputed_df, spark):
    
    per_ctu_count_partyids_with_positive_interactions = imputed_df.where("te_2month = 1").\
        groupBy('CTU').agg(countDistinct('party_id'))

    per_ctu_count_partyids_with_all_interactions = imputed_df.\
        groupBy('CTU').agg(countDistinct('party_id'))

    
    joined_df = suffix_and_join_dfs( per_ctu_count_partyids_with_positive_interactions,
                                   per_ctu_count_partyids_with_all_interactions, 'CTU')

    proportion_of_positives = calculate_proportion_of_partyids_of_positives_per_ctu (joined_df )
    ctus_all = list(imputed_df.select('CTU').distinct().collect())
    proportion_of_positives_full_table = add_zero_proportions_to_empty_ctus (proportion_of_positives , ctus_all, spark)
    
    return proportion_of_positives_full_table
       
    
def calculate_proportion_of_partyids_of_positives_per_ctu( joined_df ):
        proportion_of_positives = joined_df.withColumn("proportion_with_positive_target", 
                                     round(col('count(party_id)')/ 
                                           col('count(party_id)_2'),2))\
                                    .select('CTU',"proportion_with_positive_target")
        
        return proportion_of_positives
    
def add_zero_proportions_to_empty_ctus (proportion_of_positives , ctus_all, spark) :
   
    for ctu in sorted(ctus_all):
        ctu_query = "ctu == {0}".format(ctu[0])
        ctu_value  = proportion_of_positives.where(ctu_query).select('CTU').collect()
        if len(ctu_value) ==0:
            new_df = spark.createDataFrame([[ctu[0],0]] , ['CTU','proportion_with_positive_target'])
            proportion_of_positives = proportion_of_positives.union(new_df)
    return proportion_of_positives



imputed_df = get_imputed_df(cfg.IMPUTATION_TRAIN_PATH, cfg.IMPUTATION_PREDICT_PATH )
proportion_of_positives  = main(imputed_df, spark)
proportion_of_positives = proportion_of_positives.withColumn('CTU', 
                                                             proportion_of_positives['CTU']\
                                                             .cast(IntegerType()))
proportion_of_positives = proportion_of_positives.orderBy(asc('CTU'))
write_to_excel(proportion_of_positives, "zone_5_ta_def_7_prop_per_ctu.py")

