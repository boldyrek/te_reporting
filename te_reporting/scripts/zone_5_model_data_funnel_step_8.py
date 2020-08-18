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
Zone 5, step 8, model data funnel

Output table:
Table 1	
  	Value
Number of customers	
Number of rows	
Number of customers with a positive outcome	
"""
import sys
sys.path.append("/home/boldyrek/mysoft/te/te_reporting/")
from helper_functions import get_imputed_df, start_spark_session, write_to_excel
import config as cfg

def get_num_party_ids_with_positive_outcome( imputed_df ):
    
    target_column = cfg.TE_TARGET_COL
    where_clause = target_column + ' == 1'
    num_party_ids_with_positive_outcome =\
            imputed_df.where(where_clause).select('party_id').distinct().count()
    
    return num_party_ids_with_positive_outcome


def main():
    
    spark = start_spark_session()
    imputed_df = get_imputed_df( cfg.IMPUTATION_TRAIN_PATH, cfg.IMPUTATION_PREDICT_PATH )

    num_party_ids = imputed_df.select("party_id").distinct().count()

    num_rows = imputed_df.count()
    
    num_party_ids_with_positive_outcome = get_num_party_ids_with_positive_outcome( imputed_df )
 
    result = spark.createDataFrame([['Number of customers',num_party_ids],
                      ['Number of rows', num_rows],
                      ['Number of customers with a positive outcome',
                       num_party_ids_with_positive_outcome]],
                      ['', 'Value']
                      )
    return result
    
result = main()
write_to_excel( result , "zone_5_model_data_funnel_step_8")

