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
Zone 5, step 10, splitting data leakage

Confirm that accounts are not shared between train and test

Output table:
Value
Number of train ids	
Number of test ids	
Number of common between train and test	

Pseudo-code
1. load
    imputed_train
    imputed_predict
2. calculate number of
    party_id in imputed_train
    party_ids in predict train
3. Calculate common party ids

"""

import sys
sys.path.append("/home/boldyrek/mysoft/te/te_reporting/")
from helper_functions import get_imputed_df, start_spark_session, load_df
from col_stats import *
import config as cfg
from helper_functions import *

split_train = load_df( cfg.SPLIT_TRAIN_PATH )
split_predict= load_df( cfg.SPLIT_PRED_PATH )

num_of_train_ids = split_train.select('party_id').distinct().count()

num_of_test_ids = split_predict.select('party_id').distinct().count()

num_of_common_between_train_and_test = split_train.select('party_id').distinct().join(
    split_predict.select('party_id').distinct(),\
    ['party_id'], how='inner').select('party_id').count()
    
output_df = spark.createDataFrame([[num_of_train_ids, 
                        num_of_test_ids,
                        num_of_common_between_train_and_test  ]],
                     ["num_of_train_ids", 
                        "num_of_test_ids",
                        "num_of_common_between_train_and_test"]
                     )
                     
write_to_excel(output_df, "zone_5_spli_data_leakage_st_10")
