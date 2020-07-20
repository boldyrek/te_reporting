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
seed = 3
# DATA_PATH = "/efs/cerebri/model_data/"
DATA_PATH = "/efs/cerebri/model_data/"

SAVE_DATA_PATH = "/efs/cerebri/model_results/pouya_test/"

MODEL_PATH  = "/efs/cerebri/model_results/pouya_test/"

LOG_PATH = "/efs/cerebri/model_results/pouya_test/"

SAVE_CONFIG_PATH = "/efs/cerebri/model_results/pouya_test/config/"

TE_SPLITTING_PATH = "/efs/cerebri/model_results/results_2_12b_1/enriched_train_2020_02_22_"

TE_FEATURE_SELECTION_TRAIN_PATH = "/efs/cerebri/model_results/results_2_12b_1/run_all/sampled_2020_02_22_"
TE_FEATURE_SELECTION_VALIDATE_PATH = "/efs/cerebri/model_results/results_2_12b_1/run_all/sampled_val_2020_02_22_"
TE_FEATURE_SELECTION_PREDICT_PATH = "/efs/cerebri/model_results/results_2_12b_1/run_all/split_pred_2020_02_22_"

TE_MODELING_TRAIN_PATH = "/efs/cerebri/model_results/results_2_12b_1/run_all/corrected_MRR/final_train_2020_02_22_"
TE_MODELING_VALIDATE_PATH = "/efs/cerebri/model_results/results_2_12b_1/run_all/corrected_MRR/final_validate_2020_02_22_"
TE_MODELING_PREDICT_PATH = "/efs/cerebri/model_results/results_2_12b_1/run_all/corrected_MRR/final_predict_2020_02_22.pkl"
##########################################################################################################
## INTERMEDIATE DATA PATH
##########################################################################################################
GEN_PREPROCESS_PATH = SAVE_DATA_PATH + "general_preprocessing_"

PREPROCESSED_PATH = SAVE_DATA_PATH + "preprocessing_"

IMPUTATION_TRAIN_PATH = SAVE_DATA_PATH + "imputed_train_"
IMPUTATION_PREDICT_PATH = SAVE_DATA_PATH + "imputed_predict_"

ENRICH_PATH = SAVE_DATA_PATH + "enriched_train_"

PRED_SPLIT_PATH = SAVE_DATA_PATH + "split_pred_"
TRAIN_SPLIT_PATH = SAVE_DATA_PATH + "split_train_"
VALIDATE_SPLIT_PATH = SAVE_DATA_PATH + "split_validate_"

SAMPLED_VAL_PATH = SAVE_DATA_PATH + "sampled_val_"
SAMPLED_TRAIN_PATH = SAVE_DATA_PATH + "sampled_"

FINAL_TRAIN_PATH = SAVE_DATA_PATH + "final_train_"
FINAL_VALIDATE_PATH = SAVE_DATA_PATH + "final_validate_"
FINAL_PREDICT_PATH = SAVE_DATA_PATH + "final_predict_"

MODEL_PATH = SAVE_DATA_PATH + "model_"
SCORE_PATH = SAVE_DATA_PATH + "pred_scores_"
PLOTS_PATH = SAVE_DATA_PATH + "plot_"
##########################################################################################################
## OPERATION VARIABLES
##########################################################################################################
DATA_FILE_DICT = {1:"flb_1_12B_V_1.tsv",
                  2:"flb_1_12B_V_2.tsv",
                  3:"flb_1_12B_V_3.tsv",
                  4:"flb_1_12B_V_4.tsv",
                  5:"flb_1_12B_V_5.tsv",
                  6:"flb_1_12B_V_6.tsv",
                  7:"flb_1_12B_V_7.tsv",
                  8:"flb_1_12B_V_8.tsv",
                  9:"flb_1_12B_V_9.tsv",
                  10:"flb_1_12B_V_10.tsv",
                  11:"flb_1_12B_V_11.tsv",
                  12:"flb_1_12B_V_12.tsv",
                  13:"flb_1_12B_V_13.tsv",
                  14:"flb_1_12B_V_14.tsv",
                  15:"flb_1_12B_V_15.tsv",
                  16:"flb_1_12B_V_16.tsv",
                  17:"flb_1_12B_V_17.tsv",
                  18:"flb_1_12B_V_18.tsv",
                  19:"flb_1_12B_V_19.tsv",
                  20:"flb_1_12B_V_20.tsv",
                  21:"flb_1_12B_V_21.tsv",
                  22:"flb_1_12B_V_22.tsv",
                  23:"flb_1_12B_V_23.tsv",
                  24:"flb_1_12B_V_24.tsv",
                  25:"flb_1_12B_V_25.tsv",
                  26:"flb_1_12B_V_26.tsv"}
# Mention the number of training files from the list of files above
# Due the memory restrictions of VM/pandas only part of the data is used for optimization
NO_TRAIN_FILE = 26
# Train until an earlier date - format: 'YYYY-MM-DD' or False to use all data
# have a 14-day CTU 0:
TRAIN_END_DATE = '2019-07-09'

TRAIN_START_DATE = '2018-07-01'
# This flag incidates if can use the reference events occuring after the TRAIN_END_DATE
# This is used to include labels in the train data and tested on out of sample test customers
INCLUDE_TEST_LABELS = 'True'
# train or predict or both
# train, predict, both
STEP = 'both'
# Example: 2quarters, 3months, 8weeks
TE_WINDOW = "3month"
# There should be a number followed by one of the units in TE_CTU_UNITS
# Example: 2day, 3week, 1month, 4quarter, 1year
CTU_UNIT = "2week"
#day to months
DAY_TO_MONTH = 30.45

REFERENCE_EVENT = "CAI_TELCO_EVENT_TYPE_CHURN"

N_YEARS = 3

TE_TARGET_COL = "te_" + TE_WINDOW
# keep churn
KEEP_TERMINAL_CTU = True
#start point to run pipeline
RUN_ALL_RACKS = True
RUNNING_RACK = "Modeling" # ["Modeling", "FeatureSelection", "Splitting"]
#run QA tests
QA_TEST = False
# THERESHOLD FOR OUTLIERS
OUTLIER_THRESHOLD = 10
#model based on mrr
PRICE_MODEL = False
PRICE_THRESHOLD = 200
##if the proportion of test need to be corrected
CORRECT_PROPORTION = False
ACT_DIST = 0.038## correct proportion of churners
#is there validation model?
VALIDATION_MODEL = False
#reduce size of cv data to avoid memory errors
REDUCED_CV = False
#drop correlated features in orchestrator
DROP_FEATURE = False
#remove low variance features after enrichment
LOW_VAR_ENRICHED = False
#transform fetures
TRANSFORM = False
#threshold to remove short journeys
SHORT_TENURE = 90
#test on in_time sample for out of sample customers
IN_TIME_TEST = True

# List the handpicked features for experiments
EXPERIMENT_SELECTED_FEATURES = False
# If True for EXPERIMENT_SELECTED_FEATURES, add the list of handpicked features below
SELECTED_FEATURE_LIST = []
# Columns groups to change
CHANGE_COL_GROUPS = ['IMPUTE_ZERO_COLS', 'CUMSUM_COLS', 'FACTOR_COLS', 'FFILL_COLS',\
'FLB_TD_COLS', 'EVENT_COLS', 'TD_COLS', 'CUMMEDIAN_COLS', 'CUMMAX_COLS',\
'REF_AGG_COLS', 'AGGREGATION_COLS', 'POSITIVE_INTERACTIONS', 'OUTBOUND_INTERACTIONS',\
'CATEGORICAL_COLS', 'NEGATIVE_INTERACTIONS', 'INBOUND_INTERACTIONS']
################################################################################################
## TE TIME UNITS
################################################################################################
TE_CONV_DICT = {"day":{"month": 1/30.45, "year": 1/365,"week": 1/7,"quarter": 1/90, "day": 1},
            "week":{"month": 1/4, "year": 1/52,"day": 7,"quarter": 1/12, "week": 1},
            "month":{"week": 4, "year": 1/12,"day": 30.45,"quarter": 1/3, "month": 1},
            "quarter":{"month": 3, "year": 1/4,"day": 91, "week": 12, "quarter": 1},
             }

TE_CTU_UNITS = ["day", "week", "month", "quarter", "year"]

TE_WINDOW_UNITS_DICT = {
            "month":12,
            "quarter":4,
            "week":52
             }
################################################################################################
## List of columns to be removed/transformed in the final model
################################################################################################
CORRELATED_FEATURES = [
'td_last_CAI_TELCO_FL_IN_NETWORK_SUM'
]

TRANSFORMED_FEATURES = [
'CAI_TELCO_DURATION_SEC_SUM'
]

