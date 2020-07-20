#!/home/dev/.conda/envs/py365/bin/python3.6
"""
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation

Version: ttp_2_5
Author: Sathish K Lakshmipathy
Purpose: config file - new purchase
"""
##########################################################################################################
## LOCATION
##########################################################################################################
seed = 3

DATA_PATH = "/home/boldyrek/mysoft/te/data/"

SAVE_DATA_PATH = "/home/boldyrek/mysoft/te/"

MODEL_PATH = "/home/boldyrek/mysoft/te/"

LOG_PATH = "/home/boldyrek/mysoft/te/"

SAVE_CONFIG_PATH = "/home/boldyrek/mysoft/te/"


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
TEST_SCORE_PATH = SAVE_DATA_PATH + "test_scores_"

##########################################################################################################
## OPERATION VARIABLES
##########################################################################################################

DATA_FILE_DICT = {1:'synthetic_data_v1.csv'}

# Mention the number of training files from the list of files above
# Due the memory restrictions of VM/pandas only part of the data is used for optimization
NO_TRAIN_FILE = 2

# Train until an earlier date - format: 'YYYY-MM-DD' or False to use all data
TRAIN_END_DATE = '2020-04-30'

TRAIN_START_DATE = '2019-04-01'
# This flag incidates if can use the reference events occuring after the TRAIN_END_DATE
# This is used to include labels in the train data and tested on out of sample test customers
INCLUDE_TEST_LABELS = 'False'

# Example: 2quarters, 3months, 8weeks
# test hyperparameters - If true, the model would be built with data
# until period 2 and tested on period 1 data
# If not, the model would be built with data until period 1
# train, predict, both
STEP = 'both'

# Example: 2quarters, 3months, 8weeks
TE_WINDOW = "2month"

# Options: day, week, month, quarter, year
# CTU_UNIT should be less than or equal to TE_WINDOW
CTU_UNIT = "1month"

REFERENCE_EVENT = ["cai_ins_grs_vuc"]

N_YEARS = 7

TE_TARGET_COL = "te_" + TE_WINDOW

#start point to run pipeline
RUN_ALL_RACKS = True
RUNNING_RACK = "Modeling" # ["Modeling", "FeatureSelection", "Splitting"]

TE_SPLITTING_PATH = "/efs/cerebri/model_results/results_2_12b_1/enriched_train_2020_02_22_"

TE_FEATURE_SELECTION_TRAIN_PATH = "/jupyter-notebooks/sathish/tf_tta_data/te_test/sampled_2020_02_22_"
TE_FEATURE_SELECTION_VALIDATE_PATH = "/jupyter-notebooks/sathish/tf_tta_data/te_test/sampled_val_2020_02_22_"
TE_FEATURE_SELECTION_PREDICT_PATH = "/jupyter-notebooks/sathish/tf_tta_data/te_test/split_pred_2020_02_22_"

TE_MODELING_TRAIN_PATH = "/jupyter-notebooks/sathish/tf_tta_data/te_test/final_train_2020_02_22_"
TE_MODELING_VALIDATE_PATH = "/jupyter-notebooks/sathish/tf_tta_data/te_test/final_validate_2020_02_22_"
TE_MODELING_PREDICT_PATH = "/jupyter-notebooks/sathish/tf_tta_data/te_test/final_predict_2020_02_22.pkl"

# Is there a need to build validation models (compare fullpipelines)
VALIDATION_MODEL = False

# Reduce CV proportion
# this is because the sampling portion would get bigger and the cv size needs to be reduced
REDUCED_CV = False

# Run QA test?
QA_TEST = False

#test on in_time sample for out of sample customers
IN_TIME_TEST = True

#drop correlated features in orchestrator
DROP_FEATURE = False

#remove low variance features after enrichment
LOW_VAR_ENRICHED = False

#Log transformation for specific columns
# IF true make sure that the columns in TRANSFORMED_FEATURES are fine
TRANSFORM = False

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
## MISC
################################################################################################

TE_CONV_DICT = {"day":{"month": 1/30.45, "year": 1/365,"week": 1/7,"quarter": 1/90, "day": 1},
            "week":{"month": 1/4, "year": 1/52,"day": 7,"quarter": 1/12, "week": 1},
            "month":{"week": 4, "year": 1/12,"day": 30.45,"quarter": 1/3, "month": 1},
            "quarter":{"month": 3, "year": 1/4,"day": 91, "week": 12, "quarter": 1},
             }

TE_CTU_UNITS = ["day", "week", "month", "quarter", "year"]

TE_WINDOW_UNITS_DICT = {
            "month":12,
            "quarter":4
             }
