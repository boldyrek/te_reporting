#!/home/dev/.conda/envs/py365/bin/python3.6
"""
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation

Version: ttp_2_5
Author: Sathish K Lakshmipathy
Purpose: config file - new purchase
"""
import numpy as np
##########################################################################################################
## LOCATION
##########################################################################################################
seed = 47

DATA_PATH = "/jupyter-notebooks/sathish/tf_tta_data/"

SAVE_DATA_PATH = "/jupyter-notebooks/sathish/tf_tta_data/atlas/"

MODEL_PATH = "/home/dev/tf/tf_model_data/"

LOG_PATH = "/jupyter-notebooks/sathish/tf_tta_data/atlas_log/"

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

SAMPLED_CV_PATH = SAVE_DATA_PATH + "sampled_cv_"
SAMPLED_VAL_PATH = SAVE_DATA_PATH + "sampled_val_"
SAMPLED_FULL_PATH = SAVE_DATA_PATH + "sampled_full_"

FINAL_TRAIN_PATH = SAVE_DATA_PATH + "final_train_"
FINAL_VALIDATE_PATH = SAVE_DATA_PATH + "final_validate_"
FINAL_PREDICT_PATH = SAVE_DATA_PATH + "final_predict_"

MODEL_PATH = SAVE_DATA_PATH + "model_"
SCORE_PATH = SAVE_DATA_PATH + 'pred_scores_'

##########################################################################################################
## OPERATION VARIABLES
##########################################################################################################

# Give the tab separated filename
FEED_DATE = None
#TRAIN_FILE_NAME = "training_data/training_data_2019-07-30.tsv"
DATA_FILE_DICT = {1:'prediction_data_2019-08-13_1.tsv'
                 ,2:'prediction_data_2019-08-13_2.tsv'}

# Train until an earlier date - format: 'YYYY-MM-DD' or False to use all data
END_DATE = '2019-08-01'

# DEFAULT ALGO NAME = XGB
# Change to desired algorithm
ML_ALGO = 'xgb'

# train or predict
# test hyperparameters - If true, the model would be built with data
# until period 2 and tested on period 1 data
# If not, the model would be built with data until period 1
# train, predict, both
STEP = 'train'

TRAIN_PARAMETERS = {'cv': {'if_cv': True,
                           'cv_folds': 3,
                           'cv_metric': 'roc_auc',
                           'time_series_cv': False
                          },
                   'validate': {'if_validate': True,
                               'validate_folds': 3,
                               'validate_metric': 'mean',
                               'time_series_validate': True,
                               'total_val_ctu': 24
                               }
                   }

# Sampling on imbalanced dataset
#IS_TE_SAMPLED = True

# True to optimize TE parameters
OPTIMIZE_TE = False

# True to optimize ML algorithm
OPTIMIZE_ML = False

# Use an existing to score, if required (Optional)
USE_CUSTOM_MODEL = False

CUSTOM_MODEL_NAME = None

##########################################################################################################
## USER DEFINED PIPELINE VARIABLES
##########################################################################################################

# Example: 2quarters, 3months, 8weeks
TE_WINDOW = "3month"

# Options: day, week, month, quarter, year
# CTU_UNIT should be less than or equal to TE_WINDOW
CTU_UNIT = "quarter"

REFERENCE_EVENT = ["purchase_new", "purchase_used"]

# For feature Selection
FEATURE_SELECTION = {'correlation_threshold':0.85,
                     'sparsity_limit':0.9
                     }

LABEL_LAG = 1

LAG_CTU = 3

# Optimizable features
ROLLING_AGG_CTU = 3

N_YEARS = 7

ROLLING_AGG_FUNC_LIST = ['sum']

TE_TARGET_COL = "te_" + TE_WINDOW

PREDICTION_WINDOW = {"td_last_ref_event_min": 365, "td_last_ref_event_max": None}

PREDICTION_WINDOW_TD_COL = "td_last_ref_event"

################################################################################################
## PARAMETERS
################################################################################################


def te_parameters(optimize=False):
    """
    Has the te parameters - period of aggregation and downsample periods
    """
    if optimize:

        te_param_dict = {
            "agg_window": [6, 9, 12] #in months
            # ,'downsample_percent':[0.85, 0.9, 0.92]
            # ,'upsample_percent':[0.18, 0.25, 0.30]
        }
    else:
        te_param_dict = {"agg_window": 12,
                        "upsample_percent": 0.25,
                        "downsample_percent": 0.9}

    return te_param_dict

################################################################################################
## INITIAL ML PARAMETERS
################################################################################################

def hyperparameters(optimize=False):
    """
    Has the set of model parameters to tune for XGBoost
    Stored as a dictionary
    """
    if optimize:

        param_dict = {
            "num_boost_round": range(400, 800, 50),
            "eta": [0.03, 0.05, 0.07, 0.09, 1.0],
            "max_depth": [3, 4, 5, 6],
            "min_child_weight" : list([i/ float(100) for i in range(0, 105, 10)]),
            "subsample": [i / float(10) for i in range(5, 10, 1)],
            "colsample_bytree": [i / float(10) for i in range(5, 10, 1)],
        }

    else:

        param_dict = {"num_boost_round": 600, "eta": 0.03, "max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight" : 1}

    return param_dict

################################################################################################
## MISC
################################################################################################

TE_CONV_DICT = {"day":{"month": 1/30.45, "year": 1/365,"week": 1/7,"quarter": 1/90, "day": 1},
            "week":{"month": 1/4, "year": 1/52,"day": 7,"quarter": 1/12, "week": 1},
            "month":{"week": 4, "year": 1/12,"day": 30.45,"quarter": 1/3, "month": 1},
            "quarter":{"month": 3, "year": 1/4,"day": 91, "week": 12, "quarter": 1},
             }

TE_WINDOW_UNITS_DICT = {
            "month":12,
            "quarter":4
             }
