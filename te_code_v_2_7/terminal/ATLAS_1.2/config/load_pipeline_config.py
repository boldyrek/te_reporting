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
from general_preprocessing.general_preprocessing_churn import GENERALPREPROCESSING

from preprocessing.preprocessing import PREPROCESSING

from imputation.ctu_imputation_churn import CTUIMPUTATION

from enrich_data.time_series_enrichment import TSENRICHDATA

from splitting.customer_ctu_splitting_churn import CTUSPLITTING

from sampling.sampling_high_imbalance import HIGHIMBALANCESAMPLING

from feature_selection.corr_feature_select import CORR_FILT
from feature_selection.boruta_feature_select import BORUTA_PY
from feature_selection.boostaroota_feature_select import BOOSTAROOTA_PY
from feature_selection.RFE import RFE_TE

from modeling.xgb import XGB
from modeling.lgbm import LGBM

from general_preprocessing.attribute_general_preprocessing import GeneralPreprocessAttribute
from preprocessing.attribute_preprocessing import PreprocessingAttribute
from imputation.attribute_imputation import ImputationAttribute
from enrich_data.attribute_enrich_data import EnrichDataAttribute
from splitting.attribute_splitting import SplitAttribute
from sampling.attribute_sampling import SamplingAttribute
from feature_selection.attribute_feature_select import FeatureSelectionAttribute

from modeling.attribute_modeling import ModelAttribute

import util.ml_utils as ml_utils


def load_configs():
    tmap = {"GeneralPreprocessing":1, "Preprocessing":2, "Imputing":3,"Enriching":4,"Splitting":5, "Sampling":6, "FeatureSelection":7, "Modeling":8}

    TRAIN_PARAMETERS = {'cv': {'if_cv': True,
                               'cv_folds': 3,
                               'cv_metric': 'lift_statistic',
                               'cv_aggregation': 'mean',
                               'time_series_cv': False
                              },
                       'validate': {'validate_folds': 0,
                                   'validate_metric': 'lift_statistic',
                                   'validate_aggregation':'mean',
                                   'time_series_validate': True,
                                   'total_val_ctu': 50
                                   }
                       }

    pipeline_best = {
        1: {
        "module": GENERALPREPROCESSING,
        "attribute": GeneralPreprocessAttribute,
        "agg_window": 3,
        "agg_unit": "month"
    },

        2: {
        "module": PREPROCESSING,
        "attribute":PreprocessingAttribute,
    },

        3: {
        "module": CTUIMPUTATION,
        "attribute":ImputationAttribute
        },

        4: {
        "module": TSENRICHDATA,
        "seed": 3,
        "attribute": EnrichDataAttribute,
        "agg_window": 3,
        "agg_unit": "month",
        "label_lag": 1,
        "lag_ctu": 2,
        "rolling_agg_ctu": 6,
        "rolling_agg_func_list": ['std','skew', 'trend', 'mean'],
        "cyclic_period":["month", "quarter","week"]
    },
        5: {
        "module": CTUSPLITTING,
        "seed": 3,
        "test_ctu": 0,
        "train_pct": 0.60,
        "train_val_pct": 1.0,
        "low_var_split_threshold": 0.95,
        "attribute":SplitAttribute,
        "if_cv":TRAIN_PARAMETERS['cv']['if_cv'],
        "folds_cv": TRAIN_PARAMETERS['cv']['cv_folds'],
        "folds_validation": TRAIN_PARAMETERS['validate']['validate_folds']
    },
        6: {
        "module": HIGHIMBALANCESAMPLING,
        "seed": 3,
        "attribute":SamplingAttribute,
        "upsample_percent": 0.25,
        "downsample_percent": 0.95,
        "if_cv":TRAIN_PARAMETERS['cv']['if_cv'],
        "folds_cv": TRAIN_PARAMETERS['cv']['cv_folds'],
        "folds_validation": TRAIN_PARAMETERS['validate']['validate_folds'],
        "total_val_ctu":TRAIN_PARAMETERS['validate']['total_val_ctu']
    },
        7: {
            "module": CORR_FILT,
            "seed": 3,
            "attribute":FeatureSelectionAttribute,
            'correlation_threshold': 1,
            'low_var_threshold': 1,
            'low_entropy_threshold': 0.2,
            'sparsity_limit': 1,
            "folds_cv": TRAIN_PARAMETERS['cv']['cv_folds'],
            "folds_validation": TRAIN_PARAMETERS['validate']['validate_folds'],
    },
#         7: {
#             "module": RFE_TE,
#             "seed": 3,
#             "attribute":FeatureSelectionAttribute,
#             "n_step": 5,
#             "n_features": 100,
#             "folds_cv": TRAIN_PARAMETERS['cv']['cv_folds'],
#             "folds_validation": TRAIN_PARAMETERS['validate']['validate_folds'],
#     },
#            8: {
#         "module": XGB,
#         "seed": 3,
#         "attribute":ModelAttribute,
#         "all_metrics": ml_utils.classifier_metrics,
#         "lift_table":ml_utils.lift_table,
#         "cv_metric": TRAIN_PARAMETERS['cv']['cv_metric'],
#         "cv_aggregation": TRAIN_PARAMETERS['cv']['cv_aggregation'],
#         "validate_metric":TRAIN_PARAMETERS['validate']['validate_metric'],
#         "validate_aggregation": TRAIN_PARAMETERS['validate']['validate_aggregation'],
#         "folds_cv": TRAIN_PARAMETERS['cv']['cv_folds'],
#         "folds_validation": TRAIN_PARAMETERS['validate']['validate_folds'],
#         "if_optimize": False,
#         "if_plot_validate": True,
#         "iteration": 1,
#         "plot_feature_importance":ml_utils.shap_feature_importances,
#         "hyperparameters": {"num_boost_round": 350,
#                         "eta": 0.03,
#                         "max_depth": 5,
#                         "subsample": 0.8,
#                         "colsample_bytree": 0.7
#                            },
#         "hyperparameters_arrays": {
#             "num_boost_round": range(100, 400, 100),
#             "eta": [0.03, 0.05],
#             "max_depth": [4, 5, 6],
#             "subsample": [i / float(10) for i in range(5, 10, 1)],
#             "colsample_bytree": [i / float(10) for i in range(5, 10, 1)]
#             },
#            }

          8: {
            "module": LGBM,
                "seed": 3,
                "attribute":ModelAttribute,
                "all_metrics": ml_utils.classifier_metrics,
                "lift_table":ml_utils.lift_table,
                "cv_metric": TRAIN_PARAMETERS['cv']['cv_metric'],
                "cv_aggregation": TRAIN_PARAMETERS['cv']['cv_aggregation'],
                "validate_metric":TRAIN_PARAMETERS['validate']['validate_metric'],
                "validate_aggregation": TRAIN_PARAMETERS['validate']['validate_aggregation'],
                "folds_cv": TRAIN_PARAMETERS['cv']['cv_folds'],
                "folds_validation": TRAIN_PARAMETERS['validate']['validate_folds'],
                "cv_pct": 0.6,
                "if_optimize": False,
                "if_plot_validate": True,
                "iteration": 2,
                "plot_feature_importance":ml_utils.shap_feature_importances,
                "hyperparameters": {"n_estimators": 250, # number of base learners that will be trained
                                "max_depth": 8, # Maximum tree depth for base learners
                                "learning_rate": 0.03, # Each weight (in all the trees) will be multiplied by this value: smaller = slower
                                "num_leaves": 250, #  Maximum tree leaves for base learners
#                                 "reg_lambda": 0.9, # L2 regularization term on weights: limit how extreme the weights at the leaves can become. encourages the weights to be small: higher values encourage more. better to be large
                                "reg_alpha": 0.1, # L1 regularization term on weights: limit how extreme the weights at the leaves can become. encourages weights to go to 0: higher values encourage more. better to be small
                                "feature_fraction": 0.5,# LightGBM will select n% of features before training each tree. can help with overfitting
                                "bagging_freq": 0, # perform bagging at every k iteration, zero = no bagging
                                "bagging_fraction": 0.8, # randomly select part of data without resampling; bagging_freq should be non zero
                                "bagging_seed": 3, # random seed for bagging
                                "is_unbalance": "False" #  true if training data are unbalanced; while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities
                                   },
                "hyperparameters_arrays": {
                    "n_estimators": [100, 400, 250],
                    "max_depth": [3, 8, 12],
                    "learning_rate": [0.1, 0.001, 0.03],
                    "num_leaves": [250, 150],
#                     "reg_lambda": [0.0, 0.9, 0.7],
                    "reg_alpha": [0.1, 0.07, 0.0],
#                     "feature_fraction": [0.5, 0.8],
#                     "bagging_freq":  [0, 5, 10],
#                     "bagging_fraction": [ 0.5, 0.9],
#                     "is_unbalance": ["False", "True"]
                     }
         },
    }
#   }
    modules_candidates = {
        # 4: [
        #     {   # differenct aggregation windows
        #         "module": TSENRICHDATA,
        #         "seed": 3,
        #         "attribute": EnrichDataAttribute,
        #         "agg_window": 9,
        #         "agg_unit": "month"
        #         "label_lag": 1,
        #         "lag_ctu": 3,
        #         "rolling_agg_ctu": 3,
        #         "rolling_agg_func_list": ['sum']
        #     },
        #     {   # differenct aggregation windows
        #         "module": TSENRICHDATA,
        #         "seed": 3,
        #         "attribute": EnrichDataAttribute,
        #         "agg_window": 6,
        #         "agg_unit": "month"
        #         "label_lag": 1,
        #         "lag_ctu": 3,
        #         "rolling_agg_ctu": 3,
        #         "rolling_agg_func_list": ['sum']
        #     }

        #],
        # 5: [
        #     {
        #         "module": CTUSPLITTING,
        #         "seed": 3,
        #         "attribute":SplitAttribute,
        #         "if_validate": TRAIN_PARAMETERS['validate']['if_validate'],
        #         "if_cv":TRAIN_PARAMETERS['cv']['if_cv'],
        #         "folds_cv": TRAIN_PARAMETERS['cv']['cv_folds'],
        #         "folds_validation": TRAIN_PARAMETERS['validate']['validate_folds']
        #     },
        #
        #],
#         7: [
#              {
#                   "module": BORUTA_PY,
#                   "seed": 3,
#                   "attribute":FeatureSelectionAttribute,
#                   "two_step": False, # False corresponds to original boruta
#                   "z_score_percentile":100, # Can set threshold for confirm/reject based on z score of importances
#                   "max_iter": 50
#              },
#              {
#                   "module": RFE,
#                   "seed": 3,
#                   "attribute":FeatureSelectionAttribute
#                   "n_step": 5,
#                   "n_features":100
#              },
#             {
#                   "module": BOOSTAROOTA_PY,
#                   "seed": 3,
#                   "attribute":FeatureSelectionAttribute,
#                   "boostaroota_metric": TRAIN_PARAMETERS['cv']['cv_metric'], # metric for evaluation
#                   "model_name":'xgb_class', # xgb_class/xgb_reg
#                   "cutoff": 4, # Set the cutoff threshold for the feature importance
#                   "iters": 10, # Number of algorithm iterations
#                   "max_rounds": 100, #The number of times the core BoostARoota algorithm will run. Each round eliminates more and more features
#                   "delta":0.1 # Stopping criteria Number
#              },
#             {
#                  "module": CORR_FILT,
#                  "seed": 3,
#                  "attribute":FeatureSelectionAttribute,
#                  'correlation_threshold':0.95,
#                  'sparsity_limit':0.95
#             }

#         ]
    }
    return pipeline_best, modules_candidates, tmap
