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
"""
XGB modeling class for Té.

@author: Sathish K Lakshmipathy
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""
module_name = 'XGB'
version_name = '1.2'

import pandas as pd
import numpy as np
import pickle as pkl
import random

import xgboost as xgb


from base_pd.base_modeling_pd import PdBaseModeling as BaseModeling

class XGB(BaseModeling):
    """
    This is a class for XGBoost modeling.

    Attributes:
        rack_order (int): the order of the rack (i.e. 8).
        data_plus_meta (dict): dictionary contining data and meta data.
        racks_map (dict): dictionary of all racks and their orders.
    """

    def __init__(self, rack_order, data_plus_meta, racks_map):
        """
        Constructor for XGBoost models.

        Parameters:
            rack_order (int): the order of the rack (i.e. 8).
            data_plus_meta (dict): dictionary contining data and meta data.
            racks_map (dict): dictionary of all racks and their orders.

        Returns:
            none
        """

        super(XGB, self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)


    def fit(self, X_train, y_train, model_params):
        """
        Runs the XGBoost model with the given parameters.

        Parameters:
            X_train (dataframe): features for training.
            y_train (str): target column for training.
            model_params (dict): dictionary containing hyperparameter names and values.

        Returns:
            xgb_clf: the booster object
        """
        np.random.seed(42)
        # if no custom objective functions are passed in, add default function for classification
        if "objective" not in model_params.keys():
            model_params["objective"] = "binary:logistic"
        # suppress output
        model_params["silent"] = 1
        # random number seed
        model_params["seed"] = 42
        # boosting rounds should be sent as a separate parameters
        num_boost_round = model_params["num_boost_round"]
        # Pass parameters except num_boost_round
        params = {}
        for k, v in model_params.items():
            if k != "num_boost_round":
                params[k] = v
        # Change data frame to DMatrix
        predictors = sorted(X_train.columns.values)
        dtrain = xgb.DMatrix(data=X_train.values, label=y_train.values)
        xgb_clf = xgb.train(params=params, num_boost_round=num_boost_round, dtrain=dtrain)

        return xgb_clf

    def predict(self, model, X_df, y_df):

        """
        Makes predictions and stores both prob and prediction in a dataframe.

        Parameters:
            model (model object): fit model.
            X_df (dataframe): features for test.
            y_df (str): target column for test. target is just to add labels, it's not used in scoring.

        Returns:
            pred_df: dataframe containing customers id, actual labels, predicted probability, and predicted class.
        """
        pred_df = pd.DataFrame(index=X_df.index)

        pred_df["label"] = y_df

        dvalidate = xgb.DMatrix(data=X_df.values, label=y_df.values)

        pred_df["probability"] = model.predict(dvalidate, validate_features=True)

        pred_df["prediction"] = np.where(pred_df["probability"] >= 0.5, 1, 0)

        return pred_df
