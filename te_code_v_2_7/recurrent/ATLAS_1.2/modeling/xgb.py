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

    def __init__(self, rack_order, data_plus_meta, racks_map):
        """
        Constructor for ARIMAX modeling module for BOC
        :param: object_splited: the enriched object.
    			config: the JSON configuration namespace.
        :return: none
        :raises: none
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
