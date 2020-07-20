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
#!/usr/bin/env python3
"""
Implementing Boostaroota feature selection.
This algorithm is developed in a similar spirit with Boruta feature selection and
with utilizing XGBoost as the base model while the impact is reducing the run time of Boruta
feature selection method.

@author: Eyal Ben Zion
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import operator
import warnings
########################################################################################
#
# Main Class and Methods
#
########################################################################################
class BOOSTAROOTA(object):

    def __init__(self, metric=None, model_name=None, cutoff=4, iters=10, max_rounds=100, delta=0.1, silent=False):
        """
        Constructor for  Boostaroota feature selection method.

        Parameters:
            metric: specify the evaluation metric; any metric that can be fit to the XGBoost model can be used.
            model_name: name of model.
            cutoff (int): set the cutoff threshold for the feature importance.
            iters (int): number of algorithm iterations.
            max_rounds(int): number of times the core BoostARoota algorithm will run.
            each round eliminates more and more features.
            delta (float): stopping criteria number.
            silent(boolean): set to True if don't want to see the BoostARoota output printed.

        Returns:
            none.
        """

        self.metric = metric
        self.model_name = model_name
        self.cutoff = cutoff
        self.iters = iters
        self.max_rounds = max_rounds
        self.delta = delta
        self.silent = silent
        self.keep_vars_ = None

        #Throw errors if the inputted parameters don't meet the necessary criteria
        # if (metric is None) and (model_name is None):
        #     raise ValueError('you must enter one of metric or model_name as arguments')
        # if cutoff <= 0:
        #     raise ValueError('cutoff should be greater than 0. You entered' + str(cutoff))
        # if iters <= 0:
        #     raise ValueError('iters should be greater than 0. You entered' + str(iters))
        # if (delta <= 0) | (delta > 1):
        #     raise ValueError('delta should be between 0 and 1, was ' + str(delta))
        #
        # #Issue warnings for parameters to still let it run
        # if (metric is not None) and (model_name is not None):
        #     warnings.warn('You entered values for metric and model_name, defaulting to model_name and ignoring metric')
        # if delta < 0.02:
        #     warnings.warn("WARNING: Setting a delta below 0.02 may not converge on a solution.")
        # if max_rounds < 1:
        #     warnings.warn("WARNING: Setting max_rounds below 1 will automatically be set to 1.")


    def fit(self, x, y):
        """
        Fits method for feature selection.

        Parameters:
            x (dataframe): input dataframe.
            y (1d array-like): target column.

        Returns:
            self.
        """

        self.keep_vars_ = _BoostARoota(x, y,
                                       metric=self.metric,
                                       model_name = self.model_name,
                                       cutoff=self.cutoff,
                                       iters=self.iters,
                                       max_rounds=self.max_rounds,
                                       delta=self.delta,
                                       silent=self.silent)

        return self


    def transform(self, x):
        """
        Transforms method for keep selected features from the input dataframe.

        Parameters:
            x (dataframe): input dataframe

        Returns:
            dataframe after filtering through feature selection
        """

        if self.keep_vars_ is None:
            raise ValueError("You need to fit the model first")

        return x[self.keep_vars_]


    def fit_transform(self, x, y):
        """
        Takes dataframe and target as input and applies fit method for
        boostaroota feature selection.

        Parameters:
            x(dataframe): input dataframe.
            y (1d array-like): target column.

        Returns:
            self dataframe after applying feature selection transform on input dataframe x.
        """

        self.fit(x, y)

        return self.transform(x)
########################################################################################
#
# Helper Functions to do the Heavy Lifting
#
########################################################################################
def _create_shadow(x_train):
    """
    Takes all X variables (features), creates copies and randomly shuffles them.

    Parameters:
        x_train (dataframe): the dataframe to create shadow features on.

    Returns:
        new_x (dataframe): dataframe 2x width.
        shadow_names (list): list of shadow names for removing later.
    """

    x_shadow = x_train.copy()
    for c in x_shadow.columns:
        np.random.shuffle(x_shadow[c].values)
    # rename the shadow
    shadow_names = ["SV_" + str(i) for i in x_shadow.columns]
    x_shadow.columns = shadow_names
    # Combine to make one new dataframe
    new_x = pd.concat([x_train, x_shadow], axis=1)
    return new_x, shadow_names
########################################################################################
#
# BoostARoota
#
########################################################################################
def create_features_Scores(df, df2, importance, this_round, i, silent):
    """
    Cretae feature scores.

    Parameters:
        df (dataframe): input dataframe.
        df2 (dataframe): dataframe with importance values.
        importance (float) : importance value of features.
        this_round (int) : number of rounds.
        i (int): iteration number.
        silent(boolean): a condition check parameter, Set to True if don't want to see the BoostARoota output printed.

    Returns:
        df (dataframe): input dataframe merged with dataframe with importance values.
        df2 (dataframe): dataframe with importance values.
    """

    df2['fscore' + str(i)] = importance
    df2['fscore'+str(i)] = df2['fscore'+str(i)] / df2['fscore'+str(i)].sum()
    df = pd.merge(df, df2, on='feature', how='outer')
    if not silent:
        print("Round: ", this_round, " iteration: ", i)
    return df, df2


def select_features(df, shadow_names, cutoff, delta, orig, silent):
    """
    Selects features with the help of shadow variables for each feature.

    Parameters:
        df (dataframe): input dataframe.
        shadow_names (list): name of shadow columns.
        cutoff(float) : a cutoff value to control maximim range of shadows.
        delta (float): a control parameter required for stopping criteria.
        orig (int): number of features in input dataframe.
        silent(boolean): a condition check parameter, set to True if don't want to see the BoostARoota output printed.

    Returns:
        criteria (boolean): True if # features in shadow_names is higher than a threshold else False.
        series (1d array-like): feature column of real_vars dataframe.
    """

    df['Mean'] = df.mean(axis=1)
    #Split them back out
    real_vars = df[~df['feature'].isin(shadow_names)]
    shadow_vars = df[df['feature'].isin(shadow_names)]
    # Get mean value from the shadows
    max_shadow = shadow_vars['Mean'].max() * cutoff
    real_vars = real_vars[(real_vars.Mean > max_shadow)]
    #Check for the stopping criteria
    #Basically looking to make sure we are removing at least 10% of the variables, or we should stop
    if not silent:
        print('The prop is {} and the thresh is {}'.format((len(real_vars['feature']) / float(orig)), (delta)))
    if (len(real_vars['feature']) / float(orig)) < delta:
        criteria = True
    else:
        criteria = False

    return criteria, real_vars['feature']


def _reduce_vars_xgb_reg(x, y, orig, metric, this_round, cutoff, n_iterations, delta, silent):
    """
    function to run the boostaroota feature select_features for regression.

    Parameters:
        x(dataframe): input dataframe.
        y(1d array-like): Target column.
        orig(int): number of features in input dataframe.
        metric: Metric to optimize in XGBoost.
        this_round(int): Round number so it can be printed to screen.
        cutoff(float) : a cutoff value to control maximim range of shadows.
        n_iterations(int) : iteration number.
        delta (float): a control parameter required for stopping criteria.
        silent(boolean): a condition check parameter, Set to True if don't want to see the BoostARoota output printed.

    Returns:
        output of running select_features function.
    """
    #Set up the parameters for running the model in XGBoost
    param = {'booster': 'gbtree',
            'importance_type': 'gain',
             # 'objective': 'reg:squarederror',
             'eval_metric': 'rmse',
             'nthread': 1,
             'seed': 3,
             'verbosity': 0}

    for i in range(1, n_iterations+1):
        # Create the shadow variables and run the model to obtain importances
        new_x, shadow_names = _create_shadow(x)
        bst = xgb.XGBRegressor(**param)
        bst.fit(new_x, y)
        if i == 1:
            df = pd.DataFrame({'feature': new_x.columns})
            df2 = df.copy()
            pass

        importance = bst.feature_importances_
        df, df2 = create_features_Scores(df, df2, importance, this_round, i, silent)

    return select_features(df, shadow_names, cutoff, delta, orig, silent)


def _reduce_vars_xgb_class(x, y, orig, metric, this_round, cutoff, n_iterations, delta, silent):
    """
    function to run the boostaroota feature select_features for classification.

    Parameters:
        x(dataframe): input dataframe.
        y(1d array-like): Target column.
        orig(int): number of features in input dataframe.
        metric: Metric to optimize in XGBoost.
        this_round(int): Round number so it can be printed to screen.
        cutoff(float) : a cutoff value to control maximim range of shadows.
        n_iterations(int) : iteration number.
        delta (float): a control parameter required for stopping criteria.
        silent(boolean): a condition check parameter, Set to True if don't want to see the BoostARoota output printed.

    Returns:
        output of running select_features function.
    """
    #Set up the parameters for running the model in XGBoost
    param = {'booster': 'gbtree',
             'learning_rate': 0.3,
             'importance_type': 'gain',
             'objective':'binary:logistic',
             'eval_metric': 'auc',
             'nthread': 1,
             'seed': 3,
             'verbosity': 0}

    if (sum(y)/len(y)) < 0.1:
        param['scale_pos_weight'] = 1.2

    print (list(x.columns))

    for i in range(1, n_iterations+1):
        # Create the shadow variables and run the model to obtain importances
        new_x, shadow_names = _create_shadow(x)

        bst = xgb.XGBClassifier(**param)
        bst.fit(new_x, y)
        if i == 1:
            df = pd.DataFrame({'feature': new_x.columns})
            df2 = df.copy()
            pass

        importance = bst.feature_importances_
        df, df2 = create_features_Scores(df, df2, importance, this_round, i, silent)

    return select_features(df, shadow_names, cutoff, delta, orig, silent)


def _BoostARoota(x, y, metric, model_name, cutoff, iters, max_rounds, delta, silent):
    """
    Main function to run the boostaroota feature select_features.

    Parameters:
        x (dataframe): input dataframe.
        y (1d array-like): target column.
        metric: metric to optimize in XGBoost.
        model_name (str) : name of the model.
        cutoff (float) : a cutoff value to control maximim range of shadows.
        iters (int) : number of iterations.
        max_rounds (int): maximum number of rounds.
        delta (float): a control parameter required for stopping criteria.
        silent (boolean): a condition check parameter, Set to True if don't want to see the BoostARoota output printed.

    Returns:
         keep_vars (list): output of running select_features function.
    """
    new_x = x.copy()
    length_columns = len(list(x))
    #Run through loop until "crit" changes
    i = 0
    j = 0
    while True:
        #Inside this loop we reduce the dataset on each iteration exiting with keep_vars
        i += 1
        if model_name == 'xgb_class':
            crit, keep_vars = _reduce_vars_xgb_class(new_x,
                                               y,
                                               length_columns,
                                               metric=metric,
                                               this_round=i,
                                               cutoff=cutoff,
                                               n_iterations=iters,
                                               delta=delta,
                                               silent=silent)
        elif model_name == 'xgb_reg':
            crit, keep_vars = _reduce_vars_xgb_reg(new_x,
                                               y,
                                               length_columns,
                                               metric=metric,
                                               this_round=i,
                                               cutoff=cutoff,
                                               n_iterations=iters,
                                               delta=delta,
                                               silent=silent)

        if len(keep_vars) == 0:
            cutoff /= 2
            j += 1
            if (j >= 5):
                keep_vars = list(new_x)
                break
        elif crit | (i >= max_rounds):
            break  # exit and use keep_vars as final variables
        else:
            new_x = new_x[keep_vars].copy()
            if not silent:
                print("Number of features in round {} is {}".format(i, len(list(new_x))))
    if not silent:
        print("BoostARoota ran successfully! Algorithm went through ", i, " rounds.")
        print("Final number of features is {}".format(len(keep_vars)))

    return keep_vars
