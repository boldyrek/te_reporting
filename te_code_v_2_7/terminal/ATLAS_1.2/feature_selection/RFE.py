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
@author:
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""
module_name = 'Recursive Feature Elimination'
version_name = '1.2'

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

from base_pd.base_feature_selection_pd import PdBaseFeatureSelect as BaseFeatureSelect


class RFE_TE(BaseFeatureSelect):
    """
    This is a class for Recursive Feature Elimination (RFE) feature selection.

    Attributes:
        rack_order (int): the order of the rack (i.e. 7).
        data_plus_meta (dict): dictionary contining data and meta data.
        racks_map (dict): dictionary of all racks and their orders.
    """

    def __init__(self, rack_order, data_plus_meta, racks_map):
        """
        Constructor for RFE feature selection.

        Parameters:
            rack_order (int): the order of the rack (i.e. 7).
            data_plus_meta (dict): dictionary contining data and meta data.
            racks_map (dict): dictionary of all racks and their orders.

        Returns:
           ises none
        """

        super(RFE_TE, self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)


    def fit(self, df, cfg):
        """
        Perfomrs RFE feature selection.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            selected_features: list of selected variable names.
        """

        date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']

        all_features = [x for x in df.columns if x not in cfg['drop_cols']+[cfg['ID_COL'], cfg['CTU_COL']]+date_cols]

        for i in all_features:
            if sum(df[i].isnull()) > 0:
                df[i] = df[i].fillna(0)

        X = df[all_features]
        y = df[cfg['TE_TARGET_COL']]

        if (sum(y)/len(y)) < 0.1:
            class_ratio = (len(y) - sum(y))/sum(y)
            print ("Class Ratio:", class_ratio)
            class_weight = dict({1:class_ratio, 0:1.5})
            max_depth = 4
            n_estimators = 100
        else:
            class_weight = None
            max_depth = 4
            n_estimators = 100

        param = {
                     'bootstrap':True,
                     'class_weight':class_weight,
                     'criterion':'gini',
                     'max_depth': max_depth, 'max_features':'auto', 'max_leaf_nodes':None,
                     'min_impurity_decrease' :0.0, 'min_impurity_split':None,
                     'min_samples_leaf':2, 'min_samples_split':10,
                     'min_weight_fraction_leaf':0.0, 'n_estimators':n_estimators,
                     'oob_score':False,
                     'random_state':121,
                     'verbose':0,
                     'warm_start':False
            }


        estimator = RandomForestClassifier(**param)

        feat_selector = RFE(estimator, cfg['n_features'], cfg['n_step'])
        feat_selector.fit(X, y)

        selected_features = [col for (col, id_bool) in zip(all_features, feat_selector.support_) if id_bool]

        return selected_features


    def transform(self, df, selected_feature):
        """
        Removes variables in selected_feature from dataframe.

        Parameters:
            df (dataframe): dataframe.
            selected_feature (list): list of variable names to be removed.

        Returns:
            filtered_data: dataframe excluding the variables in selected_feature.
        """

        feature_name = selected_feature
        filtered_data = df[feature_name]

        return filtered_data
