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
@author: Sathish Lakshmipathy
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""
module_name = 'Boruta_Feature_Selection'
version_name = '1.2'

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from util.boostaroota import BOOSTAROOTA

from base_pd.base_feature_selection_pd import PdBaseFeatureSelect as BaseFeatureSelect


class BOOSTAROOTA_PY(BaseFeatureSelect):
    """
    This is a class for Boostaroota feature selection.

    Attributes:
        rack_order (int): the order of the rack (i.e. 7).
        data_plus_meta (dict): dictionary contining data and meta data.
        racks_map (dict): dictionary of all racks and their orders.
    """

    def __init__(self, rack_order, data_plus_meta, racks_map):
        """
        Constructor for Boostaroota feature selection.

        Parameters:
            rack_order (int): the order of the rack (i.e. 7).
            data_plus_meta (dict): dictionary contining data and meta data.
            racks_map (dict): dictionary of all racks and their orders.

        Returns:
            none
        """

        super(BOOSTAROOTA_PY, self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)

    def fit(self, df, cfg):
        """
        Performs Boostaroota feature selection.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            selected_features: list of selected variable names.
        """

        date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
        all_features = [x for x in df.columns if x not in cfg['drop_cols']+[cfg['ID_COL'], cfg['CTU_COL']]+date_cols]

        X = df[all_features]
        y = df[cfg['TE_TARGET_COL']]

        feat_selector = BOOSTAROOTA(metric=cfg['boostaroota_metric'], model_name=cfg['model_name'], cutoff=cfg['cutoff'], iters=cfg['iters'], max_rounds=cfg['max_rounds'], delta=cfg['delta'], silent=False)

        selected_features = feat_selector.fit(X, y)
        print("Boostaroota selected feature count:", len(selected_features.keep_vars_))

        return list(selected_features.keep_vars_)


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
