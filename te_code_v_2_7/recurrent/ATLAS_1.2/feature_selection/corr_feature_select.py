#!/usr/bin/env python3
"""
@author: Sathish Lakshmipathy
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""
module_name = 'Corr_Feature_Selection'
version_name = '1.2'

import numpy as np
import pandas as pd
import operator

from base_pd.base_feature_selection_pd import PdBaseFeatureSelect as BaseFeatureSelect


class CORR_FILT(BaseFeatureSelect):

    def __init__(self, rack_order, data_plus_meta, racks_map):
        """
        Constructor for correlation base feature selection method
        param:
            data_plus_meta: A dictionary inclusing all imputing attributes
            order:
        :return selected freatures with high correlation with target.
        :raises none
        """

        super(CORR_FILT, self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)


    def fit(self, df, cfg):
        """
        Includes functions for feature selection.

        filter_sparse_cols: adds variable names with high zero proportion to a list.
        filter_single_value_features: adds variable names with only one value to a list.
        filter_low_variance_features: adds variable names with a dominant mode value to a list.
        filter_correlated_features: adds variable names that are highly-correlated with other variables to a list.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            selected_features: list of selected variable names.
        """
        # Exclude columns from drop list
        all_features = [x for x in df.columns if x not in cfg['drop_cols']+[cfg['ID_COL'], cfg['CTU_COL']]]

        if cfg['sparsity_limit'] == 1:
            non_sparse_cols = list(set(list(all_features)))
        else:
            non_sparse_cols = self.filter_sparse_cols(df[all_features],cfg)
        #high variance function will also exclue the single value cols
        non_single_valued_cols = self.filter_single_value_features(df[non_sparse_cols])
        if cfg['low_var_threshold'] == 1:
            high_var_cols = non_single_valued_cols
        else:
            high_var_cols = self.filter_low_variance_features(df[non_single_valued_cols], cfg)
        high_var_cols.sort()

        if cfg['correlation_threshold'] == 1:
            selected_features = high_var_cols
        else:
            selected_features = \
                  self.filter_correlated_features(df[high_var_cols], cfg['correlation_threshold'])
        selected_features.sort()

        return  list(set(selected_features))


    def transform(self, df, selected_feature):
        """
        Removes variables in selected_feature from dataframe.

        Parameters:
            df (dataframe): dataframe.
            selected_feature (list): list of variable names to be removed.

        Returns:
            filtered_data: dataframe excluding the variables in selected_feature.
        """

        feature_name = [x for x in selected_feature if x in df.columns]
        filtered_data = df[feature_name]

        return filtered_data


    def filter_correlated_features(self, df, corr_coef_threshold=0.85):

        """
        Get columns that has correlation < corr_coef

        Parameters:
            df (dataframe): dataframe.
            corr_coef_threshold (float): correlation threshold; deafult value is 0.85

        Returns:
            selected_features: list of variables that are correlated less than the threshold
        """
        # Get upper triangle of correlation matrix
        corr_matrix = df.corr().abs()
        upper_corrs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        drop_features = [col for col in upper_corrs.columns if any(upper_corrs[col] > corr_coef_threshold)]
        selected_features = [col for col in df.columns if col not in drop_features]

        return selected_features
