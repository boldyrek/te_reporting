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
import numpy as np
import pandas as pd
import random

from splitting.attribute_splitting import SplitAttribute
from base.base_splitting import BaseSplitting
from base.base_rack import BaseRack


class PdBaseSplitting(BaseSplitting):
    """
    This is the base class for splitting data.

    Attributes:
        module_name (str): name of module.
        version_name (str): version name of module.
        rack_order (int): the order of the rack (i.e. 5).
        data_plus_meta (dict): dictionary contining data and meta data.
        racks_map (dict): dictionary of all racks and their orders.
    """

    def __init__(self, module_name, version_name, rack_order, data_plus_meta, racks_map):

        """
        Constructor to initialize the splitting module.

        Attributes:
            module_name (str): name of module.
            version_name (str): version name of module.
            rack_order (int): the order of the rack (i.e. 5).
            data_plus_meta (dict): dictionary contining data and meta data.
            racks_map (dict): dictionary of all racks and their orders.
        """

        super(PdBaseSplitting , self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)


    def filter_low_variance_features(self, df, cfg):
        """
        Creates a list of variables where the mode represents most of values. most will be defined in config.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            list of the names of low-variance variables.
        """

        list_of_low_var_count = []
        cols = [x for x in df.columns if x not in [cfg['TE_TARGET_COL'] , 'yr_week', cfg['REF_EVENT_COL']]]

        for i in cols:

            if sum(df[i].isnull()) > 0:
                df[i] = df[i].fillna(0)

            values = df[i].value_counts()
            if values.max()/values.sum() > cfg['low_var_split_threshold']:
                list_of_low_var_count.append(i)

        return list(set(list_of_low_var_count))


    def transform(self, df, high_var_cols):
        """
        Removes low-variance columns.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            filtered_data: dataframe excluding low-variance features.
        """

        feature_name = high_var_cols
        filtered_data = df[feature_name]

        return filtered_data


    def offset_date(self, df, cfg):
        """
        Gets the last CTU with which has total performance window labels (Targets).

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            ctu_offset-1 (int): CTU where targets are defined with complete performance window.
        """
        # Get the ctu which has the full look forward performance window
        ctu_offset = df[df['datetime'] <= (cfg["last_date"] - pd.to_timedelta(cfg['TE_WINDOW_DAY'],unit='d'))][cfg['CTU_COL']].min()
        print ('last date with full labels: ', (cfg["last_date"] - pd.to_timedelta(cfg['TE_WINDOW_DAY'],unit='d')))
        print ('ctu_offset: ', ctu_offset)

        return ctu_offset-1


    def splitting_train_validate_data(self, df, cfg):
        """
        Splits train data into train and validation based on CTU number.

        Parameters:
            df (dataframe): train dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df_train: train dataframe (excludes CTUs in validation; out-of-sampled).
            df_validate: validation dataframe (includes m (i.e, number of validation folds)
            CTUs; out-of-sampled).
        """

        df_validate = df[df[cfg['CTU_COL']] <= cfg['folds_validation']]

        df_train = df[df[cfg['CTU_COL']] > cfg['folds_validation']]

        print ("\nSplit Train and validation set - :")
        print ("TRAIN - number of customers:", df_train[cfg['ID_COL']].nunique())
        print ("VALIDATE - number of customers:", df_validate[cfg['ID_COL']].nunique())

        return df_train, df_validate


    def splitting_customer_ctu_train_validate(self, df, cfg):
        """
        Splits train data into train and validate proportions based on the train_val percent and CTU number.

        Parameters:
            df (dataframe): train dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df_train: train dataframe (excludes CTUs in validation; out-of-sampled).
            df_validate: validation dataframe (includes m (i.e, number of validation folds)
            CTUs; out-of-sampled)
        """
        # Get train data ids
        all_ids = df[cfg['ID_COL']].unique()

        np.random.seed(cfg['seed'])
        id_tr = np.random.choice(all_ids, int(len(all_ids) * cfg['train_val_pct']), replace=False)

        df_train,_ = self.splitting_train_validate_data(df[df[cfg['ID_COL']].isin(id_tr)], cfg)

        _, df_validate = self.splitting_train_validate_data(df[~df[cfg['ID_COL']].isin(id_tr)], cfg)

        return   df_train, df_validate


    def splitting_predict_data(self, df, cfg):
        """
        Extracts the predict set out of the enriched data based on CTU number.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df_pred: test dataframe (all customers who has data in test CTU; not out-of-sample yet).
        """

        df_pred = df[df[cfg['CTU_COL']] == cfg['test_ctu']]
        #TODO:check this next two lines. the goal is to remove accounts with reference events in the test CTU
        ids_terminal = df_pred[df_pred[cfg['REF_EVENT_COL']]==1][cfg['ID_COL']].unique()

        df_pred = df_pred[~df_pred[cfg['ID_COL']].isin(ids_terminal)]

        print (f"\nSplit Prediction set - {cfg['CTU_COL']} = 0:")
        print ("No of customers in pred set:", df_pred[cfg['ID_COL']].nunique())
        print ("Maximum & Minimum datetime in the prediction set:", df_pred['datetime'].max(), str(','), df_pred['datetime'].min())

        return df_pred

    def splitting_predict_data_intime(self, df, cfg):
        """
        Splits test-train data based on the train_val percent for the In-time random sample test.

        Parameters:
            df (dataframe): full dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df_train: train dataframe (out-of-sampled).
            df_pred: prediction dataframe (out-of-sampled)
            CTUs; out-of-sampled)
        """
        # Get train data ids
        all_ids = df[cfg['ID_COL']].unique()

        np.random.seed(cfg['seed'])
        id_tr = np.random.choice(all_ids, int(len(all_ids) * cfg['train_val_pct']), replace=False)

        df_train = df[df[cfg['ID_COL']].isin(id_tr)]
        df_pred = df[~df[cfg['ID_COL']].isin(id_tr)]

        # randommly selecting one active CTU for each client in the prediction data
        df_pred = df_pred.groupby(cfg['ID_COL']).apply(lambda x: x.sample(1)).reset_index(drop=True)
        
        return df_train, df_pred


    def splitting_cv_random_folds(self, df, cfg):
        """
        Creates cv splits (only ids and CTUs for each fold) on train data

        random n folds based on the indices

        Parameters:
            df (dataframe): train dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            cv_id_dict (dict): dictionary with n (number of folds) keys;
            each key has a dataframe with customers ids and CTUs.
        """
        # Number of elements in each fold
        n_samples_per_fold = int(df.shape[0]/cfg['folds_cv'])
        # Get list of indices in each fold
        val_id_list = [list(df.index[i * n_samples_per_fold:(i + 1) * n_samples_per_fold]) for i in range(0,cfg['folds_cv'])]
        # Save the keys for each fold
        cv_id_dict = {}
        for i in range(0, cfg['folds_cv']):
            cv_id_dict[i+1] = df.loc[val_id_list[i], [cfg['ID_COL'],cfg['CTU_COL']]]

        return cv_id_dict
