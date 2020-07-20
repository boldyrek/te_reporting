#!/home/dev/.conda/envs/py365/bin/python3.6
"""
Implements base class for feature selection

@author: Sathish Lakshmipathy
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""
import numpy as np
import pandas as pd
import pickle
import gc

from base.base_feature_selection import BaseFeatureSelect
from splitting.splitting_dataset import SplitDataset


class PdBaseFeatureSelect(BaseFeatureSelect):

    def __init__(self, module_name, version_name, rack_order, data_plus_meta, racks_map):
        """
        Constructor to initialize the feature selection module.

        :param
        	module_name: name of the feature selection module.
            version_name: version of feature selection module.
            data_plus_meta: A dictionary including all feature selection attributes
            rack_order:
        :return none
        :raises none
        """
        super(PdBaseFeatureSelect , self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)

        self.data_plus_meta_[self.rack_order_].data_ = SplitDataset()

    def run(self):
        """
        Constructor to run the feature selection module.

        Parameters
            self

        Returns:
            none
        """

        data_dict = self.data_plus_meta_[self.rack_order_ -1].data_
        cfg = self.data_plus_meta_[self.rack_order_].config_

        for k,v in data_dict.train_set_dict_.items():
            print('loading: ', k, v)

            tr_df, cv_df = pickle.load(open(v, "rb"))

            selected_feature = self.fit(tr_df, cfg)
            selected_feature = list(set(selected_feature + [cfg['TE_TARGET_COL'], cfg['ID_COL'], cfg['CTU_COL']]))
            selected_feature.sort()
            # save the full training data features selected to use for prediction data
            tr_df = self.transform(tr_df, selected_feature)
            if k == 'full':
                tr_selected_feature = selected_feature
                cv_df = pd.DataFrame(columns=selected_feature)
            else:
                cv_df = self.transform(cv_df, selected_feature)

            final_train_file = cfg['FINAL_TRAIN_PATH'] + cfg['t_date']+ '_' + str(k) + '.pkl'
            pickle.dump((tr_df, cv_df), open(final_train_file, "wb" ), protocol=4)
            self.data_plus_meta_[self.rack_order_].data_.train_set_dict_[k] = final_train_file

        del tr_df, cv_df

        if cfg['VALIDATION_MODEL']:
            for ke,val in data_dict.validate_set_dict_.items():
                dt, cv = pickle.load(open(val, "rb"))

                selected_feature = self.fit(dt, cfg)
                selected_feature = list(set(selected_feature + [cfg['TE_TARGET_COL'], cfg['ID_COL'], cfg['CTU_COL']]))

                dt = self.transform(dt, selected_feature)

                cv = self.transform(cv, selected_feature)

                final_val_file = cfg['FINAL_VALIDATE_PATH'] + cfg['t_date']+ '_' + str(ke) + '.pkl'
                pickle.dump((dt, cv), open(final_val_file, "wb" ), protocol=4)
                # if there are no folds for validation, there would be only one validation file with index 0
                self.data_plus_meta_[self.rack_order_].data_.validate_set_dict_[ke] = final_val_file

            del dt, cv

        if cfg['is_final_pipeline']:

            pred_data = pd.DataFrame()
            # Iterate over all the prediction files
            for ke,val in data_dict.predict_set_dict_.items():
                dt = pickle.load(open(val, "rb"))

                # Use the features form full training dataset
                dt = self.transform(dt, tr_selected_feature)
                pred_data = pd.concat([dt, pred_data], axis=0)

            final_pred_file = cfg['FINAL_PREDICT_PATH'] + cfg['t_date'] + '.pkl'
            pickle.dump(pred_data, open(final_pred_file, "wb" ), protocol=4)
            self.data_plus_meta_[self.rack_order_].data_.predict_set_dict_['pred'] = final_pred_file


            del dt, pred_data
            gc.collect()

        print ("\nTrain Filename:")
        [print(key,":\n",value) for key,value in self.data_plus_meta_[self.rack_order_].data_.train_set_dict_.items()]
        if cfg['VALIDATION_MODEL']:
            print ("\nValidate Filename:")
            [print(key,":\n",value) for key,value in self.data_plus_meta_[self.rack_order_].data_.validate_set_dict_.items()]
        print ("\nPrediction Filename:")
        [print(key,":\n",value) for key,value in self.data_plus_meta_[self.rack_order_].data_.predict_set_dict_.items()]


    def filter_sparse_cols(self, df, cfg):
        """
        Gets columns that has values > 0 for atleast (1-sparsity_limit)% of the rows

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            list of columns with data in atleast (1-sparsity_limit)% of rows.
        """

        ref_cols = [x for x in df.columns if cfg['TE_TARGET_COL'] in x]

        sparse_pctle = df.quantile(cfg['sparsity_limit'])

        return list(set(list(sparse_pctle[sparse_pctle > 0].index) + ref_cols))


    def filter_single_value_features(self, df):
        """
        Removes columns with only one value.

        Parameters:
            df (dataframe): dataframe.

        Returns:
            list of columns with more than 1 unique numerical value.
        """

        non_singular = df.select_dtypes(include=[np.number])

        return list((non_singular.nunique() > 1).index)


    def filter_low_variance_features(self, df, cfg):
        """
        Removes columns where the mode represents most of values.

        threshold is defined in config

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            list_of_high_var_count: list of columns where the proportion of mode is less than the threshold
        """

        list_of_high_var_count = []
        for i in df.columns:
            values = df[i].value_counts()
            if values.max()/values.sum() < cfg['low_var_threshold']:
                list_of_high_var_count.append(i)

        return list_of_high_var_count
