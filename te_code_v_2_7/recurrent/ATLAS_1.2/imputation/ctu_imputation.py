#!/usr/bin/env python3
"""
General Preprocessing class for Té.

@author: Sathish K Lakshmipathy
@version: 1.0
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""
module_name = 'CTU_Imputation'
version_name = '1.2'

import pandas as pd
import gc

from base_pd.base_imputation_pd import PdBaseImputation as BaseImputation

class CTUIMPUTATION(BaseImputation):

    def __init__(self, rack_order, data_plus_meta, racks_map):
        """
        Constructor for imputation - specifically imputing for empty CTUs

        :param: df: raw data
    			config: the .py configuration namespace.
        :return: none
        :raises: none
        """
        super(CTUIMPUTATION, self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)

    def run(self):
        """
        Constructor to run the imputation module.

        :param
            self
        :return raw data as dataframe
        :raises none
        """
        data_dict = self.data_plus_meta_[self.rack_order_ - 1].data_
        cfg = self.data_plus_meta_[self.rack_order_].config_
        self.data_plus_meta_[self.rack_order_].data_ = {}

        for k,v in data_dict.items():
            df = pd.read_feather(v)

            df[cfg['EVENT_DATE']] = pd.to_datetime(df[cfg['EVENT_DATE']])

            file_id = [x for x,y in cfg['DATA_FILE_DICT'].items() if k==y][0]

            # Get filepath
            if cfg['STEP'].lower() in ["train", "both"]:
                # If the test CTU labels are included for training
                if cfg['INCLUDE_TEST_LABELS']:
                    print("Including the full test labels in data..")
                    df = self.training_imputation(df, cfg)
                    df = self.filter_data_by_date(df, cfg)

                else:
                    df = self.filter_data_by_date(df, cfg)
                    df = self.training_imputation(df, cfg)

                save_file = cfg['IMPUTATION_TRAIN_PATH'] + cfg['t_date'] + '_' + str(file_id) + '.feather'
                df.reset_index(drop=True).to_feather(save_file)
                self.data_plus_meta_[self.rack_order_].data_[k] = save_file

            elif cfg['STEP'].lower()== "predict":
                df = self.prediction_imputation(df, cfg)
                save_file = cfg['IMPUTATION_PREDICT_PATH'] + cfg['t_date'] + '_' + str(file_id)+ '.feather'

                df.to_feather(save_file)
                self.data_plus_meta_[self.rack_order_].data_[k] = save_file

        del df
        gc.collect()

        self.data_plus_meta_[self.rack_order_].config_['binary_cols'] += ['imputed_ctu']
