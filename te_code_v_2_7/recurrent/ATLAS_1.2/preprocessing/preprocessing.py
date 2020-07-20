#!/usr/bin/env python3
"""
Preprocessing class for Té.

@author: Sathish K Lakshmipathy
@version: 1.0
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""
module_name = 'Preprocessing'
version_name = '1.2'

import pandas as pd
import gc

from base_pd.base_preprocessing_pd import PdBasePreprocessing as BasePreprocessing


class PREPROCESSING(BasePreprocessing):

    def __init__(self, rack_order, data_plus_meta, racks_map):
        """
        Constructor for preprocessing

        :param: df: general preprocessed df
    			config: the .py configuration namespace.
        :return: none
        :raises: none
        """
        super(PREPROCESSING, self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)


    def run(self):
        """
        Constructor to run preprocessing module.

        :param
            self
        :return imputed vinatge as dataframe
        :raises none
        """
        data_list = self.data_plus_meta_[self.rack_order_ - 1].data_
        cfg = self.data_plus_meta_[self.rack_order_].config_
        self.data_plus_meta_[self.rack_order_].data_ = {}

        for k,v in data_list.items():

            print ("Preprocessing:",v)

            df = pd.read_feather(v)

            df[cfg['EVENT_DATE']] = pd.to_datetime(df[cfg['EVENT_DATE']])

            df.sort_values(by=[cfg['ID_COL'],cfg['EVENT_DATE']], ascending=[True, True], inplace=True)
            
            df = self.create_te_ctu(df, cfg)

            file_id = [x for x,y in cfg['DATA_FILE_DICT'].items() if k==y][0]

            save_file = cfg['PREPROCESSED_PATH']+ cfg['t_date'] + '_' + str(file_id) + '.feather'

            df.to_feather(save_file)

            self.data_plus_meta_[self.rack_order_].data_[k] =  save_file
        self.data_plus_meta_[self.rack_order_].config_ = cfg

        del df
        gc.collect()
