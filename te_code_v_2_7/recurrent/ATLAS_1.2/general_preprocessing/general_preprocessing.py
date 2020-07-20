#!/usr/bin/env python3
"""
General Preprocessing class for Té.

@author: Sathish K Lakshmipathy
@version: 1.0
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""
module_name = 'GeneralPreprocessing'
version_name = '1.2'

import numpy as np
import pandas as pd
import gc
import datetime as dt

from base_pd.base_general_preprocessing_pd import PdBaseGeneralPreprocess as BaseGeneralPreprocess
import util.operation_utils as op_util

class GENERALPREPROCESSING(BaseGeneralPreprocess):

    def __init__(self, rack_order, data_plus_meta, racks_map):
        """
        Constructor for General Preprocessing

        :param: df: raw data
    			config: the .py configuration namespace.
        :return: none
        :raises: none
        """
        super(GENERALPREPROCESSING, self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)


    def run(self):
        """
        Constructor to run the general preprocessing module.

        :param
            self
        :return raw data as dataframe
        :raises none
        """
        data_list = self.data_plus_meta_[self.rack_order_ - 1].data_
        cfg = self.data_plus_meta_[self.rack_order_].config_
        self.data_plus_meta_[self.rack_order_].data_ = {}
        cfg = op_util.get_agg_window_from_cfg(cfg)
        #create CTU_UNIT
        cfg = op_util.get_te_ctu_from_cfg(cfg)
        print('data path:',cfg['DATA_PATH'])
        for i,j in data_list.items():
            print('data path',cfg['DATA_PATH'])
            print('j',j)
            #df = pd.read_csv(cfg['DATA_PATH'] + j, sep="\t", parse_dates=[cfg['EVENT_DATE'], cfg['PURCHASE_DATE']], low_memory=False)
            df = pd.read_csv(cfg['DATA_PATH'] + j)

            df = self.general_preprocess_data(df, cfg)

            df = self.filter_past_sparse_data(df, cfg)

            save_file = cfg['GEN_PREPROCESS_PATH'] + cfg['t_date'] + '_' + str(i) + '.feather'

            df.to_feather(save_file)
            # Get filepath
            self.data_plus_meta_[self.rack_order_].data_[j] = save_file

        self.data_plus_meta_[self.rack_order_].config_ = op_util.get_column_groups(df, cfg)
        del df
        gc.collect()


    def general_preprocess_data(self, df, cfg):
        """
        Contains the Té complete preprocessing
        1. Filter data by date
        2. clean column names
        3. Check if all customers have atleast one reference event
        """
        df.sort_values(by=[cfg['ID_COL'],cfg['EVENT_DATE'], cfg['EVENT_ID']], ascending=[True, True, True], inplace=True)

        df = self.clean_data(df, cfg)

        #df = self.flb_imputation(df, cfg)

        #df = self.remove_negative_tenure(df, cfg)

        #df = self.remove_customers_with_short_journey(df, cfg)

        df = self.check_reference_event(df, cfg)

        #df = self.create_event_cols(df, cfg)##TODO keep it for now, no need if flb has sum_over_journey cols

        df = self.create_ref_event_cols(df, cfg)

        df = self.create_td_cols(df, cfg)

        df = self.create_yr_ctu_cols(df, cfg)

        df = self.create_semantic_interaction_cols(df, cfg)

        df = self.create_interaction_td_cols(df, cfg)

        df = self.event_filters(df, cfg)

        df = self.remove_timeline_0_purchases(df, cfg)

        return df.reset_index()
