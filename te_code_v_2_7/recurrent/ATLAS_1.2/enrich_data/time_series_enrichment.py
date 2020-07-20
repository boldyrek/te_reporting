#!/usr/bin/env python3
"""
Data Enrichment class for Té.

@author: Sathish K Lakshmipathy
@version: 1.0
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""
module_name = 'TimeSeries_Enriching'
version_name = '1.2'

import pandas as pd
import gc


from base_pd.base_enrich_data_pd import PdBaseEnrichData as BaseEnrichData

class TSENRICHDATA(BaseEnrichData):

    def __init__(self, rack_order , data_plus_meta, racks_map):
        """
        Constructor for Enriching Data

        :param: df: raw data
    			config: the .py configuration namespace.
        :return: none
        :raises: none
        """
        super(TSENRICHDATA, self).__init__(module_name, version_name, rack_order , data_plus_meta, racks_map)


    def run(self):
        """
        Constructor to run the Enriching module.

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

            df = self.aggregate_over_days(df, cfg, cfg['agg_window'])##TODO: check lists in get_column_groups in operation_utils.py

            df = self.aggregate_over_ctu(df, cfg)##TODO: check lists in get_column_groups in operation_utils.py

            #if cfg['PRICE_MODEL']:
            #    df = self.remove_small_price(df, cfg)

            df = self.apply_training_filter(df, cfg)

            save_file = cfg['ENRICH_PATH'] + cfg['t_date'] + '_no_outliers_' + str(file_id) + '.feather'

            df.to_feather(save_file)

            df = self.check_outliers(df, cfg, file_id)

            save_file = cfg['ENRICH_PATH'] + cfg['t_date'] + '_' + str(file_id) + '.feather'

            df.to_feather(save_file)

            del df

            self.data_plus_meta_[self.rack_order_].data_[k] = save_file

        gc.collect()
