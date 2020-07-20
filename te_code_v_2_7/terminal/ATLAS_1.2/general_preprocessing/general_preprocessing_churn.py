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

from base_pd.base_general_preprocessing_pd_churn import PdBaseGeneralPreprocess as BaseGeneralPreprocess
import util.operation_utils as op_util

class GENERALPREPROCESSING(BaseGeneralPreprocess):
    """
    This is a class for conducting general preprocessing on data.

    Attributes:
        rack_order (int): the order of the rack (i.e. 0).
        data_plus_meta (dict): dictionary contining data and meta data.
        racks_map (dict): dictionary of all racks and their orders.
    """

    def __init__(self, rack_order, data_plus_meta, racks_map):
        """
        Constructor for General Preprocessing

        Parameters:
            rack_order (int): the order of the rack (i.e. 0).
            data_plus_meta (dict): dictionary contining data and meta data.
            racks_map (dict): dictionary of all racks and their orders.

        Returns:
            none
        """
        super(GENERALPREPROCESSING, self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)


    def run(self):
        """
        Constructor to run the general preprocessing module.

        Parameters
            self

        Returns:
            none
        """
        data_list = self.data_plus_meta_[self.rack_order_ - 1].data_
        cfg = self.data_plus_meta_[self.rack_order_].config_
        self.data_plus_meta_[self.rack_order_].data_ = {}
        #create aggregation windoe
        cfg = op_util.get_agg_window_from_cfg(cfg)
        #create CTU_UNIT
        cfg = op_util.get_te_ctu_from_cfg(cfg)

        for i,j in data_list.items():

            df = pd.read_csv(cfg['DATA_PATH'] + j, sep="\t",parse_dates=cfg['PARSE_DATE_COLS'], low_memory=False)

            df = self.general_preprocess_data(df, cfg)

            df = self.filter_past_sparse_data(df, cfg)

            save_file = cfg['GEN_PREPROCESS_PATH'] + cfg['t_date'] + '_' + str(i) + '.feather'

            df.to_feather(save_file)

            self.data_plus_meta_[self.rack_order_].data_[j] = save_file

        self.data_plus_meta_[self.rack_order_].config_ = op_util.get_column_groups(df, cfg)
        del df
        gc.collect()


    def general_preprocess_data(self, df, cfg):
        """
        Contains functions to preprocess data in Té

        Parameters:
            df (dataframe): raw dataframe
            cfg (dict): configuration dictionary

        Returns:
            df: preprocessed dataframe
        """
        df.sort_values(by=[cfg['ID_COL'],cfg['EVENT_DATE'], cfg['EVENT_ID']], ascending=[True, True, True], inplace=True)

        df = self.clean_data(df,cfg)

        df = self.flb_imputation(df, cfg)

        df = self.remove_negative_tenure(df, cfg)

        df = self.remove_customers_with_short_journey(df, cfg)

        df = self.create_ref_event_cols(df, cfg)

        df = self.truncate_churn_journey(df, cfg)

        df = self.create_yr_ctu_cols(df, cfg)

        df = self.create_semantic_interaction_cols(df, cfg)

        df = self.create_interaction_td_cols(df, cfg)

        return df.reset_index()
