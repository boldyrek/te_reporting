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
module_name = 'CTU_Imputation'
version_name = '1.2'

import pandas as pd
import gc

from base_pd.base_imputation_pd_churn import PdBaseImputation as BaseImputation

class CTUIMPUTATION(BaseImputation):
    """
    This is a class for imputing empty CTUs.

    Attributes:
        rack_order (int): the order of the rack (i.e. 3).
        data_plus_meta (dict): dictionary contining data and meta data.
        racks_map (dict): dictionary of all racks and their orders.
    """

    def __init__(self, rack_order, data_plus_meta, racks_map):
        """
        Constructor for imputation - specifically imputing for empty CTUs.

        Parameters:
            rack_order (int): the order of the rack (i.e. 3).
            data_plus_meta (dict): dictionary contining data and meta data.
            racks_map (dict): dictionary of all racks and their orders.

        Returns:
            none
        """

        super(CTUIMPUTATION, self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)

    def run(self):
        """
        Constructor to run the imputation module.

        Parameters
            self

        Returns:
            none
        """

        data_dict = self.data_plus_meta_[self.rack_order_ - 1].data_
        cfg = self.data_plus_meta_[self.rack_order_].config_
        self.data_plus_meta_[self.rack_order_].data_ = {}

        for k,v in data_dict.items():

            df = pd.read_feather(v)

            df[cfg['EVENT_DATE']] = pd.to_datetime(df[cfg['EVENT_DATE']])

            file_id = [x for x,y in cfg['DATA_FILE_DICT'].items() if k==y][0]

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
            #TODO: check the following functions (not used in Vonage)
            elif cfg['STEP'].lower()== "predict":
                df = self.prediction_imputation(df, cfg)

                if cfg['KEEP_TERMINAL_CTU']:

                    df = self.remove_terminal_events(df, cfg)
                else:
                    df = self.remove_churn_ctu(df, cfg)

                save_file = cfg['IMPUTATION_PREDICT_PATH'] + cfg['t_date'] + '_' + str(file_id)+ '.feather'
                df.reset_index(drop=True).to_feather(save_file)
                self.data_plus_meta_[self.rack_order_].data_[k] = save_file

        del df
        gc.collect()

        self.data_plus_meta_[self.rack_order_].config_['binary_cols'] += ['imputed_ctu']
