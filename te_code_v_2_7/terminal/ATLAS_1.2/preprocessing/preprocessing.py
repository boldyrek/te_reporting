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
    """
    This is a class for conducting preprocessing on data.

    Attributes:
        rack_order (int): the order of the rack (i.e. 0).
        data_plus_meta (dict): dictionary contining data and meta data.
        racks_map (dict): dictionary of all racks and their orders.
    """

    def __init__(self, rack_order, data_plus_meta, racks_map):
        """
        Constructor for Preprocessing

        Parameters:
            rack_order (int): the order of the rack (i.e. 0).
            data_plus_meta (dict): dictionary contining data and meta data.
            racks_map (dict): dictionary of all racks and their orders.

        Returns:
            none
        """
        super(PREPROCESSING, self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)


    def run(self):
        """
        Constructor to run the preprocessing module.

        Parameters:
            self

        Returns:
            none
        """
        data_list = self.data_plus_meta_[self.rack_order_ - 1].data_
        cfg = self.data_plus_meta_[self.rack_order_].config_
        self.data_plus_meta_[self.rack_order_].data_ = {}

        for k,v in data_list.items():

            print ("Preprocessing:",v)
            df = pd.read_feather(v)

            df[cfg['EVENT_DATE']] = pd.to_datetime(df[cfg['EVENT_DATE']])

            df.sort_values(by=[cfg['ID_COL'],cfg['EVENT_DATE'],cfg['EVENT_ID']], ascending=[True, True, True], inplace=True)

            df = self.create_te_ctu(df, cfg)

            file_id = [x for x,y in cfg['DATA_FILE_DICT'].items() if k==y][0]

            save_file = cfg['PREPROCESSED_PATH']+ cfg['t_date'] + '_' + str(file_id) + '.feather'

            df.to_feather(save_file)

            self.data_plus_meta_[self.rack_order_].data_[k] =  save_file
        self.data_plus_meta_[self.rack_order_].config_ = cfg

        del df
        gc.collect()
