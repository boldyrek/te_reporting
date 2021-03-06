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
#!/home/dev/.conda/envs/py365/bin/python3.6
"""
Sampling for heavily imbalanced data class for Té.

@author: Sathish K Lakshmipathy
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""
module_name = 'HighImbalance_Sampling'
version_name = '1.2'

import pandas as pd
import numpy as np

from base_pd.base_sampling_pd import PdBaseSampling as BaseSampling

class HIGHIMBALANCESAMPLING(BaseSampling):
    """ 
    This is a class for imbalance sampling full train data and cross validation folds.
      
    Attributes: 
        rack_order (int): the order of the rack (i.e. 6). 
        data_plus_meta (dict): dictionary contining data and meta data. 
        racks_map (dict): dictionary of all racks and their orders.
    """

    def __init__(self, rack_order, data_plus_meta, racks_map):
        """
        Constructor for high imbalance sampling for Té.

        Parameters: 
            rack_order (int): the order of the rack (i.e. 6). 
            data_plus_meta (dict): dictionary contining data and meta data.
            racks_map (dict): dictionary of all racks and their orders.
        
        Returns: 
            none
        """
        
        super(HIGHIMBALANCESAMPLING, self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)


    def sample_data(self, df, cfg):
        """
        Sampling by downsampling majority class followed by upsampling minority class.
        
        Parameters:
            df (dataframe): dataframe to be down/up sampled.
            cfg (dict): configuration dictionary.
            
        Returns:
            df: sampled dataframe.
        """
        
        df = self.downsample_majority_class(df, cfg)
        df = self.upsample_minority_class(df, cfg)

        return df
