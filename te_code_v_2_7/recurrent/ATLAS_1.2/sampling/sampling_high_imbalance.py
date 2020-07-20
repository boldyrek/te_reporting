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

    def __init__(self, rack_order, data_plus_meta, racks_map):
        """
        Constructor for high imbalance sampling for Té
        :param: sampling ratios
    			config: Té config
        :return: none
        :raises: none
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
