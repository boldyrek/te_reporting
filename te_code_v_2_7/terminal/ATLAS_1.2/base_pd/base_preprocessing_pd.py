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
import numpy as np
import pandas as pd
import random

from base.base_preprocessing import BasePreprocessing

class PdBasePreprocessing(BasePreprocessing):
    """
    This is the base class for conducting preprocessing on data.

    Attributes:
        module_name (str): name of module
        version_name (str): version name of module
        rack_order (int): the order of the rack (i.e. 2).
        data_plus_meta (dict): dictionary contining data and meta data.
        racks_map (dict): dictionary of all racks and their orders.
    """

    def __init__(self, module_name, version_name, rack_order, data_plus_meta, racks_map):
        """
        Constructor to initialize the preprocessing module.

        Attributes:
            module_name (str): name of module
            version_name (str): version name of module
            rack_order (int): the order of the rack (i.e. 2).
            data_plus_meta (dict): dictionary contining data and meta data.
            racks_map (dict): dictionary of all racks and their orders.
        """
        super(PdBasePreprocessing, self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)

    def get_yr_ctu_tuples(self, cfg, s_yr, e_yr):
        """
        Creates a tuple of year and number of CTU units per years based on the type of CTU (quarter, month, week, day).

        Parameters:
            cfg (dict): configuration dictionary.
            s_yr (int): year of the earliest data point.
            e_yr (int): year of the latest data point.

        Returns:
            list: list of tuples showing each year of data and total number of weeks in that year
        """

        if cfg['CTU_UNIT_NAME'] in cfg['TE_WINDOW_UNITS_DICT'].keys():
            return [(i, cfg['TE_WINDOW_UNITS_DICT'][cfg['CTU_UNIT_NAME']]) for i in np.arange(s_yr, e_yr + 1, 1)]
        elif 'day' in cfg['CTU_UNIT_NAME']:
            return [(i, 366 if i % 4==0 else 365) for i in np.arange(s_yr, e_yr+1,1)]
        else:
            print ("CTU unit not set up. Please choose one of", cfg["TE_CTU_UNITS"])


    def get_year_ctu(self, start_date, end_date, cfg):
        """
        Creates all possible year_ctu_units (e.g., year_weeks) for each event of a customer.

        Parameters:
            start_date (date): earliest date in data.
            end_date (date): latest date in data.
            cfg (dict): configuration dictionary.

        Returns:
            df: dataframe with only one column which has all possible year-ctu_units (e.g., year_week) in data.
        """

        s_yr = start_date.year
        s_ctu = getattr(start_date, cfg['CTU_UNIT_NAME'])

        e_yr = end_date.year
        e_ctu = getattr(end_date, cfg['CTU_UNIT_NAME'])

        p_st = int(str(s_yr)+('0'+str(s_ctu))[-2:])
        p_end = int(str(e_yr)+('0'+str(e_ctu))[-2:])

        yr_ctu = 'yr_' + cfg['CTU_UNIT_NAME']
        ctu_tuple_list = self.get_yr_ctu_tuples(cfg, s_yr, e_yr)

        ls = [[int(str(yr)+('0'+str(z))[-2:]) for z in np.arange(1, ctu + 1, 1)] for yr, ctu in ctu_tuple_list]

        df = pd.DataFrame(columns=[yr_ctu])
        df[yr_ctu] = [item for sublist in ls for item in sublist]

        # Remove the year ctus not in the data
        df = df[(df[yr_ctu] >= p_st) & (df[yr_ctu] <= p_end)]

        return df.sort_values(by = yr_ctu,ascending=False)

    def create_te_ctu(self, df, cfg):
        """
        Creates ctu for each event of a customer.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df: dataframe.
        """

        ctu_col = 'yr_'+ cfg['CTU_UNIT_NAME']
        # The last CTU should always be where the training data ends
        if not cfg['TRAIN_END_DATE']:
            cfg["last_date"] = df[cfg['EVENT_DATE']].max()
        else:
            cfg["last_date"] = pd.to_datetime(cfg['TRAIN_END_DATE'])
        # Create period_df
        period_df = self.get_year_ctu(df[cfg['EVENT_DATE']].min(), cfg["last_date"], cfg)

        # Assign period number for each month
        cfg["CTU_OFFSET"] = int(''.join(filter(str.isdigit, cfg['CTU_UNIT'])))

        period_df[cfg['CTU_COL']] = 0
        period_df[cfg['CTU_COL']][::cfg["CTU_OFFSET"]] = 1
        period_df[cfg['CTU_COL']] = period_df[cfg['CTU_COL']].cumsum()-1
        # Get the period column for data df
        df = pd.merge(df, period_df, on=ctu_col, how = 'left')
        #fill missing values with -1 (Full labels)
        df[cfg['CTU_COL']].fillna(-1, inplace= True)

        print ("\n'ctu' column added")
        print ("No of customers:", df[cfg['ID_COL']].nunique())
        print ("No of rows, columns:", df.shape)

        return df
