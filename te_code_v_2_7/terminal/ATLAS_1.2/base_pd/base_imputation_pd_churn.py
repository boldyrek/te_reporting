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
import gc
import pandas.tseries.offsets as t_offset

from base.base_imputation import BaseImputation
from util.operation_utils import get_te_ctu_from_cfg

class PdBaseImputation(BaseImputation):
    """
    This is the base class for imputing missing CTUs in data.

    Attributes:
        module_name (str): name of module.
        version_name (str): version name of module.
        rack_order (int): the order of the rack (i.e. 3).
        data_plus_meta (dict): dictionary contining data and meta data.
        racks_map (dict): dictionary of all racks and their orders.
    """

    def __init__(self, module_name, version_name, rack_order, data_plus_meta, racks_map):
        """
        Constructor to initialize the imputation module.

        Attributes:
            module_name (str): name of module.
            version_name (str): version name of module.
            rack_order (int): the order of the rack (i.e. 3).
            data_plus_meta (dict): dictionary contining data and meta data.
            racks_map (dict): dictionary of all racks and their orders.
        """
        super(PdBaseImputation , self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)

    def add_rows_no_event_ctu_train(self, df, cfg):
        """
        Adds an empty row for ctus with no event in them.

        Excludes all new rows added until the first ctu of values from original dataset.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df_ctu: dataframe with one row per ctu for each customer.
            ids: ids of customers with at least one imputed ctu.
        """

        df.sort_values(by=[cfg['ID_COL'], cfg['EVENT_DATE'], cfg['EVENT_ID']], ascending = [True, True, True], inplace = True)

        df_ctu = df.groupby([cfg['ID_COL'], cfg['CTU_COL']], as_index=False).last()

        categories_list = list(df_ctu[cfg['CTU_COL']].unique())
        cat_dtype  = pd.api.types.CategoricalDtype(categories = categories_list, ordered = True)

        df_ctu[cfg['CTU_COL']] = df_ctu[cfg['CTU_COL']].astype(cat_dtype)
        df_ctu = df_ctu.groupby([cfg['ID_COL'], cfg['CTU_COL']], as_index = False).last()
        # change CTU back to ineger
        df_ctu[cfg['CTU_COL']] = df_ctu[cfg['CTU_COL']].astype(int)
        # Sort them by id and period
        df_ctu.sort_values([cfg['ID_COL'], cfg['CTU_COL']], ascending=[True,False], inplace=True)
        df_ctu.set_index([cfg['ID_COL'], cfg['CTU_COL']], inplace=True)
        # Exclude rows that needs backfill
        df_ctu['temp'] = df_ctu.groupby(cfg['ID_COL'])[cfg['EVENT_ID']].fillna(method='ffill')
        df_ctu = df_ctu[~df_ctu['temp'].isnull()]
        # Get imputed row ids
        ids = df_ctu[df_ctu[cfg['EVENT_ID']].isnull()].index

        print ("\nCreated empty ctu rows:")
        print ("No of customers with empty ctus:", df_ctu.index.get_level_values(cfg['ID_COL']).nunique())
        print ("No of empty ctu rows, columns:", df_ctu.shape)

        return df_ctu.drop(columns=['temp']), ids


    # def add_rows_no_event_ctu_predict(self, df, cfg):
    #     """
    #     Adds an empty row for ctus with no event in them.
    #
    #     Exclude all new rows added until the first ctu of values from original dataset.
    #
    #     Parameters:
    #         df (dataframe): dataframe.
    #         cfg (dict): configuration dictionary.
    #
    #     Returns:
    #         df_ctu: dataframe with one row per ctu for each customer.
    #         ids: ids of customers with at least one imputed ctu
    #     """
    #     # Get the last row for each customer
    #     df.sort_values(by=[cfg['ID_COL'],cfg['EVENT_DATE'], cfg['EVENT_ID']], ascending=[True, True, True], inplace=True)
    #     df_ctu = df.groupby(cfg['ID_COL'],as_index=False).last()
    #
    #     # Create empty rows with no values
    #     categories_list = list(df_ctu[cfg['CTU_COL']].unique())
    #     cat_dtype  = pd.api.types.CategoricalDtype(categories = categories_list,  ordered = True)
    #
    #     df_ctu[cfg['CTU_COL']] = df_ctu[cfg['CTU_COL']].astype(cat_dtype)
    #
    #     df_ctu = df_ctu.groupby([cfg['ID_COL'], cfg['CTU_COL']]).last()
    #
    #     # Sort them by id and period
    #     # Exclude rows that needs backfill
    #     df_ctu = df_ctu.sort_index(by = [cfg['ID_COL'], cfg['CTU_COL']],ascending = [True,False])
    #     df_ctu['temp'] = df_ctu.groupby(cfg['CTU_COL'])[cfg['EVENT_DATE']].fillna(method='ffill')
    #     df_ctu = df_ctu[~df_ctu['temp'].isnull()]
    #
    #     ids = df_ctu[df_ctu[cfg['EVENT_ID']].isnull()].index
    #
    #     print ("\nCreated empty ctu rows - prediction, required ctus only:")
    #     print ("No of customers with empty ctus:", df_ctu.index.get_level_values(cfg['ID_COL']).nunique())
    #     print ("No of empty ctu rows, columns:", df_ctu.shape)
    #
    #     return df_ctu.drop(columns=['temp']), ids


    def fill_no_event_ctu(self, df_a, cfg):
        """
        Fills data for each ctu with no events.

        Values depend on the type of the column.
        customer_interactions (CUMSUM_COLS, NLP_COLS, and INTERACTION_EVENT_COLS) columns are filled with zero.
        forward_fill_cols (FACTOR_COLS) columns are filled with zero.
        td columns are filled by adding the number of days equal to the number of days in a ctu.

        Parameters:
            df_a (dataframe): dataframe with one row per ctu for each customer.
            cfg (dict): configuration dictionary.

        Returns:
            df_ctu: dataframe with one row per ctu for each customer.
        """

        df_a.sort_values([cfg['ID_COL'],cfg['CTU_COL']],ascending=[True,False],inplace=True)
        df = pd.DataFrame(index=df_a.index, columns=df_a.columns)
        grp = df_a.groupby([cfg['ID_COL']])

        df[cfg['customer_interactions']] = df_a[cfg['customer_interactions']].fillna(0)

        forward_fill_cols = cfg['CUMMAX_COLS'] + cfg['CUMMEDIAN_COLS'] + cfg['cont_cols'] + cfg['binary_cols']

        df[forward_fill_cols] = grp[forward_fill_cols].fillna(method='ffill')

        customer_agg_cols_1 = [x for x in cfg['customer_agg_cols'] if 'td_' in x] + cfg['FLB_TD_COLS']
        customer_agg_cols_2 = [x for x in cfg['customer_agg_cols'] if 'td_' not in x]

        df[customer_agg_cols_2] = grp[customer_agg_cols_2].fillna(method='ffill')

        df[customer_agg_cols_1] = df_a[customer_agg_cols_1]
        # For td columns, add number of days equal to a period for each empty period
        for i in customer_agg_cols_1:

            df["touchpoint_count"] = np.where(df[i].isnull(), 0, 1)
            df["touchpoint_count"] = df.groupby([cfg['ID_COL']])["touchpoint_count"].cumsum()
            df['temp'] = np.where(df[i].isnull(),cfg['CTU_UNIT_DAY'], 0)
            df['temp'] = df.groupby([cfg['ID_COL'], "touchpoint_count"])['temp'].cumsum()
            df['temp'] = np.where(df[i].isnull(),df['temp'], 0)

            df[i] = df.groupby([cfg['ID_COL']])[i].fillna(method='ffill')
            df[i] = np.where(df[i] > 0, df[i] + df['temp'], df[i])
            df[i] = np.where(df[i] < 0, -1, df[i])


        df[cfg['EVENT_DATE']] = pd.to_datetime(df_a[cfg['EVENT_DATE']])
        df["touchpoint_count"] = np.where(df[cfg['EVENT_DATE']].isnull(), 0, 1)
        df["touchpoint_count"] = df.groupby([cfg['ID_COL']])["touchpoint_count"].cumsum()
        df['temp'] = np.where(df[cfg['EVENT_DATE']].isnull(), cfg['CTU_UNIT_DAY'], 0)
        df['temp'] = df.groupby([cfg['ID_COL'], "touchpoint_count"])['temp'].cumsum()
        df['temp'] = np.where(df[cfg['EVENT_DATE']].isnull(),df['temp'], 0)

        df[cfg['EVENT_DATE']] = df.groupby([cfg['ID_COL']])[cfg['EVENT_DATE']].fillna(method='ffill')
        df[cfg['EVENT_DATE']] = df[cfg['EVENT_DATE']] + pd.TimedeltaIndex(df['temp'], unit='D')

        rem_cols = [x for x in df_a.columns if x not in forward_fill_cols+cfg['customer_agg_cols']+cfg['customer_interactions']+cfg['FLB_TD_COLS']+[cfg['EVENT_DATE']]]
        df[rem_cols] = df_a[rem_cols]

        del grp, df_a
        gc.collect()

        print ("\nAdded empty ctu rows:")
        print ("No of customers in imputation:", df.index.get_level_values(cfg['ID_COL']).nunique())
        print ("No of rows, columns:", df.shape)

        return df.drop(columns=['temp', 'touchpoint_count'])


    def remove_ctu_churn_customers(self, df_ctu, cfg):
        """
        Remove CTUs after the churn event.

        Parameters:
            df_ctu (dataframe): dataframe with one row per ctu for each customer.
            cfg (dict): configuration dictionary.

        Returns:
            df_ctu: dataframe with one row per ctu for each customer.
        """

        df_ctu = df_ctu.sort_index(by = [cfg['ID_COL'], cfg['CTU_COL']],ascending = [True,False])

        df_ctu['temp'] = np.where(df_ctu[cfg['REF_EVENT_COL']].isin([0,1]), 1, np.nan)

        df_ctu['temp'] = df_ctu.groupby(cfg["ID_COL"])['temp'].fillna(method = 'bfill')

        df_ctu = df_ctu[~df_ctu['temp'].isnull()]

        return df_ctu.drop('temp', axis =1)


    def training_imputation(self, df, cfg):
        """
        Imputes  values for empty CTUs

        Parameters:
            df (dataframe): dataframe
            cfg (dict): configuration dictionary

        Returns:
            df: dataframe
        """
        # Adding empty rows
        df_ctu, ids = self.add_rows_no_event_ctu_train(df, cfg)
        #Filling empty rows
        df_ctu = self.fill_no_event_ctu(df_ctu, cfg)
        # Remove empty CTUs for churned customers
        # after the churn event
        df_ctu = self.remove_ctu_churn_customers(df_ctu, cfg)

        df_ctu = df_ctu[df_ctu.index.isin(ids)]

        df.set_index([cfg['ID_COL'],cfg['CTU_COL']],inplace=True)
        # Add imputed_ctu column to df
        df['imputed_ctu'] = 0
        df_ctu['imputed_ctu'] = 1

        df = pd.concat([df,df_ctu], axis=0)

        df = self.create_target(df.reset_index(), cfg)

        if cfg['KEEP_TERMINAL_CTU']:
            # Remove events occured in the termination date
            df = self.remove_terminal_events(df, cfg)
        else:
            # Remove terminal CTU for customers with terminal event
            df = self.remove_terminal_ctu(df, cfg)

        print ("\nNumber of imputed rows after combining:", df['imputed_ctu'].sum())
        print ("Te positives in imputed data:", sum(df[df['imputed_ctu']==1][cfg['TE_TARGET_COL']]))
        print ("Te positives in available data:", sum(df[df['imputed_ctu']==0][cfg['TE_TARGET_COL']]))

        del df_ctu
        gc.collect()

        return df


    def prediction_imputation(self, df, cfg):
        """
        Imputes  values for empty CTUs

        Parameters:
            df (dataframe): dataframe
            cfg (dict): configuration dictionary

        Returns:
            df: dataframe
        """

        #Adding empty rows
        df_ctu, ids = self.add_rows_no_event_ctu_predict(df, cfg)

        df_ctu = self.remove_ctu_churn_customers(df_ctu, cfg)
        #Filling empty rows
        df_ctu = self.fill_no_event_ctu(df_ctu, cfg)

        df_ctu = df_ctu[df_ctu.index.isin(ids)]

        df.set_index([cfg['ID_COL'],cfg['CTU_COL']],inplace=True)

        # Add imputed_ctu column to df
        df['imputed_ctu'] = 0
        df_ctu['imputed_ctu'] = 1

        df = pd.concat([df,df_ctu], axis=0)

        df = self.create_target(df.reset_index(), cfg)

        print ("\nNumber of imputed rows after combining:", df['imputed_ctu'].sum())
        print ("Te positives in imputed data:", sum(df[df['imputed_ctu']==1][cfg['TE_TARGET_COL']]))
        print ("Te positives in available data:", sum(df[df['imputed_ctu']==0][cfg['TE_TARGET_COL']]))

        del df_ctu
        gc.collect()

        return df


    def create_target(self, df, cfg):
        """
        Creates target column based on TE performance window (TE_WINDOW_DAY).

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df: dataframe.
        """

        df = df.sort_values(by=[cfg['ID_COL'],cfg['CTU_COL']], ascending=[True,False])

        df['temp_date'] = df[df[cfg['REF_EVENT_COL']]==1][cfg['EVENT_DATE']]

        df['temp_date'] = df.groupby(cfg['ID_COL'])['temp_date'].shift(-1)

        df['temp_date'] = df.groupby(cfg['ID_COL'])['temp_date'].bfill()

        df[cfg['TE_TARGET_COL']] = np.where((df['temp_date'] - df[cfg['EVENT_DATE']]).dt.days < cfg['TE_WINDOW_DAY'], 1, 0)

        df = df[df[cfg['CTU_COL']] != -1]

        print ("\nCreated target column:")
        print("Te window length:", cfg['TE_WINDOW_DAY'])
        print ("No of customers with positives:", df[df[cfg['TE_TARGET_COL']]==1][cfg['ID_COL']].nunique())
        print ("Total positives:", df[cfg['TE_TARGET_COL']].sum())

        return df.drop(columns=['temp_date'])


    def filter_data_by_date(self, df, cfg):
        """
        Excludes data after the TRAIN_END_DATE

        Removes rows for which the performance window is less than TE_WINDOW_DAY

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df: dataframe.
        """
        # Use data until date mentioned below to train the model and validate on the period after that
        if cfg['TRAIN_END_DATE']:
            print("Include data before date:", cfg['TRAIN_END_DATE'])
            df = df[df[cfg['EVENT_DATE']] <= pd.to_datetime(cfg['TRAIN_END_DATE'])]

        print ("\nAfter applying END_DATE filter:")
        print ("No of customers:", df[cfg['ID_COL']].nunique())
        print ("No of rows, columns:", df.shape)

        return df


    def remove_terminal_ctu(self, df, cfg):
        """
        Removes the rows corresponding to the CTU in which the account churns

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df: dataframe.
        """

        df['temp_ctu'] = np.where(df[cfg['REF_EVENT_COL']]==1, df[cfg['CTU_COL']], np.nan)

        df['temp_ctu'] = df.groupby(df[cfg['ID_COL']])['temp_ctu'].transform('min').fillna(-999)

        df = df[df[cfg['CTU_COL']] > df['temp_ctu']]

        return df.drop(columns = ['temp_ctu'])


    def remove_terminal_events(self, df, cfg):
        """
        Removes data for events in the churn date

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df: dataframe.
        """

        df['temp'] = df.groupby([cfg['ID_COL']])[cfg['EVENT_DATE']].transform('max')
        df['temp_2'] = np.where(df['temp'] == df[cfg['EVENT_DATE']], 1,0)
        ids = df[df[cfg['REF_EVENT_COL']] == 1][cfg['ID_COL']].unique()
        churners = df[df[cfg['ID_COL']].isin(ids)]
        non_churners = df[~df[cfg['ID_COL']].isin(ids)]
        churners = churners[churners['temp_2']!=1]
        df_2 = pd.concat([churners, non_churners])

        return df_2.drop(columns = ['temp','temp_2'])
