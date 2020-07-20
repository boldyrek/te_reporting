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

class PdBaseImputation(BaseImputation):

    def __init__(self, module_name, version_name, rack_order, data_plus_meta, racks_map):
        """
        Constructor to initialize the enriching module.

        :param
        	module_name: the name of the imputing module.
            version_name: the version of imputing module.
            dict_metadata:
            order:
        :return none
        :raises none
        """
        super(PdBaseImputation , self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)

    def add_rows_no_event_ctu_train(self, df, cfg):
        """
        For periods with no dynamic event in them, add an empty row
        Exclude all new rows added until the first period of values from original dataset
        No need to backfill information for periods that do not have information
        """
        # Create empty rows with no values
        df.sort_values(by=[cfg['ID_COL'],cfg['EVENT_DATE'], cfg['EVENT_ID']], ascending=[True, True, True], inplace=True)
        df_ctu = df.groupby([cfg['ID_COL'],cfg['CTU_COL']],as_index=False).last()
        
        categories_list = list(df_ctu[cfg['CTU_COL']].unique())
        cat_dtype  = pd.api.types.CategoricalDtype(categories = categories_list,  ordered = True)
        
        df_ctu[cfg['CTU_COL']] = df_ctu[cfg['CTU_COL']].astype(cat_dtype)
        
#         df_ctu[cfg['CTU_COL']] = df_ctu[cfg['CTU_COL']].astype('category')
        df_ctu = df_ctu.groupby([cfg['ID_COL'],cfg['CTU_COL']], as_index=False).last()

        # Sort them by id and period
        df_ctu.sort_values([cfg['ID_COL'],cfg['CTU_COL']],ascending=[True,False],inplace=True)
        df_ctu.set_index([cfg['ID_COL'],cfg['CTU_COL']],inplace=True)

        # Exclude rows that needs backfill
        df_ctu['temp'] = df_ctu.groupby(cfg['ID_COL'])[cfg['EVENT_ID']].fillna(method='ffill')
        df_ctu = df_ctu[~df_ctu['temp'].isnull()]

        # Get imputed row ids
        ids = df_ctu[df_ctu[cfg['EVENT_ID']].isnull()].index

        print ("\nCreated empty ctu rows:")
        print ("No of customers with empty ctus:", df_ctu.index.get_level_values(cfg['ID_COL']).nunique())
        print ("No of empty ctu rows, columns:", df_ctu.shape)

        return df_ctu.drop(columns=['temp']), ids

    def add_rows_no_event_ctu_predict(self, df, cfg):
        """
        For periods with no dynamic event in them, add an empty row
        For prediction set, we dont have to forward fill everywhere, only for the most recent period
        """
        # Get the last row for each customer
        df.sort_values(by=[cfg['ID_COL'],cfg['EVENT_DATE'], cfg['EVENT_ID']], ascending=[True, True, True], inplace=True)
        df_ctu = df.groupby(cfg['ID_COL'],as_index=False).last()

        # Create empty rows with no values
#         df_ctu[cfg['CTU_COL']] = df_ctu[cfg['CTU_COL']].astype('category',ordered=True)
               
        categories_list = list(df_ctu[cfg['CTU_COL']].unique())
        cat_dtype  = pd.api.types.CategoricalDtype(categories = categories_list,  ordered = True)
        
        df_ctu[cfg['CTU_COL']] = df_ctu[cfg['CTU_COL']].astype(cat_dtype)

        df_ctu = df_ctu.groupby([cfg['ID_COL'], cfg['CTU_COL']]).last()

        # Sort them by id and period
        # Exclude rows that needs backfill
        df_ctu = df_ctu.sort_index(by = [cfg['ID_COL'], cfg['CTU_COL']],ascending = [True,False])
        df_ctu['temp'] = df_ctu.groupby(cfg['CTU_COL'])[cfg['EVENT_DATE']].fillna(method='ffill')
        df_ctu = df_ctu[~df_ctu['temp'].isnull()]

        ids = df_ctu[df_ctu[cfg['EVENT_ID']].isnull()].index

        print ("\nCreated empty ctu rows - prediction, required ctus only:")
        print ("No of customers with empty ctus:", df_ctu.index.get_level_values(cfg['ID_COL']).nunique())
        print ("No of empty ctu rows, columns:", df_ctu.shape)

        return df_ctu.drop(columns=['temp']), ids

    def fill_no_event_ctu(self, df_a, cfg):
        """
        Fill data for each period with no events
        Values depend on the type of the column
        td columns are filled by adding the number of days equal to the number of days in a period

        Column groups are obtained from the get_column_groups in the operation_util.py file
        """

        df = pd.DataFrame(index=df_a.index, columns=df_a.columns)
        grp = df_a.groupby([cfg['ID_COL']]).fillna(method='ffill')
        # Customer interaction columns inidividual event columns
        # they are imputed with 0s
        
        df[cfg['customer_interactions']] = df_a[cfg['customer_interactions']].fillna(0)
        # Columns where the last true value is filled are grouped under forward filled Column.
        # It includes static columns, binary columns and customer information
        forward_fill_cols = cfg['CUMMAX_COLS'] + cfg['CUMMEDIAN_COLS'] + cfg['cont_cols'] + cfg['binary_cols']

        df[forward_fill_cols] = grp[forward_fill_cols].fillna(method='ffill')
        # For the imput
        customer_agg_cols_1 = [x for x in cfg['customer_agg_cols'] if 'td_' in x]
        customer_agg_cols_2 = [x for x in cfg['customer_agg_cols'] if 'td_' not in x]

        df[customer_agg_cols_2] = grp[customer_agg_cols_2].fillna(method='ffill')

        df[customer_agg_cols_1] = df_a[customer_agg_cols_1]
        # For td columns, add number of days equal to a period for each empty period
        print ("Adding td values for empty periods...")
        for i in customer_agg_cols_1:
            
            df["touchpoint_count"] = np.where(df[i].isnull(), 0, 1)
            df["touchpoint_count"] = df.groupby([cfg['ID_COL']])["touchpoint_count"].cumsum()
            df['temp'] = np.where(df[i].isnull(),cfg['TE_WINDOW_DAY'], 0)
            df['temp'] = df.groupby([cfg['ID_COL'], "touchpoint_count"])['temp'].cumsum()
            df['temp'] = np.where(df[i].isnull(),df['temp'], 0)

            df[i] = df.groupby([cfg['ID_COL']])[i].fillna(method='ffill')
            df[i] = np.where(df[i] > 0, df[i] + df['temp'], df[i])
            # print(df[i])
            # exit()

        df[cfg['EVENT_DATE']] = pd.to_datetime(df_a[cfg['EVENT_DATE']])
        # print(df[cfg['EVENT_DATE']])
        df["touchpoint_count"] = np.where(df[cfg['EVENT_DATE']].isnull(), 0, 1)
        df["touchpoint_count"] = df.groupby([cfg['ID_COL']])["touchpoint_count"].cumsum()
        df['temp'] = np.where(df[cfg['EVENT_DATE']].isnull(),cfg['TE_WINDOW_DAY'], 0)
        df['temp'] = df.groupby([cfg['ID_COL'], "touchpoint_count"])['temp'].cumsum()
        df['temp'] = np.where(df[cfg['EVENT_DATE']].isnull(),df['temp'], 0)

        df[cfg['EVENT_DATE']] = df.groupby([cfg['ID_COL']])[cfg['EVENT_DATE']].fillna(method='ffill')
        df[cfg['EVENT_DATE']] = df[cfg['EVENT_DATE']] + pd.TimedeltaIndex(df['temp'], unit='D')
        # print(df[cfg['EVENT_DATE']])
        rem_cols = [x for x in df_a.columns if x not in forward_fill_cols+cfg['customer_agg_cols']+cfg['customer_interactions']+[cfg['EVENT_DATE']]]
        df[rem_cols] = df_a[rem_cols]

        del grp, df_a
        gc.collect()
    
        print ("\nAdded empty ctu rows:")
        print ("No of customers in imputation:", df.index.get_level_values(cfg['ID_COL']).nunique())
        print ("No of rows, columns:", df.shape)

        return df.drop(columns=['temp', 'touchpoint_count'])

    def training_imputation(self, df, cfg):
        """
        Imputing values for empty CTUs - training only

        :INPUT: df read after adding additional Columns
        :OUTPUT: df with rows imputed for empty CTUs
        """
        # Adding empty rows
        df_ctu, ids = self.add_rows_no_event_ctu_train(df, cfg)
        print("5:",len(df.columns) - len(list(set(df.columns))))

        #Filling empty rows
        df_ctu = self.fill_no_event_ctu(df_ctu, cfg)
        print("6:",len(df.columns) - len(list(set(df.columns))))

        df_ctu = df_ctu[df_ctu.index.isin(ids)]

        df.set_index([cfg['ID_COL'],cfg['CTU_COL']],inplace=True)

        # Add imputed_ctu column to df
        df['imputed_ctu'] = 0
        df_ctu['imputed_ctu'] = 1

        df = pd.concat([df,df_ctu], axis=0)

        print("7:",len(df.columns) - len(list(set(df.columns))))
        df = self.create_target(df.reset_index(), cfg)

        print ("\nNumber of imputed rows after combining:", df['imputed_ctu'].sum())
        print ("Te positives in imputed data:", sum(df[df['imputed_ctu']==1][cfg['TE_TARGET_COL']]))
        print ("Te positives in available data:", sum(df[df['imputed_ctu']==0][cfg['TE_TARGET_COL']]))

        del df_ctu
        gc.collect()

        return df

    def prediction_imputation(self, df, cfg):
        """
        Imputing values for empty CTUs - prediction only

        :INPUT: df read after adding additional Columns
        :OUTPUT: df with rows imputed for empty CTUs
        """

        #Adding empty rows
        df_ctu, ids = self.add_rows_no_event_ctu_predict(df, cfg)

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

        #TODO: Get only the maximum number of rows required for next aggregation
        #TODO:df = df[df.index.get_level_values(cfg['CTU_COL']) <= cfg['n_hist_periods+1]
        #TODO:print (cfg_dict['te_parameters']()['agg_window'])

        del df_ctu
        gc.collect()

        return df

    def get_ctu_end_func(self, cfg):
        """
        Depending on the input CTU, the end of CTU is required to get the target happening
        after the current CTU

        This function returns the end of the CTU date which added to the EVENT_DATE and the purchase window follow that
        """

        if cfg['CTU_UNIT_NAME'] in cfg["TE_CTU_UNITS"]:
            if cfg['CTU_UNIT_NAME'] == 'week':
                return t_offset.Week(weekday=6)
            elif cfg['CTU_UNIT_NAME'] == 'day':
                return t_offset.Week(weekday=0)
            else:
                func_name = cfg['CTU_UNIT_NAME'].title()+'End'
                return getattr(t_offset,func_name)(0)
        else:
            print("CTU should be one of", cfg["TE_CTU_UNITS"])


    def create_target(self, df, cfg):
        """
        Create target column based on CTU te_window_size

        :INPUT: df with CTU size and reference event
        :OUTPUT: df with target column
        """

        df['temp_date'] = df[df[cfg['REF_EVENT_COL']]==1][cfg['EVENT_DATE']]

        df['temp_date'] = df.groupby(cfg['ID_COL'])['temp_date'].shift(-1)

        df['temp_date'] = df.groupby(cfg['ID_COL'])['temp_date'].bfill()

        df[cfg['TE_TARGET_COL']] = np.where((df['temp_date'] - (df[cfg['EVENT_DATE']] + self.get_ctu_end_func(cfg))).dt.days < cfg['TE_WINDOW_DAY'], 1, 0)

        print ("\nCreated target column:")
        print("Te window length:", cfg['TE_WINDOW_DAY'])
        print ("No of customers with positives:", df[df[cfg['TE_TARGET_COL']]==1][cfg['ID_COL']].nunique())
        print ("Total positives:", df[cfg['TE_TARGET_COL']].sum())

        return df.drop(columns=['temp_date'])
