import numpy as np
import pandas as pd
import random
import gc
import pandas.tseries.offsets as t_offset
import datetime

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
    #     For periods with no dynamic event in them, add an empty row
    #     For prediction set, we dont have to forward fill everywhere, only for the most recent period
    #     """
    #     # Get the last row for each customer
    #     df.sort_values([cfg['ID_COL'], cfg['EVENT_DATE']], ascending=[True,True], inplace=True)
    #     df_ctu = df.groupby(cfg['ID_COL'],as_index=False).last()
    #
    #     # Create empty rows with no values
    #     df_ctu[cfg['CTU_COL']] = df_ctu[cfg['CTU_COL']].astype('category',ordered=True)
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


    def training_imputation(self, df, cfg):
        """
        Imputing values for empty CTUs - training only

        :INPUT: df read after adding additional Columns
        :OUTPUT: df with rows imputed for empty CTUs
        """

        df_ctu, ids = self.add_rows_no_event_ctu_train(df, cfg)

        df_ctu = self.fill_no_event_ctu(df_ctu, cfg)

        df_ctu = df_ctu[df_ctu.index.isin(ids)]

        df.set_index([cfg['ID_COL'],cfg['CTU_COL']],inplace=True)
        # Add imputed_ctu column to df
        df['imputed_ctu'] = 0
        df_ctu['imputed_ctu'] = 1

        df = pd.concat([df,df_ctu], axis=0)
        df = self.create_target(df.reset_index(), cfg)
        ##TODO: in churn version, keeping or removing terminal ctu is implemented here
        ##in repurchse. check the logic there and decide if we need to remove
        ##reference events (or all events that has occured in that date) from data
        ##(check the churn version if want to implement remove_terminal_ctu and remove_terminal_events)

        print ("Number of imputed rows after combining:", df['imputed_ctu'].sum())
        print ("Te positives in imputed data:", sum(df[df['imputed_ctu']==1][cfg['TE_TARGET_COL']]))
        print ("Te positives in available data:", sum(df[df['imputed_ctu']==0][cfg['TE_TARGET_COL']]))

        del df_ctu
        gc.collect()
        return df

    # def prediction_imputation(self, df, cfg):
    #     """
    #     Imputing values for empty CTUs - prediction only
    #
    #     :INPUT: df read after adding additional Columns
    #     :OUTPUT: df with rows imputed for empty CTUs
    #     """
    #
    #     print ("Adding empty rows....")
    #     df_ctu, ids = self.add_rows_no_event_ctu_predict(df, cfg)
    #
    #     print ("Filling empty rows....")
    #     df_ctu = self.fill_no_event_ctu(df_ctu, cfg)
    #
    #     df_ctu = df_ctu[df_ctu.index.isin(ids)]
    #     print ("Number of imputed rows:", df_ctu.shape[0], len(ids))
    #
    #     df.set_index([cfg['ID_COL'],cfg['CTU_COL']],inplace=True)
    #
    #     # Add imputed_ctu column to df
    #     df['imputed_ctu'] = 0
    #     df_ctu['imputed_ctu'] = 1
    #
    #     df = pd.concat([df,df_ctu], axis=0)
    #
    #     df = self.create_target(df.reset_index(), cfg)
    #
    #     print ("Number of imputed rows after combining:", df['imputed_ctu'].sum())
    #     print ("Te positives in imputed data:", sum(df[df['imputed_ctu']==1][cfg['TE_TARGET_COL']]))
    #     print ("Te positives in available data:", sum(df[df['imputed_ctu']==0][cfg['TE_TARGET_COL']]))
    #
    #     #TODO: Get only the maximum number of rows required for next aggregation
    #     #TODO:df = df[df.index.get_level_values(cfg['CTU_COL']) <= cfg['n_hist_periods+1]
    #     #TODO:print (cfg_dict['te_parameters']()['agg_window'])
    #
    #     del df_ctu
    #     gc.collect()
    #
    #     return df


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
