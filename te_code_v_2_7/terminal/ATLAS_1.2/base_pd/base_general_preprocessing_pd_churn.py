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
import gc
import datetime as dt

from base.base_general_preprocessing import BaseGeneralPreprocess

class PdBaseGeneralPreprocess(BaseGeneralPreprocess):
    """
    This is the base class for conducting general preprocessing on data.

    Attributes:
        module_name (str): name of module
        version_name (str): version name of module
        rack_order (int): the order of the rack (i.e. 1).
        data_plus_meta (dict): dictionary contining data and meta data.
        racks_map (dict): dictionary of all racks and their orders.
    """

    def __init__(self, module_name, version_name, rack_order, data_plus_meta, racks_map):
        """
        Constructor to initialize the general preprocessing module.

        Attributes:
            module_name (str): name of module
            version_name (str): version name of module
            rack_order (int): the order of the rack (i.e. 1).
            data_plus_meta (dict): dictionary contining data and meta data.
            racks_map (dict): dictionary of all racks and their orders.
        """
        super(PdBaseGeneralPreprocess, self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)

    def clean_data(self, df, cfg):
        """
        Performs basic data cleaning

        This function replaces some characters with appropriate ones,
        drops column with duplicate names,
        drop columns in drop column list in config

        Parameters:
            df (dataframe): dataframe
            cfg (dict): configuration dictionary

        Returns:
            df: dataframe
        """

        df.columns = [x.replace("/","_") for x in df.columns]
        df.columns = [x.replace("[","") for x in df.columns]
        df.columns = [x.replace("]","") for x in df.columns]

        print ("Dataframe shape:", df.shape)

        df = df.loc[:, ~df.columns.duplicated()]

        print ("Number of columns after removing duplicates:", df.shape[1])

        df = df.drop(columns = cfg['COLS_TO_DROP'], axis=1 )

        print ("Number of columns after dropping cols_to_drop:", df.shape[1])

        return df


    def flb_imputation(self, df, cfg):
        """
        Imputes missing values based on config lists (impute with zero or fill forward)

        Parameters:
            df (dataframe): dataframe
            cfg (dict): configuration dictionary

        Returns:
            df: dataframe
        """

        df[cfg['FFILL_COLS']] = df.groupby(cfg['ID_COL'])[cfg['FFILL_COLS']].fillna(method='ffill').fillna(0)

        df[cfg['IMPUTE_ZERO_COLS']] = df[cfg['IMPUTE_ZERO_COLS']].fillna(0)

        return df


    def remove_negative_tenure(self, df, cfg):
        """
        Removes rows of data where tenure is negative: removes the events that occured before activation date

        Parameters:
            df (dataframe): dataframe
            cfg (dict): configuration dictionary

        Returns:
            df: dataframe
        """

        df = df[df['CAI_TELCO_TENURE_DAYS'] >= 0]

        return df


    def remove_customers_with_short_journey(self, df, cfg):
        """
        Removes customers whose journey is less than a threshold (e.g., 90 days)

        Parameters:
            df (dataframe): dataframe
            cfg (dict): configuration dictionary

        Returns:
            df: dataframe
        """

        df_train_end = df
        if cfg['TRAIN_END_DATE']:
            df_train_end = df_train_end[df_train_end[cfg['EVENT_DATE']] <= pd.to_datetime(cfg['TRAIN_END_DATE'])]

        df_train_end['temp'] = df_train_end.groupby(cfg['ID_COL'])['CAI_TELCO_TENURE_DAYS'].transform(max)
        ids = df_train_end[df_train_end.temp <= cfg['SHORT_TENURE']][cfg['ID_COL']].unique()

        df = df[df[cfg['ID_COL']].isin(ids)==False]

        return df.drop(columns = ['temp'])


    def truncate_churn_journey(self, df, cfg):
        """
        Excludes events after churn date

        Parameters:
            df (dataframe): dataframe
            cfg (dict): configuration dictionary

        Returns:
            df: dataframe
        """

        df['temp_churn_event'] = np.where(df[cfg['REF_EVENT_COL']] == 1, 1, np.nan)
        df.sort_values([cfg['ID_COL'], cfg['EVENT_DATE']], inplace =True)
        df['temp_churn_event'] = df.groupby([cfg['ID_COL']])['temp_churn_event'].fillna(method='bfill')
        df['temp'] = df.groupby([cfg['ID_COL']])['temp_churn_event'].transform(max)
        df['temp_churn_event'] = np.where(df['temp'] == 1, df['temp_churn_event'], 999)
        df = df[~df['temp_churn_event'].isnull()]

        return df.drop(columns = ['temp_churn_event'])


    def create_ref_event_cols(self, df, cfg):
        """
        Creates an extra column for reference event

        Parameters:
            df (dataframe): dataframe
            cfg (dict): configuration dictionary

        Returns:
            df: dataframe
        """

        df[cfg['REF_EVENT_COL']] = np.where(df[cfg['REFERENCE_EVENT']] == 1, 1, 0)

        print ("\nCreated reference event counts")
        print ("No of customers:", df[cfg['ID_COL']].nunique())
        print ("No of rows, columns:", df.shape)

        return df.drop(columns = [cfg['REFERENCE_EVENT']])


    def create_yr_ctu_cols(self, df, cfg):
        """
        Creates one column for year of event, one for ctu unit name (e.g., week) and one for year_ctu_unit (week; e.g., 201904)

        The year-ctu column created here will be used later to assign ctus to each row of data

        Parameters:
            df (dataframe): dataframe
            cfg (dict): configuration dictionary

        Returns:
            df: dataframe
        """

        cfg['CTU_UNIT_NAME'] = [i for i in cfg["TE_CTU_UNITS"] if i in cfg['CTU_UNIT']][0]
        yr_ctu = 'yr_'+ cfg['CTU_UNIT_NAME']

        df['year'] = (df[cfg['EVENT_DATE']]).dt.year
        df[cfg['CTU_UNIT_NAME']] = df[cfg['EVENT_DATE']].apply(lambda x: getattr(x,cfg['CTU_UNIT_NAME']))

        df[yr_ctu] = df['year'].astype(str)+('0' + df[cfg['CTU_UNIT_NAME']].astype(str)).str[-2:]
        df[yr_ctu] = pd.to_numeric(df[yr_ctu])

        return df


    def create_semantic_interaction_cols(self, df, cfg):
        """
        Creates column groups based on type of interactions

        Parameters:
            df (dataframe): dataframe
            cfg (dict): configuration dictionary

        Returns:
            df: dataframe
        """

        inbound_cols = [x for x in cfg['INBOUND_INTERACTIONS'] if x in df.columns]
        outbound_cols = [x for x in cfg['OUTBOUND_INTERACTIONS'] if x in df.columns]
        positive_interactions = [x for x in cfg['POSITIVE_INTERACTIONS'] if x in df.columns]
        negative_interactions = [x for x in cfg['NEGATIVE_INTERACTIONS'] if x in df.columns]


        df['event_inbound_interactions'] = df[inbound_cols].sum(axis=1)
        df['event_outbound_interactions'] = df[outbound_cols].sum(axis=1)
        df['event_positive_interactions'] = df[positive_interactions].sum(axis=1)
        df['event_negative_interactions'] = df[negative_interactions].sum(axis=1)

        print ("\nCreated interaction events:")
        print ("No of customers:", df[cfg['ID_COL']].nunique())
        print ("No of rows, columns:", df.shape)

        return df


    def create_interaction_td_cols(self, df, cfg):
        """
        Creates td (i.e., time since last event) columns

        Parameters:
            df (dataframe): dataframe
            cfg (dict): configuration dictionary

        Returns:
            df: dataframe
        """

        dynamic_cols = [x for x in cfg['TD_COLS'] if x in df.columns]

        for i in dynamic_cols:

            df['interaction_flag'] = np.where(df[i] > 0, 1, 0)

            df['interaction_count'] = df.groupby(cfg['ID_COL'])['interaction_flag'].cumsum()

            df["grp_start_date"] = df.groupby([cfg['ID_COL'],'interaction_count'])[cfg['EVENT_DATE']].transform("min")

            df['td_last_'+i] = (df[cfg['EVENT_DATE']] - df['grp_start_date']).dt.days.fillna(0)

            # Make the value to -1 where the interaction has not happened yet
            df['td_last_'+ i] = np.where(df['interaction_count'] == 0, -1, df['td_last_'+ i])

        df = df.drop(columns=['interaction_flag','interaction_count','grp_start_date'])

        print ("\nCreated td columns for all interactions:")
        print ("No of customers:", df[cfg['ID_COL']].nunique())
        print ("No of rows, columns:", df.shape)

        return df


    def filter_past_sparse_data(self, df, cfg):
        """
        Includes only the last n (i.e. 3) years of data

        Parameters:
            df (dataframe): dataframe
            cfg (dict): configuration dictionary

        Returns:
            df: dataframe
        """

        start_date = (pd.to_datetime('today') - pd.DateOffset(years=cfg['N_YEARS']))

        st_dt = df[df[cfg['EVENT_DATE']] >= start_date][cfg['EVENT_DATE']].min()

        agg_months = cfg['AGG_WINDOW_MONTH']

        te_months = cfg['TE_WINDOW_MONTH']

        s_date = st_dt - pd.DateOffset(months=agg_months + te_months)

        df = df[df[cfg['EVENT_DATE']] >= s_date]

        print (f"\nRemoved data older than {cfg['N_YEARS']} years:")
        print ("No of customers:", df[cfg['ID_COL']].nunique())
        print ("No of rows, columns:", df.shape)

        return df.reset_index()


## If FLB has sum over journey columns, there is no need to call this function in run()
    def create_event_cols(self, df, cfg):
        """
        This function deaggregates the journey aggregations that the spark pipeline does to create event level columns
        Also retains the columns with aggregations
        Need to be run only until the spark pipeline is changed

        INPUT: data df from spark pipeline
        OUTPUT: df with De-aggregated and aggregated columns
        """

        # aggregate events to cumulative sum
        agg_cols = [x for x in cfg['EVENT_COLS'] if x in df.columns]
        temp = df.groupby(cfg['ID_COL'])[agg_cols].cumsum()

        # Rename columns with aggregations
        temp.columns = ['expanding_'+x for x in temp.columns]

        df = pd.concat([df, temp], axis=1)

        del temp
        gc.collect()

        print ("\nAggregated and preserved columns aggregated")
        print ("No of customers:", df[cfg['ID_COL']].nunique())
        print ("No of rows, columns:", df.shape)

        return df


## If FLB has tenure days, there is no need to call this function in run()
    def create_td_cols(self, df, cfg):
        """
        Creates temporal displacement features (td)

        td - time from the first event (already exists)

        Parameters:
        df: Pandas dataframe, datset
        ref_events: List, reference events

        Output:
        Pandas dataframe, modified dataset
        """

        df["journey_start"] = df.groupby(cfg['ID_COL'])[cfg['EVENT_DATE']].transform("min")

        # Calculate time since last reference event and time to next reference event
        # Use -999 as placeholder for timeline 0 and timelines that don't end
        df["td_start"] = (df[cfg['EVENT_DATE']] - df["journey_start"]).dt.days

        df = df.drop(['journey_start'], axis=1)

        print ("\nCreated td columns for reference event:")
        print ("No of customers:", df[cfg['ID_COL']].nunique())
        print ("No of rows, columns:", df.shape)

        return df
