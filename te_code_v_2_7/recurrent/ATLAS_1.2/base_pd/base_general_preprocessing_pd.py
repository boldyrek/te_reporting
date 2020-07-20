import numpy as np
import pandas as pd
import gc
import datetime as dt

from base.base_general_preprocessing import BaseGeneralPreprocess

class PdBaseGeneralPreprocess(BaseGeneralPreprocess):

    def __init__(self, module_name, version_name, rack_order, data_plus_meta, racks_map):
        """
        Constructor to initialize the imputing module.

        :param
        	module_name: the name of the imputing module.
            version_name: the version of imputing module.
            dict_metadata:
            order:
        :return none
        :raises none
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


        df = df.loc[:, ~df.columns.duplicated()]

        df = df.drop(columns = cfg['COLS_TO_DROP'], axis=1 )

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


    def check_reference_event(self, df, cfg):
        """
        This function:
        Makes sure that each customer has atleast one reference event of interest

        :INPUT: dataframe and the referenc event list
        :OUTPUT: dataframe where the customer has atleast one reference event in the column 'leaf_type'
        """
        # Make sure that the customer has atleast one reference event of interest (new purchase/ used vehicle purchase/ type of service)
        df = df.groupby(cfg['ID_COL']).filter(lambda x: True in list(x[cfg['EVENT_TYPE']].isin(cfg['REFERENCE_EVENT'])))

        return df

    def create_event_cols(self, df, cfg):
        """
        This function deaggregates the journey aggregations that the spark pipeline does to create event level columns
        Also retains the columns with aggregations
        Need to be run only until the spark pipeline is changed

        INPUT: data df from spark pipeline
        OUTPUT: df with De-aggregated and aggregated columns
        """

        df.sort_values(by=[cfg['ID_COL'],cfg['EVENT_DATE']], ascending=[True, True], inplace=True)

        cumsum_cols = cfg['CUMSUM_COLS'] + [col for col in df.columns if (col[:6] == "event_") & (col not in cfg['TD_COLS'])]

        agg_cols = [x for x in cumsum_cols if x in df.columns]
        #temp = df.groupby(cfg['ID_COL'])[agg_cols].diff(axis = 0, periods = 1).fillna(0)
        # Retain columns with aggregations
        agg_data = pd.DataFrame()
        agg_data = df[agg_cols]
        agg_data.columns = ['expanding_'+x for x in agg_data.columns]

        df = df.drop(columns=agg_cols)
        df = pd.concat([df, agg_data], axis=1)

        del agg_data, temp
        gc.collect()

        return df


    def create_ref_event_cols(self, df, cfg):
        """
        Input:
        data frame with the id, timeline event column (date column to create reference events)

        Output:
        dataframe with number of reference events until that point
        """
        # print(cfg['REFERENCE_EVENT'])
        # exit()
        # Flag reference events
        df[cfg['REF_EVENT_COL']] = df[cfg['EVENT_TYPE']].apply(lambda x: 1 if x in cfg['REFERENCE_EVENT'] else 0)

        # create timeline number
        df['ref_event_count'] = df.groupby(cfg['ID_COL'])[cfg['REF_EVENT_COL']].cumsum()

        df['timeline'] = df['ref_event_count']
        return df

    def create_td_cols(self, df, cfg):
        """
        Creates temporal displacement features (td)

        td_start - time from the first event (already exists)
        td_last_ref_event - time from the most recent reference event
        time_to_next_ref_event - time until the next reference event

        Parameters:
        df: Pandas dataframe, datset
        ref_events: List, reference events

        Output:
        Pandas dataframe, modified dataset
        """

        # Identify all reference events
        # Create a timeline start column where value is reference date or null. Forward fill nulls

        mask_ref = df[cfg['EVENT_TYPE']].isin(cfg['REFERENCE_EVENT'])
        df["timeline_start"] = np.where(mask_ref, df[cfg['EVENT_DATE']], np.datetime64("NaT"))
        df["timeline_start"] = df.groupby(cfg['ID_COL'])["timeline_start"].fillna(method="ffill")

        # Fill timeline 0 start dates with the 1st event date
        df["journey_start"] = df.groupby(cfg['ID_COL'])[cfg['EVENT_DATE']].transform("min")
        df["timeline_start"] = np.where(df["timeline"] == 0, df["journey_start"], df["timeline_start"])

        # Get the next row's timeline start date and get the difference in days between it and the current row
        df["next_row_timeline_start"] = df.groupby(cfg['ID_COL'])["timeline_start"].shift(-1)
        mask_row_diff = (df["next_row_timeline_start"] - df["timeline_start"]).dt.days

        # If difference > 0 then use next row's value as next timeline start and backfill to the rest of the timeline
        df["next_timeline_start"] = np.where(mask_row_diff > 0, df["next_row_timeline_start"], np.datetime64("NaT"))
        df["next_timeline_start"] = df.groupby(cfg['ID_COL'])["next_timeline_start"].fillna(method="bfill")
        # Calculate time since last reference event and time to next reference event
        # Use -999 as placeholder for timeline 0 and timelines that don't end
        df["td_start"] = (df[cfg['EVENT_DATE']] - df["journey_start"]).dt.days
        df["td_last_ref_event"] = (df[cfg['EVENT_DATE']] - df["timeline_start"]).dt.days

        df["time_to_next_ref_event"] = (df["next_timeline_start"] - df[cfg['EVENT_DATE']]).dt.days.fillna(-999)

        del mask_ref, mask_row_diff

        return df.drop(['journey_start', 'timeline_start', 'next_timeline_start', 'next_row_timeline_start'], axis=1)


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
        Create column groups based on type of interactions
        """

        # Get frequency columns
        inbound_cols = [x for x in cfg['INBOUND_INTERACTIONS'] if x in df.columns]
        outbound_cols = [x for x in cfg['OUTBOUND_INTERACTIONS'] if x in df.columns]
        positive_interactions = [x for x in cfg['POSITIVE_INTERACTIONS'] if x in df.columns]
        negative_interactions = [x for x in cfg['NEGATIVE_INTERACTIONS'] if x in df.columns]


        df['event_inbound_interactions'] = df[inbound_cols].sum(axis=1)
        df['event_outbound_interactions'] = df[outbound_cols].sum(axis=1)
        df['event_positive_interactions'] = df[positive_interactions].sum(axis=1)
        df['event_negative_interactions'] = df[negative_interactions].sum(axis=1)

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

        dynamic_cols = [x for x in cfg['EVENT_COLS'] if x in df.columns]

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


    def event_filters(self, df, cfg):
        """
        1. Apply data specific filters - remove rows from timeline 0, events happening on the day pf purchase
        2. remove rows - actual purchase event rows

        INPUT: data df, list of events to drop and reference events list
        OUTPUT: data df with reduced rows
        """
        # Exlcude events upto first buy (remove all timelines = 0)
        df = df[df['timeline'] > 0]
        # drop rows where there are events that may have data leakage
        df = df[~df[cfg['EVENT_TYPE']].isin(cfg['DROP_EVENTS'])]
        # Remove events occuring on the day of the purchase
        df = df[((df['time_to_next_ref_event'] < 0)
    					  |(df[cfg['EVENT_TYPE']].isin(cfg['REFERENCE_EVENT']))
    					    | (df['time_to_next_ref_event'] > 1))]
        return df


    def remove_timeline_0_purchases(self, df, cfg):
        """
        This function removes first purchases
        We do not have enough information for these purchases

        INPUT: data df and reference event list
        OUTPUT: data df without the very first purchase
        """
        # Exclude new purchases that are the very first events
        df = df[~((df['td_start']==0) & (df[cfg['EVENT_TYPE']].isin(cfg['REFERENCE_EVENT'])))]

        df.sort_values(by=[cfg['ID_COL'], cfg['EVENT_DATE']], ascending=[True, True], inplace=True)
        grp = df.groupby(cfg['ID_COL'])

        # Get the first row for every customer
        top_1 = grp.head(1)
        # Check if the first row is a purchase
        top_1 = top_1[(top_1[cfg['EVENT_TYPE']].isin(cfg['REFERENCE_EVENT'])) & (top_1['timeline'] == 1)]
        # Exclude them from the df
        df = df[~df[cfg['EVENT_ID']].isin(top_1[cfg['EVENT_ID']])]

        del top_1, grp

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
