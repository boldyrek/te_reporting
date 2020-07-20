import pandas as pd
import numpy as np
from scipy.stats import linregress
import gc

from base.base_enrich_data import BaseEnrichData


class PdBaseEnrichData(BaseEnrichData):

    def __init__(self, module_name, version_name, rack_order, data_plus_meta, racks_map):
        """
        Constructor to initialize the enriching module.

        :param
        	module_name: the name of the enriching module.
            version_name: the version of enriching module.
            data_plus_meta:
            rack_order:
        :return none
        :raises none
        """
        super(PdBaseEnrichData, self).__init__(module_name, version_name, rack_order , data_plus_meta, racks_map)

    def aggregate_over_days(self, df, cfg, agg_window):
        """
        Aggregate different features based on their type over the number of days.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.
            agg_window: number of periods to look back for aggregation (i.e., 3)

        Returns:
            df: dataframe with one row per CTU for each customer.
        """

        agg_days = str(int(round(agg_window * cfg['TE_CONV_DICT'][cfg['agg_unit']]['day'],0))) + 'D'

        cols = list(set(cfg['customer_interactions'] + cfg['cont_cols'] + cfg['binary_cols'] + cfg['customer_agg_cols'] + cfg['CUMMEDIAN_COLS'] + cfg['CUMMAX_COLS'] + cfg['rolling_agg_cols']))

        df['datetime'] = pd.to_datetime(df[cfg['EVENT_DATE']])
        df.reset_index(drop=True,inplace=True)
        df.set_index('datetime', inplace=True)

        df_agg = pd.DataFrame(columns=cols)
        ##TODO: check these lists in get_column_groups in operation_utils.py
        df_agg[cfg['rolling_agg_cols']] = df.groupby(cfg['ID_COL'], sort = False)[cfg['rolling_agg_cols']].rolling(agg_days).sum()
        df_agg[cfg['CUMMAX_COLS']] = df.groupby([cfg['ID_COL']], sort = False)[cfg['CUMMAX_COLS']].rolling(agg_days).max()
        df_agg[cfg['CUMMEDIAN_COLS']] = df.groupby([cfg['ID_COL']], sort = False)[cfg['CUMMEDIAN_COLS']].rolling(agg_days).median()

        cols_last_row = cfg['cont_cols'] + cfg['binary_cols'] + cfg['customer_agg_cols'] + cfg['FLB_TD_COLS']

        df_agg[cols_last_row] = df.groupby([cfg['ID_COL']], sort = False)[cols_last_row].rolling(1).max()

        df_agg[[cfg['CTU_COL'],'yr_'+cfg['CTU_UNIT_NAME'],cfg['TE_TARGET_COL'],cfg['REF_EVENT_COL']]] = df.groupby([cfg['ID_COL']], sort = False)[cfg['CTU_COL'],'yr_'+cfg['CTU_UNIT_NAME'],cfg['TE_TARGET_COL'],cfg['REF_EVENT_COL']].rolling(1).max()

        print (f"\nAggregated over {agg_days}:")
        print ("No of customers:", df_agg.index.get_level_values(cfg['ID_COL']).nunique())
        print ("No of customers with positives:", df_agg[df_agg[cfg['TE_TARGET_COL']]==1].index.get_level_values(cfg['ID_COL']).nunique())
        print ("Total positives:", df_agg[cfg['TE_TARGET_COL']].sum())
        print ("No of rows, columns:", df_agg.shape)
        df_agg.reset_index(inplace=True)
        df_agg.sort_values([cfg['ID_COL'],cfg['CTU_COL'],'datetime'],ascending=[True,True,True],inplace=True)
        df_agg.set_index([cfg['ID_COL'],cfg['CTU_COL']],inplace=True)

        df = df_agg.groupby([cfg['ID_COL'],cfg['CTU_COL']]).tail(1)

        print (f"\nAggregating to one row per {cfg['ID_COL'],cfg['CTU_COL']}:")
        print ("No of customers:", df.index.get_level_values(cfg['ID_COL']).nunique())
        print ("No of customers with positives:", df[df[cfg['TE_TARGET_COL']]==1].index.get_level_values(cfg['ID_COL']).nunique())
        print ("Total positives:", df[cfg['TE_TARGET_COL']].sum())
        print ("No of rows, columns:", df.shape)

        del df_agg
        gc.collect()

        return df.reset_index()


    def aggregate_over_ctu(self, df,cfg):
        """
        Enrichments on data that has one row per CTU for each customer.

        1. Creates Rolling window aggregations (mean, std, skew, trend).
        2. Creates lags.
        3. Creates diff and rate.
        4. Creates cyclycal features.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df_agg: dataframe.
        """

        customer_interactions = [x for x in cfg['rolling_agg_cols'] if x in df.columns]
        ##TODO: there was a shift_col_period function to shift values of sum over journey
        #variables one row down to not include the amount in current CTU in sum over journey.
        #For Vonage we did not shift. logic of the function is copied to this file but not being called:
        # cols_to_shift = [x for x in df.columns if 'expanding_' in x]
        # df = df.sort_index(by=[cfg['ID_COL'],cfg['CTU_COL']], ascending=[True,True])
        # df = self.shift_col_period(df, cfg, -cfg['label_lag'], cols_to_shift)

        df =  self.aggregation_rolling_window(df, cfg, cfg['ID_COL'], customer_interactions)

        lag_features = customer_interactions

        for i in np.arange(1, cfg['lag_ctu']+1):
            df = self.create_lag_features(df, cfg, lag_features, i)

        df = self.create_diff_features(df, cfg, customer_interactions)
#         df = self.create_rate_features(df, cfg, customer_interactions)
        df = self.create_cyclical_features(df, cfg)

        #df = self.price_per_unit(df, cfg['PRODUCT_PRICE_UNIT'])

        #df = self.payment_fail_ratio(df, cfg)

        print (f"\nCreate columns - Rolling, Lag, Diff, Ratio (over {cfg['CTU_COL']}):")
        print ("No of customers:", df[cfg['ID_COL']].nunique())
        print ("No of customers with positives:", df[df[cfg['TE_TARGET_COL']]==1][cfg['ID_COL']].nunique())
        print ("Total positives:", df[cfg['TE_TARGET_COL']].sum())
        print ("No of rows, columns:", df.shape)

        return df.sort_index(by=[cfg['ID_COL'],cfg['CTU_COL']], ascending=[True,True])


    def shift_col_period(self, df, cfg, n_periods, col):
        """
        Move the columns over one period
        Could be shifting n_periods before or after
        positive: shift values to n_periods period before
        negative: shift values to n_periods later
        """
        # Check if the column is a list/tuple of string
        if not isinstance(col, (list, tuple)):
            print ("shifting column: ",col)
            print(len(col))
            col = [col]
        else:
            print ("shifting columns: ",col)
            print(len(col))

        # move label one row down
        df[col] = df.groupby(cfg['ID_COL'])[col].shift(n_periods, axis=0).fillna(0)
        return df


    def aggregation_rolling_window(self, df, cfg, groupby_cols, aggregation_cols):
        """
        Creates rolling aggregations for aggregation_cols over the last n (e.g., 6) CTUs for each customer.

        Aggregations are: Mean, median, min, max.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.
            groupby_cols (str): variable to group by customers with (e.g., customer id).
            aggregation_cols (list): list of variable names for which TE creates extra rolling columns.

        Returns:
            df: dataframe with rolling features.
        """

        df = df.sort_values(by=[cfg['ID_COL'],cfg['CTU_COL']], ascending=[True,False])

        temp = pd.DataFrame(index=df.index)

        df = df.loc[:,~df.columns.duplicated()]

        for i in cfg['rolling_agg_func_list']:

            temp_cols = [x + '_rolling_' + i for x in aggregation_cols]

            if i != 'trend':
                temp[temp_cols] = df.groupby(groupby_cols, as_index=False)[aggregation_cols].rolling(cfg['rolling_agg_ctu'], min_periods = 1).aggregate(i).fillna(0).reset_index(level = 0, drop=True)

            else:
                temp[temp_cols] = df.groupby(groupby_cols, as_index=False)[aggregation_cols].rolling(cfg['rolling_agg_ctu'], min_periods = 1).apply(self.get_trend, raw=False).fillna(0).reset_index(level = 0, drop=True)


        df =  pd.concat([df, temp], axis=1)

        del temp
        gc.collect()

        return df


    def get_trend(self, array):
        """
        Gets trend of features

        Trend is defined as the slope of the regression line for each feature.
        in a given rolling window (e.g., 6 CTUs).
        Lineregress function calculates a linear least-squares regression for given.
        x (array of 0 to 5) & y (values of the last 6 CTUs).

        Parameters:
            array: array of values of a feature in last n (e.g., 6) CTUs.

        Returns:
            slope: slope of the regression line which defines trend.
        """

        y = np.array(array)
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = linregress(x,y)

        return slope


    def create_lag_features(self, df, cfg, lag_cols, lag_period):
        """
        Creates lag features for the lag_cols

        lag_period is the number of CTUs lagged

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.
            lag_cols (list): list of variable names for which TE creates extra lag columns.
            lag_period (int): lag number (e.g., 1, 2).

        Returns:
            df: dataframe with lag features.
        """

        df = df.sort_index(by=[cfg['ID_COL'],cfg['CTU_COL']], ascending=[True,False])
        df_lag = df.groupby(cfg['ID_COL'], as_index=False)[lag_cols].shift(lag_period, axis=0).fillna(0)

        df_lag.columns = [x+'_lag_'+str(lag_period) for x in df_lag.columns]

        df = pd.concat([df, df_lag], axis=1)

        del df_lag
        gc.collect()

        return df


    def create_diff_features(self, df, cfg, diff_cols):
        """
        Creates diff features for the diff_cols.

        diff is the differnce in value with the previous CTU.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.
            diff_cols (list): list of variable names for which TE creates an extra diff column.

        Returns:
            df: dataframe with diff features.
        """

        df = df.sort_index(by=[cfg['ID_COL'],cfg['CTU_COL']], ascending=[True,False])
        df_diff = df.groupby(cfg['ID_COL'], as_index=False)[diff_cols].diff().fillna(0)

        df_diff.columns = [x+'_diff' for x in df_diff.columns]

        df = pd.concat([df, df_diff], axis=1)

        del df_diff
        gc.collect()

        return df


    def create_rate_features(self, df, cfg, rate_cols):
        """
        Creates rate features for the rate_cols columns.

        rate is the rate of change in value with the previous CTU

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.
            rate_cols (list): list of variable names for which TE creates an extra rate column.

        Returns:
            df: dataframe with rate features.
        """
        #TODO: rate was not used in most of TE experiment. check the function before use in future
        df = df.sort_index(by=[cfg['ID_COL'],cfg['CTU_COL']], ascending=[True,False])

        lagcols = [x for x in df.columns if '_lag_1' in x ]
        diffcols = [x for x in df.columns if '_diff' in x ]

        rate_cols = sorted(rate_cols)
        cols = [x+'_rate' for x in rate_cols]
        df_rate = pd.DataFrame(np.true_divide(df[sorted(diffcols)].to_numpy(),df[sorted(lagcols)].to_numpy()),columns=cols).fillna(0)

        df = pd.concat([df, df_rate], axis=1)

        del df_rate
        gc.collect()

        return df


    def create_cyclical_features(self, df, cfg):
        """
        Creates two cyclical features for week, month, and quarter of an event.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df: dataframe with cyclical features.
        """

        for feat in cfg['cyclic_period']:

            if feat=='day':
                df[feat] = df['datetime'].dt.dayofweek + 1
            else:
                df[feat] = df['datetime'].apply(lambda x: getattr(x,feat))

            df[feat+'_sin'] = np.sin((df[feat]-1)*(2.*np.pi/max(df[feat])))
            df[feat+'_cos'] = np.cos((df[feat]-1)*(2.*np.pi/max(df[feat])))

        return df.drop(columns=cfg['cyclic_period'])


    def price_per_unit(self, df, cfg):
        """
        Creates mrr amount per seat.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df: dataframe with mrr_per_seat.
        """
        for k,v in cfg['PRODUCT_PRICE_UNITS']:
               df['price_per_unit_'+str(k)] = np.where(df[v[0]] == 0, 0, df[v[0]]/df[v[1]])

        return df


    def payment_fail_ratio(self, df, cfg):
        """
        Create ratio of failed payments to total payments.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df: dataframe with payment_fail_ratio.
        """

        df['payment_fail_ratio'] = np.where(df[cfg['TOTAL_PAYMENTS_COL']] == 0, 0, df[cfg['FAILED_PAYMENTS_COL']]/df[cfg['TOTAL_PAYMENTS_COL']])

        return df


    def apply_training_filter(self, df, cfg):
        """
        Excludes data before the TRAIN_START_DATE.

        The first n (e.g., 3) month of data do not have enough look back window for aggregations.
        This function removes that part of the data.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df: dataframe without the first n months.
        """

        if cfg['TRAIN_START_DATE']:

            print("Include data after date:", cfg['TRAIN_START_DATE'])
            print ("\nBefore applying START_DATE filter:")
            print ("No of customers:", df[cfg['ID_COL']].nunique())
            print("No of positive rows:", df[cfg['TE_TARGET_COL']].sum())
            print("No of positive unique ids:", df[df[cfg['TE_TARGET_COL']]==1][cfg['ID_COL']].nunique())
            print ("No of rows, columns:", df.shape)

            df = df[df['datetime'] >= pd.to_datetime(cfg['TRAIN_START_DATE'])]

        print ("\nAfter applying START_DATE filter:")
        print ("No of customers:", df[cfg['ID_COL']].nunique())
        print("No of positive rows:", df[cfg['TE_TARGET_COL']].sum())
        print("No of positive unique ids:", df[df[cfg['TE_TARGET_COL']]==1][cfg['ID_COL']].nunique())
        print ("No of rows, columns:", df.shape)

        return df.reset_index()


    def remove_small_price(self, df, cfg):
        """
        Filters accounts with price less than a threshold.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df: dataframe with customers who have larger pricess.
        """

        df['temp'] = np.where(df[cfg['PRICE']] >= cfg['PRICE_THRESHOLD'], 1, 0)
        ids = df[df['temp'] == 1][cfg['ID_COL']].unique()
        df = df[df[cfg['ID_COL']].isin(ids)]

        return df.drop(columns=['temp'])


    def check_outliers(self, df, cfg, file_id):
        """
        Identifies outliers for each variable.


        Robust z-score is used to detect outliers.
        Outlying values are replaced with the closest non-outlying value.

        Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.
            file_id (int): batch number

        Returns:
            df: dataframe with treated outliers.
        """

        cfg['drop_cols'] = list(set(cfg['EXCLUDE_COLS'] + [cfg['TE_TARGET_COL'], cfg['REF_EVENT_COL']]))
        cfg['outlier_check_cols'] = cfg['FACTOR_COLS']
        cfg['outlier_check_cols'].extend(cfg['EVENT_COLS'])
        cfg['outlier_check_cols'].extend(cfg['FLB_TD_COLS'])
        cfg['outlier_check_cols'].extend(cfg['CUMSUM_COLS'])
        cfg['outlier_check_cols'].extend(cfg['INTERACTION_EVENT_COLS'])
        lags = list(set([x for x in df.columns if '_lag' in x]))
        cfg['outlier_check_cols'].extend(lags)
        td_last = list(set([x for x in df.columns if 'td_last_' in x]))
        cfg['outlier_check_cols'].extend(td_last)
        rolling = list(set([x for x in df.columns if '_rolling' in x]))
        cfg['outlier_check_cols'].extend(rolling)
        diff = list(set([x for x in df.columns if '_diff' in x]))
        cfg['outlier_check_cols'].extend(diff)
        rate = list(set([x for x in df.columns if '_rate' in x]))
        cfg['outlier_check_cols'].extend(rate)
        NLP = list(set([x for x in df.columns if 'NLP_W2V' in x]))
        cfg['outlier_check_cols'].extend(NLP)
        cfg['outlier_check_cols'] = list(set([x for x in cfg['outlier_check_cols'] if x not in cfg['drop_cols']]))
        cfg['outlier_check_cols'] = list(set([x for x in cfg['outlier_check_cols'] if x in df.columns]))

        results = pd.DataFrame(columns = ["variable", "iqr", "median of variable", "number of data points", "number of outliers",
                                 "minimum on outliers", "maximum on outliers", "average on outliers",
                                 "minimum on no outliers", "maximum on no outliers", "average on no outliers", ])

        threshold = cfg['outlier_threshold']
        zero_cols = []

        for i in df[cfg['outlier_check_cols']]:
            if df[i].sum() == 0:
                zero_cols.append(i)
            iqr = np.subtract(*np.percentile(df[i].dropna(), [75, 25]))
            var_median = df[i].dropna().median()
            var_abs_mad = abs(df[i].dropna() - var_median)
            var_mad = var_abs_mad.median()
            MeanAD = sum(abs(df[i].dropna()- df[i].dropna().mean()))/len(df[i].dropna())
            if var_mad != 0:
                modified_z = (0.6745 * (df[i].dropna() - var_median))/var_mad

                outlier_index = list(modified_z[abs(modified_z) > threshold].index)
                no_outlier_index = list(modified_z[abs(modified_z) <= threshold].index)
                df_outliers = df[i].loc[outlier_index]
                df_no_outlier = df[i].loc[no_outlier_index]
                results = results.append({"variable":i, "iqr":np.round(iqr, 2), "median of variable":np.round(var_median, 2),
                                          "number of data points":len(modified_z), "number of outliers":sum(abs(modified_z) > threshold),
                                          "minimum on outliers": np.min(df_outliers), "maximum on outliers": np.max(df_outliers), "average on outliers": np.mean(df_outliers),
                                          "minimum on no outliers": np.min(df_no_outlier), "maximum on no outliers": np.max(df_no_outlier),"average on no outliers": np.mean(df_no_outlier)},
                                         ignore_index=True)

                df['temp_up'] = np.where(df[i] > np.max(df_no_outlier), 1, 0)
                df['temp_down'] = np.where(df[i] < np.min(df_no_outlier), 1, 0)
                df[i] = np.where(df['temp_up'] == 1, np.max(df_no_outlier), df[i])
                df[i] = np.where(df['temp_down'] == 1, np.min(df_no_outlier), df[i])
                df = df.drop(['temp_up', 'temp_down'], axis =1)

            else:
                modified_z = (0.7979 * (df[i].dropna() - var_median))/MeanAD
                outlier_index = list(modified_z[abs(modified_z) > threshold].index)
                no_outlier_index = list(modified_z[abs(modified_z) <= threshold].index)
                df_outliers = df[i].loc[outlier_index]
                df_no_outlier = df[i].loc[no_outlier_index]
                results = results.append({"variable":i, "iqr":np.round(iqr, 2), "median of variable":np.round(var_median, 2),
                                          "number of data points":len(modified_z), "number of outliers":sum(abs(modified_z) > threshold),
                                          "minimum on outliers": np.min(df_outliers), "maximum on outliers": np.max(df_outliers), "average on outliers": np.mean(df_outliers),
                                          "minimum on no outliers": np.min(df_no_outlier), "maximum on no outliers": np.max(df_no_outlier),"average on no outliers": np.mean(df_no_outlier)},
                                         ignore_index=True)

                df['temp_up'] = np.where(df[i] > np.max(df_no_outlier), 1, 0)
                df['temp_down'] = np.where(df[i] < np.min(df_no_outlier), 1, 0)
                df[i] = np.where(df['temp_up'] == 1, np.max(df_no_outlier), df[i])
                df[i] = np.where(df['temp_down'] == 1, np.min(df_no_outlier), df[i])
                df = df.drop(['temp_up', 'temp_down'], axis =1)

        results.to_csv(cfg['SAVE_DATA_PATH'] + 'outliers' + '_' + str(threshold) + '_' + cfg['t_date'] + '_' + str(file_id)+'.csv', sep = ',')

        cfg['EXCLUDE_COLS'].extend(zero_cols)

        return df
