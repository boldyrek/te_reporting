#!/home/dev/.conda/envs/py365/bin/python3.6
"""
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation

Version: Té v2.5
Author: Sathish K Lakshmipathy
Purpose: Global variables for Té
"""

## COLUMN VARIABLES

ID_COL = "party_id"

EVENT_ID = "event_id"

CTU_COL = "CTU"

TIMELINE = "TIMELINE"

REF_EVENT_COL = "REF_EVENT"

EVENT_DATE = "event_date"

# PURCHASE_DATE = "invoice_date"

#EVENT_TYPE = "leaf_type"

# PARSE_DATE_COLS = []#[EVENT_DATE]

Random_Num_col = "Random_Num_col"

IMPUTE_ZERO_COLS = ['cai_ins_grs_vmc', 'cai_ins_grs_mrc',
       'cai_ins_grs_erc', 'cai_ins_grs_evmc', 'cai_ins_grs_vuc',
       'cai_ins_grs_evnt_1', 'cai_ins_grs_evnt_2', 'cai_ins_grs_evnt_3',
       'cai_ins_grs_rand']

CUMSUM_COLS = ['cai_ins_grs_vmc', 'cai_ins_grs_mrc',
       'cai_ins_grs_erc', 'cai_ins_grs_evmc', 'cai_ins_grs_vuc',
       'cai_ins_grs_evnt_1', 'cai_ins_grs_evnt_2', 'cai_ins_grs_evnt_3',
       'cai_ins_grs_rand']


FACTOR_COLS = ['cai_factor_age', 'cai_factor_1', 'cai_factor_2']

FFILL_COLS = FACTOR_COLS

FLB_TD_COLS = []

EVENT_COLS = []


TD_COLS = ['cai_ins_grs_vmc', 'cai_ins_grs_mrc',
       'cai_ins_grs_erc', 'cai_ins_grs_evmc', 'cai_ins_grs_vuc',
       'cai_ins_grs_evnt_1', 'cai_ins_grs_evnt_2', 'cai_ins_grs_evnt_3']

CUMMEDIAN_COLS = []

CUMMAX_COLS = []

REF_AGG_COLS = []


AGGREGATION_COLS = ['cai_ins_grs_vmc', 'cai_ins_grs_mrc',
       'cai_ins_grs_erc', 'cai_ins_grs_evmc', 'cai_ins_grs_vuc',
       'cai_ins_grs_evnt_1', 'cai_ins_grs_evnt_2', 'cai_ins_grs_evnt_3']


INBOUND_INTERACTIONS = ['cai_ins_grs_evnt_1', 'cai_ins_grs_evnt_2']

OUTBOUND_INTERACTIONS = ['cai_ins_grs_evnt_3']

POSITIVE_INTERACTIONS = ['cai_ins_grs_vmc', 'cai_ins_grs_mrc',
       'cai_ins_grs_erc', 'cai_ins_grs_evmc']

NEGATIVE_INTERACTIONS = [
# "cai_ins_grs_wthdr_dcpp",
# "cai_ins_grs_icpm_dcpp",
# "cai_ins_grs_wthdr_rrsp",
# "cai_ins_grs_icpm_rrsp",
# "cai_ins_grs_wthdr_nreg",
# "cai_ins_grs_icpm_nreg",
# "cai_ins_grs_wthdr_depsp",
# "cai_ins_grs_icpm_depsp",
# "cai_ins_grs_wthdr_lira",
# "cai_ins_grs_icpm_lira"
]

INTERACTION_EVENT_COLS = ['event_positive_interactions','event_negative_interactions','event_inbound_interactions','event_outbound_interactions']

DYNAMIC_COLUMNS =  [] #CUMSUM_COLS + EVENT_COLS + INTERACTION_EVENT_COLS

# list of features for group feature selection 
FEATURE_SELECTION_GROUP = []

DROP_EVENTS = []

COLS_TO_DROP = []

# VUC_COLS = ['expanding_cai_ins_grs_cntrb_m_vol_unmtch_amt', 'cai_ins_grs_cntrb_m_vol_unmtch_amt', 'cai_ins_grs_cntrb_m_vol_unmtch_u_qty', 'td_last_cai_ins_grs_cntrb_m_vol_unmtch_amt', 'cai_ins_grs_cntrb_m_vol_unmtch_amt_rolling_std', 'cai_ins_grs_cntrb_m_vol_unmtch_amt_rolling_skew', 'cai_ins_grs_cntrb_m_vol_unmtch_amt_rolling_trend', 'cai_ins_grs_cntrb_m_vol_unmtch_amt_rolling_mean', 'cai_ins_grs_cntrb_m_vol_unmtch_amt_lag_1', 'cai_ins_grs_cntrb_m_vol_unmtch_amt_diff']

EXCLUDE_COLS = ['Unnamed: 0','index','level_0', 'Index', 'ctu', 'yr_col', 'year', 'yr_month'] #+ COLS_TO_DROP

CATEGORICAL_COLS = [] #FACTOR_COLS

def create_derived_col_groups(config_dict):
    
    config_dict['PARSE_DATE_COLS'] = [v for k,v in config_dict.items() if '_DATE' in k]
    config_dict['EXCLUDE_COLS'] = list(set(config_dict['EXCLUDE_COLS'] + config_dict['COLS_TO_DROP']))
    config_dict['CATEGORICAL_COLS'] = list(set(config_dict['CATEGORICAL_COLS'] + config_dict['FACTOR_COLS']))
    config_dict['DYNAMIC_COLUMNS'] =  list(set(config_dict['CUMSUM_COLS'] + \
                                    config_dict['EVENT_COLS'] + config_dict['INTERACTION_EVENT_COLS']))
    config_dict['FFILL_COLS'] = list(set(config_dict['FFILL_COLS'] + config_dict['FACTOR_COLS']))
        
    return config_dict
