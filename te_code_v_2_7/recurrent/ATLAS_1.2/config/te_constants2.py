ID_COL = party_id
EVENT_ID = event_id
CTU_COL = CTU
TIMELINE = TIMELINE
REF_EVENT_COL = REF_EVENT
EVENT_DATE = event_date
Random_Num_col = Random_Num_col
IMPUTE_ZERO_COLS = ['cai_ins_grs_vmc', 'cai_ins_grs_mrc', 'cai_ins_grs_erc', 'cai_ins_grs_evmc', 'cai_ins_grs_vuc', 'cai_ins_grs_evnt_1', 'cai_ins_grs_evnt_2', 'cai_ins_grs_evnt_3', 'cai_ins_grs_rand']
CUMSUM_COLS = ['cai_ins_grs_vmc', 'cai_ins_grs_mrc', 'cai_ins_grs_erc', 'cai_ins_grs_evmc', 'cai_ins_grs_vuc', 'cai_ins_grs_evnt_1', 'cai_ins_grs_evnt_2', 'cai_ins_grs_evnt_3', 'cai_ins_grs_rand']
FACTOR_COLS = ['cai_factor_age', 'cai_factor_1', 'cai_factor_2']
FFILL_COLS = ['cai_factor_1', 'cai_factor_age', 'cai_factor_2']
FLB_TD_COLS = []
EVENT_COLS = []
TD_COLS = ['cai_ins_grs_vmc', 'cai_ins_grs_mrc', 'cai_ins_grs_erc', 'cai_ins_grs_evmc', 'cai_ins_grs_vuc', 'cai_ins_grs_evnt_1', 'cai_ins_grs_evnt_2', 'cai_ins_grs_evnt_3']
CUMMEDIAN_COLS = []
CUMMAX_COLS = []
REF_AGG_COLS = []
AGGREGATION_COLS = ['cai_ins_grs_vmc', 'cai_ins_grs_mrc', 'cai_ins_grs_erc', 'cai_ins_grs_evmc', 'cai_ins_grs_vuc', 'cai_ins_grs_evnt_1', 'cai_ins_grs_evnt_2', 'cai_ins_grs_evnt_3']
INBOUND_INTERACTIONS = ['cai_ins_grs_evnt_1', 'cai_ins_grs_evnt_2']
OUTBOUND_INTERACTIONS = ['cai_ins_grs_evnt_3']
POSITIVE_INTERACTIONS = ['cai_ins_grs_vmc', 'cai_ins_grs_mrc', 'cai_ins_grs_erc', 'cai_ins_grs_evmc']
NEGATIVE_INTERACTIONS = []
INTERACTION_EVENT_COLS = ['event_positive_interactions', 'event_negative_interactions', 'event_inbound_interactions', 'event_outbound_interactions']
DYNAMIC_COLUMNS = ['cai_ins_grs_evnt_1', 'event_negative_interactions', 'cai_ins_grs_vuc', 'cai_ins_grs_mrc', 'cai_ins_grs_evnt_2', 'cai_ins_grs_erc', 'cai_ins_grs_rand', 'event_inbound_interactions', 'cai_ins_grs_evnt_3', 'event_outbound_interactions', 'cai_ins_grs_vmc', 'event_positive_interactions', 'cai_ins_grs_evmc']
FEATURE_SELECTION_GROUP = []
DROP_EVENTS = []
COLS_TO_DROP = []
EXCLUDE_COLS = ['index', 'yr_col', 'Index', 'ctu', 'year', 'Unnamed: 0', 'level_0', 'yr_month']
CATEGORICAL_COLS = ['cai_factor_1', 'cai_factor_age', 'cai_factor_2']
create_derived_col_groups = <function create_derived_col_groups at 0x7f510832c950>
PARSE_DATE_COLS = ['event_date']
