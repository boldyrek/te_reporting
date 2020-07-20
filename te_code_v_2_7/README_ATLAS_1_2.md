# EXPERIMENTS WITH HANDPICKED FEATURES

## te_constants_master.py
This is the master constant file that should be created at the beginning of the first run
Any column to be added/removed for any specific aggregation shoud be done here

## Changes to make
Use the following 3 configuration points in the config.py file:
    # List the handpicked features for experiments
      EXPERIMENT_SELECTED_FEATURES = True/False
    # If True for EXPERIMENT_SELECTED_FEATURES, add the list of handpicked features below
      SELECTED_FEATURE_LIST = []
    # Columns groups to change - Add/remopve appropriate column groups
      CHANGE_COL_GROUPS = ['IMPUTE_ZERO_COLS', 'CUMSUM_COLS', 'FACTOR_COLS', 'FFILL_COLS',\
    'FLB_TD_COLS', 'EVENT_COLS', 'TD_COLS', 'CUMMEDIAN_COLS', 'CUMMAX_COLS',\
    'REF_AGG_COLS', 'AGGREGATION_COLS', 'POSITIVE_INTERACTIONS', 'OUTBOUND_INTERACTIONS',\
    'CATEGORICAL_COLS', 'NEGATIVE_INTERACTIONS', 'INBOUND_INTERACTIONS']
    # If specific column needs to be added/removed/moved from specific column group, make the change in te_constants_master.py file