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
#!/home/dev/.conda/envs/py365/bin/python3.6
import sys, os
import importlib
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import json

import config.te_constants_master as var


def update_te_constant_master(master, subset, change_col_groups):
    """
    Remove features from the master list other than the ones
    in the subset list
    
    Parameters: config('SELECTED_FEATURE_LIST') and full te_constants_master
    
    Returns:
        Edited te_constants_master
    """
    all_cols = []
    master_edited = {}
    for k in change_col_groups:
        v = master[k]
        master_edited[k] = [x for x in subset if x in v]
        all_cols.extend(v)

    cols_to_drop = [x for x in list(set(all_cols)) if x not in subset]
    master_edited['COLS_TO_DROP'] = list(set(master['COLS_TO_DROP'] + cols_to_drop))
    
    return master_edited

def get_all_configs(config_path):
    """
    Loads all configs and combine them into a dict.

    Parameters:
            config_path (str): path to config directory.

    Returns:
            cfg_dict: contains all configs (config + te_constant) as one dictionary.
    """


    spec = importlib.util.spec_from_file_location(os.path.basename(config_path).replace('.py',''), os.path.abspath(config_path))
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    print('oper util',cfg.DATA_PATH)
    #import importlib.machinery
    #loader = importlib.machinery.SourceFile(Loaderos.path.basename(config_path).replace('.py',''), os.path.abspath(config_path))
    #spec = importlib.util.spec_from_loader(loader.name, loader)
    #mod = importlib.util.module_from_spec(spec)
    #loader.exec_module(mod)

    var_list_1 = dict([(x,y) for x,y in vars(cfg).items() if not x.startswith('__')])
    var_list_2 = dict([(x,y) for x,y in vars(var).items() if not x.startswith('__')])
    
    if var_list_1['EXPERIMENT_SELECTED_FEATURES']:
        var_list_changed = update_te_constant_master(var_list_2, var_list_1['SELECTED_FEATURE_LIST'], var_list_1['CHANGE_COL_GROUPS'])
        var_list_unchanged = {key: var_list_2[key] for key in var_list_2.keys() if key not in var_list_changed.keys()}
        var_list_2 = {**var_list_changed , **var_list_unchanged}

    # update derived column groups
    var_list_2 = var.create_derived_col_groups(var_list_2)
    
    with open(os.path.dirname(var.__file__) + "/te_constants2.py", "w") as fp:
        for key, val in var_list_2.items():
            fp.write("{0} = {1}\n".format(key, val))
    # Read all config params into one dictionary
    cfg_dict = {**var_list_1, **var_list_2}

    return cfg_dict


def update_pipeline_config(pipeline_cfg, module):
    """
    Updates other racks based on the pipeline candidate module.

    example: agg_window which is a optimization parameter in enrichment module should be updated
    in the general_preprocessing module as well.

    Parameters:
            pipeline_cfg(dict): pipeline configs as a dictionary.
            module (str): module's name.

    Returns:
            pipeline_cfg: updated configs as a dictionary.
    """

    for key, value in pipeline_cfg[module].items():
        for mod, mod_val in pipeline_cfg.items():
            if mod != module:
                if key in mod_val.keys():
                    mod_val[key] = value

    return pipeline_cfg


def import_custom_source(src_file_path):
    """
    Imports file based on location and file name.

    Parameters:
            src_file_path (str): name of the module.

    Returns:
            ml_module: returns the specified package or module.
    """

    ml_module = importlib.import_module(src_file_path)

    return ml_module

def get_column_groups(df, cfg):
    """
    Groups columns according to imputation/aggregations logic.

    these groups are used in various stages of Te to aggregate and impute values from features.

    Parameters:
            df (dataframe): dataframe.
            cfg (dict): configuration dictionary.

    Returns:
            cfg: configuration dictionary with column groups each one for different type of aggregation.
    """

    cfg['drop_cols'] = list(set(cfg['EXCLUDE_COLS'] + [cfg['TE_TARGET_COL'], cfg['REF_EVENT_COL']]))
    # Get column grouped for aggregation
    cfg['cumsum_cols'] = list(set(cfg['CUMSUM_COLS'] + cfg['EVENT_COLS']))
    cfg['agg_cols'] = list(set([x for x in df.columns if x in cfg['cumsum_cols'] and x != 'td']))
    # Customer interactions columns are aggregated similarly
    cfg['customer_interactions'] = list(set([x for x in cfg['agg_cols'] + cfg['INTERACTION_EVENT_COLS'] if x not in cfg['drop_cols']]))
    cfg['rolling_agg_cols'] = list(set([x for x in cfg['AGGREGATION_COLS'] + cfg['INTERACTION_EVENT_COLS'] if x not in cfg['drop_cols']]))
    cfg['td_cols'] = list(set([x for x in df.columns if 'td_' in x and x not in cfg['drop_cols']]))
    # factor columns + cummax cols + cummedian Cols + one hot encoded(binary) cols
    cfg['cont_cols'] = list(set([x for x in cfg['FACTOR_COLS'] if x in df.columns and x not in cfg['drop_cols']]))
    cfg['customer_agg_cols'] = list(set([x for x in cfg['td_cols'] + cfg['REF_AGG_COLS'] if x not in cfg['drop_cols']] + [x for x in df.columns if 'expanding_' in x if x not in cfg['drop_cols']]))
    non_binary_cols = list(set(cfg['cumsum_cols'] + cfg['customer_interactions'] + cfg['td_cols'] + cfg['cont_cols'] + cfg['customer_agg_cols']))
    cfg['binary_cols'] = list(set([x for x in df.columns if df[x].nunique()==2 and all([True for y in df[x].unique() if y in [0,1]]) and x not in non_binary_cols and x not in cfg['drop_cols']]))

    return cfg


def get_te_window_from_cfg(cfg):
    """
    Reads the TE_WINDOW and convert the number into days, weeks, months and quarters.

    Parameters:
            cfg (dict): configuration dictionary.

    Returns:
            cfg: configuration dictionary.
    """

    te_window_size = "".join([x for x in cfg['TE_WINDOW'] if not x.isalpha()])

    for k,v in cfg['TE_CONV_DICT'].items():
        if k in cfg['TE_WINDOW']:
            for i,j in cfg['TE_CONV_DICT'][k].items():
                cfg['TE_WINDOW_' + i.upper()] = int(round(int(te_window_size) * j,0))

    return cfg


def get_te_ctu_from_cfg(cfg):
    """
    Reads the TE_CTU and convert the number into days.

    Parameters:
            cfg (dict): configuration dictionary.

    Returns:
            cfg: configuration dictionary.
    """

    te_ctu_size = "".join([x for x in cfg['CTU_UNIT'] if not x.isalpha()])

    for k,v in cfg['TE_CONV_DICT'].items():
        if k in cfg['CTU_UNIT']:
            for i,j in cfg['TE_CONV_DICT'][k].items():
                cfg['CTU_UNIT_' + i.upper()] = int(round(int(te_ctu_size) * j,0))
    return cfg


def get_agg_window_from_cfg(cfg):
    """
    Reads the TE_AGG_WINDOW and convert the number into days, weeks, months, quarters, and years.

    Parameters:
            cfg (dict): configuration dictionary.

    Returns:
            cfg: configuration dictionary.
    """

    for i,j in cfg['TE_CONV_DICT'][cfg['agg_unit']].items():
            cfg['AGG_WINDOW_' + i.upper()] = int(round(int(cfg['agg_window']) * j,0))

    return cfg


def clean_unicode(row):
    """
    Returns the list of decoded cell in the Series instead.

    Parameters:
            cfg (dict): configuration dictionary.

    Returns:
            cfg: configuration dictionary.
    """

    return [r.decode('unicode_escape').encode('ascii', 'ignore') for r in row]


def change_to_int(col):

    """
    Converts a given column into integer.

    if columns is inconvertable data type (e.g., string) returns the column as is.

    Parameters:
            col (scalar, list, tuple, 1-d array, or Series): column that needed to convert to int.

    Returns:
            col: numeric if parsing succeeded, else same type as input.
    """

    try:
        return pd.to_numeric(col)

    except ValueError:
        return col


def get_memory_usage(p_obj):

    """
    Identifies the max and average memory usage.

    Parameters:
            p_obj (pandas dataframe or numpy series): dataframe.

    Returns:
            tot_mem_usage_mb: total memory occupied by the dataframe or the numpy array in MB.
            avg_mem_usage_mb: average memory occupied by the dataframe or the numpy array in MB.
    """

    if isinstance(p_obj, pd.DataFrame):
        tot_mem_usage = p_obj.memory_usage(deep=True).sum()
        avg_mem_usage = p_obj.memory_usage(deep=True).mean()
    else:
        tot_mem_usage = p_obj.memory_usage(deep=True)
        avg_mem_usage = p_obj.memory_usage(deep=True).mean()

    tot_mem_usage_mb = tot_mem_usage/1024**2
    avg_mem_usage_mb = avg_mem_usage/1024**2

    return "{:03.2f} MB, {:03.2f} MB".format(tot_mem_usage_mb, avg_mem_usage_mb)



def downcast_dtypes(cols, d_type):

    conv_cols = pd.DataFrame()

    if 'int' in d_type:
        conv_cols = cols.apply(pd.to_numeric, downcast='unsigned')

    elif 'float'in d_type:
        conv_cols = cols.apply(pd.to_numeric, downcast='float')

    else:
        for x in cols.columns:
            if cols[x].nunique()/len(cols[x]) < 0.3:
                conv_cols[x] = cols[x].astype('category')
            else:
                conv_cols[x] = cols[x]

    return conv_cols

def compare_memory(orig_df, conv_df, typ):

    compare=pd.concat([orig_df.dtypes, conv_df.dtypes], axis=1)
    compare.columns = ['before','after']
    compare.apply(lambda x: x.value_counts, axis=1)

    print ("Total memory, avg memory used - Original - %s: %s " % (typ,get_memory_usage(orig_df)))
    print ("Total memory, avg memory used - Converted - %s: %s " % (typ,get_memory_usage(conv_df)))

    return 1

def optimize_memory_dtype(df):
    """
    Automatically infers column types
    Downcasts column data types
    """

    # Get columns named '_date'
    date_cols = [x for x in df.columns if 'date' in x]
    #signed_cols = [i for i in [x for x,y in df.dtypes.items() if 'float' in str(y)] if df[i].min()<0]
    #binary_cols = [i for i in [x for x,y in df.dtypes.items() if 'int' in str(y)] if df[i].min()==0 and df[i].max()==1]
    #unsigned_cols = [x for x,y in df.dtypes.items() if (('float' in str(y)) | ('int' in str(y))) and x not in signed_cols+binary_cols]

    # Change columns that we know from experience to their corresponding data type
    df[date_cols] = df[date_cols].apply(pd.to_datetime)

    for d_type in ['float','int','object']:

        # Get columns of each type separately
        cols_d_type = df.select_dtypes(include=[d_type])

        conv_cols_d_type = downcast_dtypes(cols_d_type,d_type)

        # Sanity check of the change

        compare_memory(cols_d_type,conv_cols_d_type, d_type)

        df[conv_cols_d_type.columns] = conv_cols_d_type

    return df
