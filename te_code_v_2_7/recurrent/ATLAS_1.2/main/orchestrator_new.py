#!/home/dev/.conda/envs/py365/bin/python3.6
"""
Example code to test the packeages

@author: Sathish K Lakshmipathy
@version: 1.2
"""
import multiprocessing as mp
import datetime
import pickle
import numpy as np
import pandas as pd
from copy import copy
import sys
import pdb
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, brier_score_loss
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import time
from collections import OrderedDict

import os, psutil, shutil
import warnings
warnings.filterwarnings("ignore")
# sys.path.append('/home/dev/cerebriai/orchestrator/')
# from utils.load_pipelines_configs_cnn_one_branch import load_configs
from config.load_pipeline_config import load_configs
import util.operation_utils as op_util
import util.ml_utils as mlutil
#import util.qa_test as qa_test
from splitting.splitting_dataset import SplitDataset



def get_final_train_pred(X_train, X_pred, cfg):
    """
    Removes customers with short journeys and creates out of sample train and test set for full label approach.

    After creating out-of-sample data, this function can perform other
    modifications on data if necessary (based on config flags).
    modifications includes removing and transforming features.

    Parameters:
            X_train (dataframe): train dataframe.
            X_pred (dataframe): test dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            X_train: out-of-sampled train dataframe.
            X_pred: out-of-sampled test dataframe.
    """

    if cfg['INCLUDE_TEST_LABELS']:
        all_ids = X_pred[cfg['ID_COL']].unique()
        np.random.seed(cfg['seed'])
        id_tr = np.random.choice(all_ids, int(len(all_ids) *(1-cfg['train_pct'])), replace=False)
        X_pred = X_pred[X_pred[cfg['ID_COL']].isin(id_tr)]
        X_train = X_train[~X_train[cfg['ID_COL']].isin(id_tr)]

    if cfg['TRANSFORM']:
        print('transformed features:\n', cfg['TRANSFORMED_FEATURES'])
        for i in cfg['TRANSFORMED_FEATURES']:
            X_pred[i] = np.where(X_pred[i] == 0, 0, np.log(X_pred[i]))
            X_train[i] = np.where(X_train[i] == 0, 0, np.log(X_train[i]))

    #if cfg['DROP_FEATURE']:
    #    print('drop ', cfg['CORRELATED_FEATURES'])
     #   X_pred = X_pred.drop(cfg['CORRELATED_FEATURES'], axis =1)
     #   X_train = X_train.drop(cfg['CORRELATED_FEATURES'], axis =1)

    return X_train, X_pred


def run_te(cfg):
    """
    Function to run TE pipeline.

    Parameters:
        cfg (dict): configuration dictionary.

    Returns:
        run_time: number of seconds to run the pipeline
        pred_file_name: path of the final prediction file
        model_file_name: path of the final model
    """

    pipeline_best, pipeline_candidates, rack = load_configs()
    process = psutil.Process(os.getpid())

    run_time = []
    tdict = {}
    # Input for instance is previous module data
    tdict[0] = pipeline_best[1]["attribute"](cfg['DATA_FILE_DICT'] , {})
    # Mark the beginning of the pipeline optimization
    cfg_full = cfg.copy()

    print(process.memory_info().rss/(2**20))
    for module in sorted(pipeline_candidates):
        # Load pipeline candidates
        configs_array = pipeline_candidates[module]
        scores = []

        if len(configs_array) > 1:
            # Load config of each pipeline candidate
            for idx, config in enumerate(configs_array):
                print ("MODULE ITERATED:", module)
                print ("CONFIG:", config)
                pipeline_best[module] = config.copy()
                # the keys from the copied module should be updated in the other modules, if exists
                # Example: agg_window which is a optimization parameter in enrich_data module should be updated
                # in the general_preprocessing module as well
                pipeline_best = op_util.update_pipeline_config(pipeline_best, module)
                # Load key for pipelines
                for key in sorted(pipeline_best):
                    print (pipeline_best[model_index]['module'].__name__)
                    module_run_time = []
                    # Create an instance of attribute & module
                    cfg_full = {**cfg_full, **pipeline_best[key]}.copy()
                    tdict[key] = pipeline_best[key]["attribute"]({},cfg_full)
                    module_object = pipeline_best[key]["module"](key, tdict, rack)
                    # Run the module and record the score as a value corresponding to the
                    start = time.time()
                    module_object.run()
                    module_object.data_plus_meta_[key].processing_time_ = time.time() - start
                    tdict = module_object.data_plus_meta_
                    # Record module processing time details
                    module_run_time.append(module_object.data_plus_meta_[key].module_name_) # Module_Name
                    module_run_time.append(round(module_object.data_plus_meta_[key].processing_time_, 2)) # Processing_Time
                    run_time.append(module_run_time)
                # Record all metrics for each complete pipeline run
                scores.append(tdict[key].score_test_)

            pipeline_best[module] = configs_array[np.argmax(scores)]
            print ("Module:",module,"\nChosen parameters:",pipeline_best[module],"\n Score:", max(scores))
    # Mark that optimization is complete
    # And pipeline_best is the final parameter dict
    # Full dataset is required only for this pipeline
    cfg['is_final_pipeline'] = True
    cfg_full = cfg.copy()
    final_data = {}
    final_config = {}
    tdict = {}

    if cfg['RUN_ALL_RACKS']:
        # With the best pipeline parameters run the full pipeline
        # optimize for the final model hyperparameters
        print('run all racks ...')
        tdict[0] = pipeline_best[1]["attribute"](cfg['DATA_FILE_DICT'] , {})

        for key in sorted(pipeline_best):

            print ("\n",key, list(rack.keys())[list(rack.values()).index(key)])
            start = time.time()
            module_run_time = []
            cfg_full = {**cfg_full, **pipeline_best[key]}.copy()
            tdict[key] = pipeline_best[key]["attribute"]({},cfg_full)
            module_object = pipeline_best[key]["module"](key, tdict, rack)
            module_object.run()
            print(process.memory_info().rss/(2**20))
            tdict = module_object.data_plus_meta_
            module_object.data_plus_meta_[key].processing_time_ = time.time() - start
            # Record module processing time details
            module_run_time.append(module_object.data_plus_meta_[key].module_name_) # Module_Name
            module_run_time.append(round(module_object.data_plus_meta_[key].processing_time_, 2)) # Processing_Time
            run_time.append(module_run_time)
            final_data[key] = module_object.data_plus_meta_[key].data_
            final_config[key] = module_object.data_plus_meta_[key].config_
        # Train final model on all data
        # And score predict dataset
        model_index = rack["Modeling"]
        data_dict = final_data[model_index - 1]
        parameters = final_config[model_index]['hyperparameters']
        cfg_model = final_config[model_index]
    else:
        cfg_full['drop_cols'] = list(set(cfg_full['EXCLUDE_COLS'] + [cfg_full['TE_TARGET_COL'], cfg_full['REF_EVENT_COL']]))

        if cfg_full['RUNNING_RACK'] == "Splitting":

            print('run from splitting rack ...')
            key = rack["Splitting"]
            cfg_full = {**cfg_full,**pipeline_best[rack["Splitting"]]}
            tdict[key-1] = pipeline_best[key-1]["attribute"]({},{})
            tdict[key-1].data_ = SplitDataset()
            data_dict = {}

            for i in range(1,len(cfg_full['DATA_FILE_DICT'])+1):
                data_dict[cfg_full['DATA_FILE_DICT'][i]] = cfg_full['TE_SPLITTING_PATH'] +str(i)+'.feather'

            tdict[key-1].data_ = data_dict

            for key in sorted(pipeline_best)[rack["Splitting"]-1:]:
                print ("\n",key, list(rack.keys())[list(rack.values()).index(key)])
                start = time.time()
                module_run_time = []
                cfg_full = {**cfg_full,**pipeline_best[key-3],
                            **pipeline_best[key-2], **pipeline_best[key-1], **pipeline_best[key]}.copy()
                tdict[key] = pipeline_best[key]["attribute"]({},cfg_full)
                module_object = pipeline_best[key]["module"](key, tdict, rack)
                module_object.run()
                print(process.memory_info().rss/(2**20))
                tdict = module_object.data_plus_meta_
                module_object.data_plus_meta_[key].processing_time_ = time.time() - start
                # Record module processing time details
                module_run_time.append(module_object.data_plus_meta_[key].module_name_) # Module_Name
                module_run_time.append(round(module_object.data_plus_meta_[key].processing_time_, 2)) # Processing_Time
                run_time.append(module_run_time)
                final_data[key] = module_object.data_plus_meta_[key].data_
                final_config[key] = module_object.data_plus_meta_[key].config_

        elif cfg_full['RUNNING_RACK'] == "FeatureSelection":
            print('run from feature selection rack ...')
            key = rack["FeatureSelection"]
            tdict[key-1] = pipeline_best[key-1]["attribute"]({},{})
            tdict[key-1].data_ = SplitDataset()
            cfg_full = {**cfg_full,**pipeline_best[rack["FeatureSelection"]]}
            for i in range(1,cfg_full['folds_cv']+1):
                    tdict[key-1].data_.train_set_dict_[i] = cfg_full['TE_FEATURE_SELECTION_TRAIN_PATH'] +str(i)+'.pkl'
            tdict[key-1].data_.train_set_dict_['full'] = cfg_full['TE_FEATURE_SELECTION_TRAIN_PATH'] +'full.pkl'

            for i in range(1,cfg_full['folds_validation']+1):
                tdict[key-1].data_.validate_set_dict_[i] = cfg_full['TE_FEATURE_SELECTION_VALIDATE_PATH'] +str(i)+'.pkl'

            for i in range(1,len(cfg_full['DATA_FILE_DICT'])+1):
                tdict[key-1].data_.predict_set_dict_[i] = cfg_full['TE_FEATURE_SELECTION_PREDICT_PATH'] +str(i)+'.pkl'

            for key in sorted(pipeline_best)[rack["FeatureSelection"]-1:]:
                print ("\n",key, list(rack.keys())[list(rack.values()).index(key)])
                start = time.time()
                module_run_time = []
                cfg_full = {**cfg_full, **pipeline_best[key-5], **pipeline_best[key-4], **pipeline_best[key-3],
                            **pipeline_best[key-2], **pipeline_best[key-1], **pipeline_best[key]}.copy()
                tdict[key] = pipeline_best[key]["attribute"]({},cfg_full)
                module_object = pipeline_best[key]["module"](key, tdict, rack)
                module_object.run()
                print(process.memory_info().rss/(2**20))
                tdict = module_object.data_plus_meta_
                module_object.data_plus_meta_[key].processing_time_ = time.time() - start
                # Record module processing time details
                module_run_time.append(module_object.data_plus_meta_[key].module_name_) # Module_Name
                module_run_time.append(round(module_object.data_plus_meta_[key].processing_time_, 2)) # Processing_Time
                run_time.append(module_run_time)
                final_data[key] = module_object.data_plus_meta_[key].data_
                final_config[key] = module_object.data_plus_meta_[key].config_

        elif cfg['RUNNING_RACK'] == "Modeling":
            print('run from modeling rack ...')
            key = rack["Modeling"]
            cfg_full = {**cfg_full,**pipeline_best[rack["Modeling"]]}
            tdict[key-1] = pipeline_best[key-1]["attribute"]({},{})
            tdict[key-1].data_ = SplitDataset()
            for i in range(1,cfg_full['folds_cv']+1):
                    tdict[key-1].data_.train_set_dict_[i] = cfg_full['TE_MODELING_TRAIN_PATH'] +str(i)+'.pkl'
            tdict[key-1].data_.train_set_dict_['full'] = cfg_full['TE_MODELING_TRAIN_PATH'] +'full.pkl'

            for i in range(1,cfg_full['folds_validation']+1):
                tdict[key-1].data_.validate_set_dict_[i] = cfg_full['TE_MODELING_VALIDATE_PATH'] +str(i)+'.pkl'

            tdict[key-1].data_.predict_set_dict_['pred'] = cfg_full['TE_MODELING_PREDICT_PATH']

            for key in sorted(pipeline_best)[rack["Modeling"]-1:]:
                print ("\n",key, list(rack.keys())[list(rack.values()).index(key)])
                start = time.time()
                module_run_time = []

                cfg_full = {**cfg_full, **pipeline_best[key-6], **pipeline_best[key-5], **pipeline_best[key-4], **pipeline_best[key-3],
                            **pipeline_best[key-2], **pipeline_best[key-1], **pipeline_best[key]}.copy()
                tdict[key] = pipeline_best[key]["attribute"]({},cfg_full)
                module_object = pipeline_best[key]["module"](key, tdict, rack)
                module_object.run()
                print(process.memory_info().rss/(2**20))
                tdict = module_object.data_plus_meta_
                module_object.data_plus_meta_[key].processing_time_ = time.time() - start
                # Record module processing time details
                module_run_time.append(module_object.data_plus_meta_[key].module_name_) # Module_Name
                module_run_time.append(round(module_object.data_plus_meta_[key].processing_time_, 2)) # Processing_Time
                run_time.append(module_run_time)
                final_data[key] = module_object.data_plus_meta_[key].data_
                final_config[key] = module_object.data_plus_meta_[key].config_
        # Train final model on all data
        # And score predict dataset
        model_index = rack["Modeling"]
        data_dict = tdict[model_index - 1].data_
        parameters = final_config[model_index]['hyperparameters']
        cfg_model = final_config[model_index]

    print ("Final model hyperparameters:\n", parameters)
    X_train,_ = pickle.load(open(data_dict.train_set_dict_['full'],'rb'))
    X_pred = pickle.load(open(data_dict.predict_set_dict_['pred'],'rb'))
    X_pred = X_pred.loc[:,~X_pred.columns.duplicated()]

    X_train, X_pred = get_final_train_pred(X_train, X_pred, cfg_model)
    #dump final train and test sets
    train_test_file_name = cfg_model['SAVE_DATA_PATH'] + 'final_train_test_out_of_sample_' + cfg_model['t_date'] + '.pkl'
    pickle.dump((X_train, X_pred), open(train_test_file_name, 'wb'), protocol=4)

    y_pred = X_pred.pop(cfg_model['TE_TARGET_COL'])
    y_train = X_train.pop(cfg_model['TE_TARGET_COL'])

    train_cols = [x for x in X_train.columns if x not in cfg_model['drop_cols']+[cfg_model['ID_COL'], cfg_model['CTU_COL']]]
    final_model = module_object.fit(X_train[train_cols], y_train, parameters)

    final_model_name = pipeline_best[model_index]['module'].__name__
    model_file_name = cfg_model['MODEL_PATH'] + final_model_name + '_' + cfg_model['t_date'] + '.pkl'
    pickle.dump((final_model, cfg_model, train_cols), open(model_file_name, 'wb'), protocol=4)

    pred_file_name = cfg_model['SCORE_PATH'] + final_model_name + '_' + cfg_model['t_date'] + '.pkl'
    pred_score = pd.DataFrame(columns = [cfg['ID_COL'], 'label','probability','prediction'], index = X_pred.index)
    pred_score = module_object.predict(final_model, X_pred[train_cols], y_pred)
    pred_score[cfg['ID_COL']] = X_pred[cfg['ID_COL']]
    y_pred = pd.DataFrame(y_pred)
    y_pred[cfg['ID_COL']] = X_pred[cfg['ID_COL']]
    pickle.dump(pred_score, open(pred_file_name, 'wb'), protocol=4)

    print('In test set ...')
    print ("train customers:",X_train[cfg['ID_COL']].nunique(), "train_positives:",y_train.sum())
    print ("test customers:",X_pred[cfg['ID_COL']].nunique(), "test_positives:",y_pred[cfg_model['TE_TARGET_COL']].sum())
    train = module_object.predict(final_model, X_train[train_cols], y_train)
    metrics_df = cfg_full['all_metrics'](y_train, train['probability'], y_pred[cfg_model['TE_TARGET_COL']],
                                                pred_score['probability'], pred_score['prediction'])

    metrics_dict = {'final_model': metrics_df}
    print ("\nLift Table:", "\nfilename:", pred_file_name)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print (cfg_full['lift_table'](pred_score))
            print (metrics_dict)

    pred_file_name_train = cfg_model['SCORE_PATH'] + final_model_name + '_train_' + cfg_model['t_date'] + '.pkl'
    pickle.dump(train, open(pred_file_name_train, 'wb'), protocol=4)

    print('In train set ...')
    metrics_df_train = cfg_full['all_metrics'](y_train, train['probability'], y_train, train['probability'], train['prediction'])
    metrics_train_dict = {'final_model': metrics_df_train}

    print ("\nLift Table:", "\nfilename:", pred_file_name_train)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print (cfg_full['lift_table'](train))
        print (metrics_train_dict)

    return run_time, [pred_file_name, model_file_name]


def main():
    
    """
    Main function to get the config and start the pipeline.

    Parameters:
        none.

    Returns:
        none.
    """

    config_path = sys.argv[1]
    cfg_dict = op_util.get_all_configs(config_path)

    cfg_dict['t_date'] = pd.to_datetime("today").strftime("%Y_%m_%d")

    cfg_dict = op_util.get_te_window_from_cfg(cfg_dict)

    cfg_dict['is_final_pipeline'] = False

    log_file = open(cfg_dict["LOG_PATH"] + "Te_logfile_" + cfg_dict['t_date'] + ".log","w")

    sys.stdout = log_file
    #run the QA tests
    if cfg_dict['QA_TEST']:
        qa_test.qa(cfg_dict)
        exit()

    #config_folder = os.getcwd() + '/config'
    #shutil.copytree(config_folder, cfg_dict['SAVE_CONFIG_PATH'])
    runtime, results = run_te(cfg_dict)
    print ("results file:", results)
    print ("runtime:", runtime)
    log_file.close()


if __name__ == '__main__':

    main()
