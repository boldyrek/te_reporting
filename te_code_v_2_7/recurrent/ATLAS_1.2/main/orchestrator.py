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
import multiprocessing as mp
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, brier_score_loss
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import os, psutil
import warnings
warnings.filterwarnings("ignore")
# sys.path.append('/home/dev/cerebriai/orchestrator/')
# from utils.load_pipelines_configs_cnn_one_branch import load_configs
from config.load_pipeline_config import load_configs
import util.operation_utils as op_util
import util.ml_utils as mlutil
import time
from collections import OrderedDict


def run_te(cfg):

    # objects = []
    # with (open("/jupyter-notebooks/sathish/tf_tta_data/atlas_1.2_full/pred_scores_XGB_2019_10_14.pkl", "rb")) as openfile:
    #     while True:
    #         try:
    #             objects.append(pickle.load(openfile))
    #         except EOFError:
    #             break
    # y_test = objects[0]["label"]
    # print(len(y_test))
    # exit()
    # test_pred = objects[0]["prediction"]
    # test_prob = objects[0]["probability"]
    # d = OrderedDict()
    # d["accuracy"] = accuracy_score(y_test, test_pred)
    # d["precision"] = precision_score(y_test, test_pred)
    # d["recall"] = recall_score(y_test, test_pred)
    # d["f1"] = f1_score(y_test, test_pred)
    # d["roc_auc"] = roc_auc_score(y_test, test_prob)
    #
    # # Normalized confusion matrix
    # cm = confusion_matrix(y_test, test_pred)
    # norm_cm  = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # tnr, fpr, fnr, tpr = norm_cm.ravel()
    # d["tpr"] = tpr
    # d["tnr"] = tnr
    # d["1-fpr"] = 1 - fpr
    # d["1-fnr"] = 1 - fnr
    #
    # # Brier gain and KS score
    # d["brier_gain"] = 1 - brier_score_loss(y_test, test_prob)
    # ks = stats.ks_2samp(test_prob[y_test == 1], train_prob[y_train == 1])
    # d["1-ks"] = 1 - ks[0]
    #
    # # Create df with test labels and probabilities and then create lift table from them
    # y_test = y_test.rename('label')
    # df_y = pd.DataFrame(y_test, columns=["label"])
    # df_y["probability"] = test_prob
    # lift_df = lift_table(df_y, "label", "probability")
    #
    # # Convert gain back to float, use np.trapz to find gain AUC, and then normalize
    # gain = lift_df["Cumulative Percent of All Positive Timelines"].str.rstrip("%").astype("float")
    # gain_auc = np.trapz(gain) / (100 * len(gain))
    # d["lift_statistic"] = gain_auc / (1 - (np.sum(y_test == 1) / y_test.shape[0]))
    # # nan = []
    # # for i in objects[0][0]["te_3month"]:
    # #     if np.isnan(i):
    # #         nan.append(i)
    # print(d)
    # exit()



    pipeline_best, pipeline_candidates, rack = load_configs()
    process = psutil.Process(os.getpid())

    run_time = []
    tdict = {}
    # Input for instance is previous module data
    tdict[0] = pipeline_best[1]["attribute"](cfg['DATA_FILE_DICT'] , {})

    # Mark the beginning of the pipeline optimization
    cfg_full = cfg.copy()

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
                    module_run_time = []
                    print(process.memory_info().rss/(2**20))
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
            print (scores)
            pipeline_best[module] = configs_array[np.argmax(scores)]
            print ("Module:",module,"\nChosen parameters:",pipeline_best[module],"\n Score:", max(scores))

    # Mark that optimization is complete
    # And pipeline_best is the final parameter dict
    # Full dataset is required only for this pipeline
    cfg['is_final_pipeline'] = True

    # With the best pipeline parameters run the full pipeline
    # optimize for the final model hyperparameters
    tdict = {}
    tdict[0] = pipeline_best[1]["attribute"](cfg['DATA_FILE_DICT'] , {})
    cfg_full = cfg.copy()
    final_data = {}
    final_config = {}
    #Make sure that the model hyperparameter is
    for key in sorted(pipeline_best):
        print(process.memory_info().rss/(2**20))
        print (key, list(rack.keys())[list(rack.values()).index(key)])
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

    print (final_data)

    # Train final model on all data
    # And score predict dataset
    model_index = rack["Modeling"]
    data_dict = final_data[model_index - 1]
    parameters = final_config[model_index]['hyperparameters']

    print ("Final model hyperparameters:\n", parameters)
    # print(data_dict.train_set_dict_['full'])
    # print(data_dict.predict_set_dict_['pred'])
    X_train,_ = pickle.load(open(data_dict.train_set_dict_['full'],'rb'))
    y_train = X_train.pop(cfg['TE_TARGET_COL'])
    print("sdf", X_train.shape)
    print(y_train.shape)
    print(y_train.sum())
    # exit()
    final_model = module_object.fit(X_train, y_train, parameters)

    final_model_name = pipeline_best[model_index]['module'].__name__
    model_file_name = cfg['MODEL_PATH'] + final_model_name + '_' + cfg['t_date'] + '.pkl'
    pickle.dump((final_model, final_config), open(model_file_name, 'wb'))
    X_pred = pickle.load(open(data_dict.predict_set_dict_['pred'],'rb'))
    y_pred = X_pred.pop(cfg['TE_TARGET_COL'])
    pred_file_name = cfg['SCORE_PATH'] + final_model_name + '_' + cfg['t_date'] + '.pkl'
    pred_score = module_object.predict(final_model, X_pred, y_pred)
    pred_score_df = pd.DataFrame(module_object.predict(final_model, X_pred, y_pred))
    pickle.dump(pred_score, open(pred_file_name, 'wb'))

    X_val_tr, X_val_te = pickle.load(open(data_dict.validate_set_dict_[1],'rb'))
    y_val_tr = X_val_tr.pop(cfg['TE_TARGET_COL'])
    y_val_te = X_val_te.pop(cfg['TE_TARGET_COL'])
    print("sdf", X_val_tr.shape)
    print(y_val_tr.shape)
    print(y_val_tr.sum())
    exit()

    final_model_te = module_object.fit(X_val_tr, y_val_tr, parameters)

    test_file_name = cfg['TEST_SCORE_PATH'] + final_model_name + '_' + cfg['t_date'] + '.pkl'
    test_score = module_object.predict(final_model_te, X_val_te, y_val_te)
    pickle.dump(test_score, open(test_file_name, 'wb'))

    lift_table_te = mlutil.lift_table2(test_score)
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    print(lift_table_te)

    # return {'run_time': run_time, 'results': [pred_file_name, model_file_name]}
    return run_time, [pred_file_name, test_file_name, model_file_name]


def main():

    # capture the config path from the run arguments
    # Column names of the dataframe to store model results
    model_cols = ['Forecast_Cycle', 'Vintage', 'Model_Trend', 'Model_Size', 'Seasons', 'Season_Duration',
                  'Target', 'Prediction', 'Module_Imputation', 'Module_Enrichment', 'Module_Split', 'Module_Model']

    config_path = sys.argv[1]
    cfg_dict = op_util.get_all_configs(config_path)
    cfg_dict['t_date'] = pd.to_datetime("today").strftime("%Y_%m_%d")

    cfg_dict = op_util.get_te_window_from_cfg(cfg_dict)

    cfg_dict['is_final_pipeline'] = False
    # old_stdout = sys.stdout
    #
    # log_file = open(cfg_dict["LOG_PATH"] + "Te_logfile_" + cfg_dict['t_date'] + ".log","w")
    # print(cfg_dict)
    # exit()
    # sys.stdout = log_file
    runtime, results = run_te(cfg_dict)
    print ("results file:", results)
    print ("runtime:", runtime)
    # log_file.close()

if __name__ == '__main__':
    # pdb.set_trace()
    main()
