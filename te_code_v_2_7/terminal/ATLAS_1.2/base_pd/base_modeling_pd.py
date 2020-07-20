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
"""
Implements class for spliting the enriched dataset

@author: Eyal Ben Zion
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""
import numpy as np
import pandas as pd
import importlib
import pickle
import gc

from base.base_attribute import BaseAttribute
from modeling.attribute_modeling import ModelAttribute
from base.base_modeling import BaseModeling

import util.ml_utils as ml_utils
import util.operation_utils as op_utils


class PdBaseModeling(BaseModeling):
    """
    This is the base class for modeling.

    Attributes:
        module_name (str): name of module.
        version_name (str): version name of module.
        rack_order (int): the order of the rack (i.e. 8).
        data_plus_meta (dict): dictionary contining data and meta data.
        racks_map (dict): dictionary of all racks and their orders.
    """

    def __init__(self, module_name, version_name, rack_order, data_plus_meta, racks_map):
        """
        Constructor to initialize the modeling module.

        Attributes:
            module_name (str): name of module.
            version_name (str): version name of module.
            rack_order (int): the order of the rack (i.e. 8).
            data_plus_meta (dict): dictionary contining data and meta data.
            racks_map (dict): dictionary of all racks and their orders.
        """

        super(PdBaseModeling , self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)


    def run(self):
        """
        Constructor to run the modeling module.

        Parameters
            self

        Returns:
            none
        """

        data_dict = self.data_plus_meta_[self.rack_order_ -1].data_
        cfg = self.data_plus_meta_[self.rack_order_].config_
        # Used for optimizing the hyperparameters of the ml algorithm
        # Uses the train data folds
        if cfg['if_optimize']:
            print ("Running hyperparameter optimization")
            self.data_plus_meta_[self.rack_order_].config_["hyperparameters"] = self.optimize(data_dict.train_set_dict_.copy(), cfg)
        # Validation fold is for checking the pipeline
        # Used for comparing multiple paths in a pipeline
        # If validation folds are required by config, the score is the aggregation function used for cv for hyperparameter tuning
        print ("Running cross-validation")
        score = self.run_cv(data_dict.validate_set_dict_, cfg, self.data_plus_meta_[self.rack_order_].config_["hyperparameters"])
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print ("\nCV metrics - Validation sets:\n", score)

        if cfg['VALIDATION_MODEL']:
            self.data_plus_meta_[self.rack_order_].score_test_ = score[cfg['validate_metric']].agg(cfg['validate_aggregation'])
            print ("Validation score:", self.data_plus_meta_[self.rack_order_].score_test_)

    def optimize(self, datasets, cfg):
        """
        Optimizes the hyperparameters of the modeling module (e.g., XGBoost, LGBM) using coordinate descent method.

        scores (i.e., value of optimization metric: recall, lift_statistic, etc.)
        are calculated for k (i.e., 3) cross-validation test folds for a value of hyperparameter.
        mean of k scores is considered as the score of that specific value of hyperparameter.
        this process repeates for all the possible values (from config) of that hyperparameter.
        the value with maximum score will be chosen as the optimized value.


        Parameters:
            datasets (dict): dictionary containing all dataframes required for cross-validation optimization.
            cfg (dict): configuration dictionary.

        Returns:
            parameters: dictionary containing the names and values of hyperparameters.
        """

        parameters = cfg['hyperparameters']
        parameter_array = cfg['hyperparameters_arrays']
        # Pass only cv indexed datasets for optimization
        datasets = {x:datasets[x] for x in datasets.keys() if x in range(1, cfg['folds_cv']+1)}
        # Optimize algorithm
        for i in range(cfg['iteration']):
            for key, value in parameter_array.items():
                scores = []
                if len(value) > 1:
                    for ii, tmp in enumerate(value):
                        parameters[key] = tmp
                        scores.append(self.run_cv(datasets, cfg, parameters)[cfg['cv_metric']].agg(cfg['cv_aggregation']))

                    parameters[key] = value[np.argmax(scores)]
                    print ("optimization parameter choice:", key,parameters[key], max(scores))
                else:
                    scores.append(np.nan)

        return parameters

    def run_cv(self, df_dict, cfg, parameters):
        """
        Runs models and get the score for cross-validation and validation models.

        Parameters:
            df_dict (dict): dictionary containing one or more train and test sets.
            cfg (dict): configuration dictionary.
            parameters (dict): dictionary containing hyperparameter names and their value

        Returns:
            scores: dataframe containing performance metrics of each run.
        """

        scores = pd.DataFrame()
        algo_name = cfg["module"].__qualname__
        print("Algorithm:", algo_name, "\n")

        for k,v in df_dict.items():

            X_train, y_train, X_test, y_test = self.read_split_data(v, cfg['TE_TARGET_COL'], cfg)
            # Exclude Id columns from getting into the model data
            cols = [x for x in X_train.columns if x not in cfg['drop_cols']+[cfg['ID_COL'], cfg['CTU_COL']]]
            print('In validation set ...')
            print ("train customers:",X_train[cfg['ID_COL']].nunique(), "train_positives:",y_train.sum())
            print ("test customers:",X_test[cfg['ID_COL']].nunique(), "test_positives:",y_test.sum())

            model = self.fit(X_train[cols], y_train, parameters)
            pred = self.predict(model, X_test[cols], y_test)
            train = self.predict(model, X_train[cols], y_train)

            metrics_df = cfg['all_metrics'](y_train, train['probability'], y_test, pred['probability'], pred['prediction'])
            metrics_dict = {algo_name: metrics_df}
            # Print lift chart only for validation dataset
            if 'validate' in v:
                print ("\nLift Table:", "\nfilename:", v)
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    print (cfg['lift_table'](pred))
                if cfg['if_plot_validate']:
                    self.save_validate_plots(pred, train, metrics_dict, cfg['PLOTS_PATH']+ cfg['t_date'] + '_'+str(k))
                # Plot shap summary plot
                cfg['plot_feature_importance'](model, X_test[cols], cfg['PLOTS_PATH'] + cfg['t_date'] + '_'+str(k))
                if cfg['RUN_ALL_RACKS']:

                    model_file_name = v.replace(".pkl", "") + '_model_' + cfg['t_date'] + '.pkl'
                else:

                    model_file_name = cfg['MODEL_PATH'] + 'final_validate_model_' + cfg['t_date'] + '.pkl'

                pickle.dump((model, cols, pd.concat([pred, X_test[cfg['ID_COL']]], axis=1)), open(model_file_name, 'wb'))

            scores = pd.concat([scores, metrics_df], axis=0, sort=True)

            del model, train, pred
        gc.collect()

        return scores


    def fit(self, df, target, parameters):
        """
        Constructor to fit the modeling module.

        Parameters:
            df (dataframe): features for training.
            target (str): target column for training.
            parameters (dict): dictionary containing hyperparameter names and values.

        Returns:
            none
        """

        return NotImplemented


    def predict(self, model, df, target):
        """
        Constructor to predict the modeling module.

        Parameters:
            model (model object): fit model.
            df (dataframe): features for test.
            target (str): target column for test. target is just to add labels, it's not used in scoring

        Returns:
            none
        """
        return NotImplemented


    def read_split_data(self, pickle_file, target, cfg):
        """
        Read data from pickled file location and split to train and test.

        gets a smaller sample of data if cross-validation files are big to avoid memory issues.

        Parameters:
            pickle_file (pickle): pickle file containing one train and test set.
            target (str): name of target column.
            cfg (dict): configuration dictionary.

        Returns:
            X_train: dataframe containing all features for training.
            y_train: dataframe containing target for training.
            X_test: dataframe containing all features for testing.
            y_test: dataframe containing target for testing.
        """
        X_train, X_test = pickle.load(open(pickle_file, "rb"))

        if cfg['if_optimize']:
            if 'validate' not in pickle_file:
                if cfg['REDUCED_CV']:
                    print('initial size: ', X_train.shape, X_test.shape)
                    all_ids_train = X_train[cfg['ID_COL']].unique()
                    all_ids_test = X_test[cfg['ID_COL']].unique()
                    np.random.seed(cfg['seed'])
                    id_tr = np.random.choice(all_ids_train, int(len(all_ids_train) * cfg['cv_pct']), replace=False)
                    id_te = np.random.choice(all_ids_test, int(len(all_ids_test) * cfg['cv_pct']), replace=False)
                    X_train = X_train[X_train[cfg['ID_COL']].isin(id_tr)]
                    X_test = X_test[X_test[cfg['ID_COL']].isin(id_te)]
                    print('final size: ', X_train.shape, X_test.shape)

        y_train = X_train.pop(target)

        y_test = X_test.pop(target)
        return X_train, y_train, X_test, y_test


    def save_validate_plots(self, test_df, train_df, metrics_dict, savepath):
        """
        Plots the confusion matrix, ks plot and spider diagram.

        Parameters:
            test_df (dataframe): dataframe containing customer ids, actual class (labels),
            predicted probability, and predicted class (0, 1) all using test set.
            train_df (dataframe): dataframe containing customer ids, actual class (labels),
            predicted probability, and predicted class (0, 1) all using training set.
            metrics_dict (dict): dictionary where keys are the names of the models
            and values are the dataframes with metrics names and values.
            savepath (str): path to save plots.

        Returns:
            none
        """

        ml_utils.plot_ks_compare_train_test(train_df['label'].values, train_df['probability'].values, test_df['label'].values, test_df['probability'].values, savepath)
        # Plot confusion matrix and save it as jpg
        ml_utils.plot_confusion_matrix(test_df, sorted(test_df['label'].unique(), reverse=True), savepath)
        # Plot spider diagram by using scores
        ml_utils.plot_spider_diagram(metrics_dict, savepath)
