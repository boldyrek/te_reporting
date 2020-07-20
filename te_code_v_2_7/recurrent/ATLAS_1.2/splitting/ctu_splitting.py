#!/usr/bin/env python3
"""
Implement splitting dataset to train/test/validate datasets

@author: Eyal Ben Zion
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""
# Create another version for spliting BOC
module_name = 'CTU_Splitting'
version_name = '1.2'

import pandas as pd
import pickle
import gc


from splitting.splitting_dataset import SplitDataset
from base_pd.base_splitting_pd import PdBaseSplitting as BaseSplitting


class CTUSPLITTING(BaseSplitting):

    def __init__(self, rack_order, data_plus_meta, racks_map):
        """
        Constructor for BOC spliting module

        parameters:
        dict_metadata: A dictionary including all resuired attributes for the object class
        dict_map : Its a dictionary that represents the relation between order number and Silo name
        """
        super(CTUSPLITTING, self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)

        self.data_plus_meta_[self.rack_order_].data_ = SplitDataset()


    def run(self):
        """
        Constructor to run the spliting module based on ctu

        ctu 0 = Latest period = predict set (outcome is unknown)
        ctu 1 = Last period where outcome is known = test period
        ctu 2 = period prior to the test period = validation period

        If there is no cv, save the train file (for each raw file)
        If there is cv, save each train file (for each raw file) + cv indices

        Save the validation file (for each raw file)
        Save prediction file (for each raw file)

        :param
            self
        :return different sets for training (train + cv indices), testing and final prediction set
        :raise
        """
        data_dict = self.data_plus_meta_[self.rack_order_ - 1].data_
        cfg = self.data_plus_meta_[self.rack_order_].config_

        low_var_cols = []
        if cfg['LOW_VAR_ENRICHED']:
            df = pd.DataFrame()
            for k,v in data_dict.items():
                df_1 = pd.read_feather(v)
                df = pd.concat([df, df_1])
                file_id = [x for x,y in cfg['DATA_FILE_DICT'].items() if k==y][0]

            low_var_cols = self.filter_low_variance_features(df, cfg)
            low_var_cols.sort()
            del df, df_1

        print('number of low variance features: ', len(low_var_cols))
        print('list of removed low variance features:\n', low_var_cols)

        # iterate over every file
        for k,v in data_dict.items():

            df = pd.read_feather(v)

            high_var_cols = []
            high_var_cols = [x for x in df.columns if x not in low_var_cols]

            df = self.transform(df, high_var_cols)

            file_id = [x for x,y in cfg['DATA_FILE_DICT'].items() if k==y][0]

            # Splitting the test train data for intime or out of time test
            if IN_TIME_TEST:
                # filter the training, testing data as per In-time
                df, df_pred = self.splitting_predict_data_intime(df, cfg)               
            else:    
                df_pred = self.splitting_predict_data(df, cfg)
                # filter the traiining data as per out-of-time test
                df = df[df[cfg['CTU_COL']] != cfg['test_ctu']]

            pred_save_file = cfg['PRED_SPLIT_PATH'] + cfg['t_date']+ '_' + str(file_id) + '.pkl'

            pickle.dump(df_pred, open(pred_save_file, "wb" ))

            self.data_plus_meta_[self.rack_order_].data_.predict_set_dict_[k] = pred_save_file

            if not cfg['INCLUDE_TEST_LABELS']:
                #TODO: should offset_date be before splitting_predict_data? check this logic
                print("Removing the partial labels using OFFSET...")
                print ("Before CTU offset :")
                print(df[cfg['CTU_COL']].value_counts())
                ctu_offset = self.offset_date(df, cfg)
                df = df[df[cfg['CTU_COL']] > ctu_offset]
                df[cfg['CTU_COL']] = df[cfg['CTU_COL']] - ctu_offset
                print ("After CTU offset :")
                print(df[cfg['CTU_COL']].value_counts())

            # Split data into train and validate and save data
            train_save_file = cfg['TRAIN_SPLIT_PATH'] + cfg['t_date'] + '_' + str(file_id) + ".pkl"

            validate_save_file = cfg['VALIDATE_SPLIT_PATH'] + cfg['t_date'] + '_' + str(file_id) + ".pkl"

            # Get validation data ofr out of time and in-time tests
            if not cfg['IN_TIME_TEST']:
                df_train, df_validate = self.splitting_customer_ctu_train_validate(df, cfg)
            else:
                if cfg['validate_folds'] != 0:
                    df_train, df_validate = self.splitting_predict_data_intime(df, cfg)
                else:
                    df_validate = df_pred

            pickle.dump(df_validate, open(validate_save_file, "wb" ))

            self.data_plus_meta_[self.rack_order_].data_.validate_set_dict_[k] = validate_save_file
            # create cv splits if required
            # This returns the full training data + indices of each of the splits in a dict
            cv_folds_dict = {}
            if cfg['if_cv']:
                cv_folds_dict = self.splitting_cv_random_folds(df_train, cfg)

            pickle.dump((df_train,cv_folds_dict), open(train_save_file, "wb" ))

            self.data_plus_meta_[self.rack_order_].data_.train_set_dict_[k] = train_save_file

            del df_train, df, df_pred, cv_folds_dict, df_validate

        gc.collect()

        print ("\nTrain Filename:")
        [print(key,":\n",value) for key,value in self.data_plus_meta_[self.rack_order_].data_.train_set_dict_.items()]
        print ("\nValidate Filename:")
        [print(key,":\n",value) for key,value in self.data_plus_meta_[self.rack_order_].data_.validate_set_dict_.items()]
        print ("\nPrediction Filename:")
        [print(key,":\n",value) for key,value in self.data_plus_meta_[self.rack_order_].data_.predict_set_dict_.items()]
