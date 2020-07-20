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

        # iterate over every file
        for k,v in data_dict.items():
            
            print ("\nFilename:", v)
            df = pd.read_feather(v)
            
            file_id = [x for x,y in cfg['DATA_FILE_DICT'].items() if k==y][0]

            pred_save_file = cfg['PRED_SPLIT_PATH'] + cfg['t_date']+ '_' + str(file_id) + '.pkl'

            # Prediction data is CTU = 0
            df_pred = self.splitting_predict_data(df, cfg)

            pickle.dump(df_pred, open(pred_save_file, "wb" ))

            self.data_plus_meta_[self.rack_order_].data_.predict_set_dict_[k] = pred_save_file

            df = df[df[cfg['CTU_COL']]!=0]
            
            

            # Split data into train and validate and save data
            train_save_file = cfg['TRAIN_SPLIT_PATH'] + cfg['t_date'] + '_' + str(file_id) + ".pkl"

            validate_save_file = cfg['VALIDATE_SPLIT_PATH'] + cfg['t_date'] + '_' + str(file_id) + ".pkl"

            df_train, df_validate = self.splitting_train_validate_data(df, cfg)
            
            pickle.dump(df_validate, open(validate_save_file, "wb" ))

            self.data_plus_meta_[self.rack_order_].data_.validate_set_dict_[k] = validate_save_file

            # create cv splits if required
            # This returns the full training data + indices of each of the splits in a dict
            cv_folds_dict = {}

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
