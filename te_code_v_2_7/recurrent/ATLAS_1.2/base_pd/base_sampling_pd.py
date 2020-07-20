#!/home/dev/.conda/envs/py365/bin/python3.6
import numpy as np
import pandas as pd
import random
import pickle
import gc
import os, psutil


from base.base_sampling import BaseSampling
from splitting.splitting_dataset import SplitDataset


class PdBaseSampling(BaseSampling):

    def __init__(self, module_name, version_name, rack_order, data_plus_meta, racks_map):

        """
        Constructor to initialize the splitting module.

        :param
        	module_name: sampling module.
            version_name: version of sampling module.
            data_plus_meta: A dictionary inclusing all sampling attributes
            order:
        :return none
        :raises none
        """

        super(PdBaseSampling , self).__init__(module_name, version_name, rack_order, data_plus_meta, racks_map)

        self.data_plus_meta_[self.rack_order_].data_ = SplitDataset()
        

    def run(self):
        """
        Constructor to run the sampling module.

        Steps:
        1. Iterate over each raw file's split - train and validate
            i. Get the train files (equal to cv_folds) with the cv index
            ii. Combine corresponding splits from each file. Until this step the index of data is the raw file name.
            In this step the index would be the cv folds. So at the end of the process there would be x files where x = cv_folds.
            If there is no cv, sample the only available train file
            iii. Store the train files after combining each of the training file (corresponding to each raw file)
        2. If there is validation based on multiple folds, sample each validation file after separating the test rows based on CTU.
        After sampling, combine each sampled data (corresponding to each raw file) into a single file for each validation_fold.
        If no validation on folds, no need to sample validation data. Combine all the validation files into one validation file

        Parameters
            self

        Returns:
            none
        """

        data_dict = self.data_plus_meta_[self.rack_order_ - 1].data_
        cfg = self.data_plus_meta_[self.rack_order_].config_
        # Reason for iterating over the same files thrice is to avoid memory issues
        # Each training file split in the splitting step
        # gets the cv ids and sample the train data but not the cv data
        # Store in the dict assigned for each split
        # concat the data of the same cv
        n_files = min(len(data_dict.train_set_dict_),cfg['NO_TRAIN_FILE'])
        print ("\nNumber of training files to be used for sampling/training:", n_files)
        # Get sampled train data and create cv files from indices
        train_dict, train_cv_dict = self.sample_train_files(data_dict, cfg, n_files)
        # Write dfs to file
        self.data_plus_meta_[self.rack_order_].data_.train_set_dict_ = self.save_train_cv(cfg['SAMPLED_TRAIN_PATH'],train_dict, train_cv_dict, cfg)
        del train_dict, train_cv_dict

        val_dict, val_cv_dict = self.sample_validate_files(data_dict, cfg, n_files)
        self.data_plus_meta_[self.rack_order_].data_.validate_set_dict_ = self.save_train_cv(cfg['SAMPLED_VAL_PATH'],val_dict, val_cv_dict, cfg)
        del val_dict, val_cv_dict
        gc.collect()
        # Create the full training dataset
        # Required only for the finalized pipeline_best
        print ("Is this final pipeline:",cfg['is_final_pipeline'])
        if cfg['is_final_pipeline']:
            train_dict = dict([(key,pd.DataFrame()) for key in ['full']])
            train_cv_dict = dict([(key,pd.DataFrame()) for key in ['full']])
            for k,v in data_dict.train_set_dict_.items():
                df,_ = pickle.load(open(v, "rb"))
                # For every train file, read the corresponding validate file
                val = pickle.load(open(data_dict.validate_set_dict_[k], "rb"))
                # The final training with all available data is when validation folds = 0 and includes all ctus (not restricted by cfg['total_val_ctu'])
                # It is stored with the index 'full' in the training file dict
                df_train,_ = self.split_sample_validate(df, val, cfg, 0, df[cfg['CTU_COL']].max())
                train_dict['full'] = pd.concat([train_dict['full'],df_train], axis=0)
            print ("\nFull train data size before feature selection - for final model:")
            print ("No of customers:", train_dict['full'][cfg['ID_COL']].nunique())
            print ("No of rows, columns:", train_dict['full'].shape)
            print ("Total positives:", train_dict['full'][cfg['TE_TARGET_COL']].sum())

            self.data_plus_meta_[self.rack_order_].data_.train_set_dict_ = {**self.data_plus_meta_[self.rack_order_].data_.train_set_dict_, **self.save_train_cv(cfg['SAMPLED_TRAIN_PATH'],train_dict, train_cv_dict, cfg)}

            del df, df_train, val, train_dict, train_cv_dict
            gc.collect()

        self.data_plus_meta_[self.rack_order_].data_.predict_set_dict_  = self.data_plus_meta_[self.rack_order_ - 1].data_.predict_set_dict_.copy()

        print ("\nTrain Filename:")
        [print(key,":\n",value) for key,value in self.data_plus_meta_[self.rack_order_].data_.train_set_dict_.items()]
        print ("\nValidate Filename:")
        [print(key,":\n",value) for key,value in self.data_plus_meta_[self.rack_order_].data_.validate_set_dict_.items()]
        print ("\nPrediction Filename:")
        [print(key,":\n",value) for key,value in self.data_plus_meta_[self.rack_order_].data_.predict_set_dict_.items()]


    def sample_data(self, df, cfg):
        """
        Constructor to sample a given dataframe

        Parameters:
            df (dataframe): dataframe to be down/up sampled.
            cfg (dict): configuration dictionary.

        Returns:
            none.
        """
        return NotImplemented


    def save_train_cv(self, file_path, train_d, cv_d, cfg):
        """
        save trian and cv files

        Parameters:
            file_path (str): path to save pickle files
            train_d (dict): dictionary containing k (i.e., number of cross-validation folds) sampled training files.
            cv_d (dict): dictionary containing k (i.e., number of cross-validation folds) non-sampled cross-validation files.
            cfg (dict): configuration dictionary.

        Returns:
            saved_files: dictionary containing k (i.e., number of cross-validation folds) pickle files
            each pickle file contains sampled fold for train and non-sampled folds for test in cross-validation
        """

        saved_files = {}
        for t_folds,val in train_d.items():
            sampled_file = file_path + cfg['t_date']+ '_' + str(t_folds) + '.pkl'
            pickle.dump((val, cv_d[t_folds]), open(sampled_file, "wb"), protocol=4)
            saved_files[t_folds] = sampled_file

        return saved_files


    def downsample_majority_class(self, df, cfg):
        """
        Downsample majority class (e.g., zeros, negatives) if the dataset is extremely imbalanced.

        Parameters:
            df (dataframe): train dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df_downsampled: downsampled dataframe if downsampling is needed.
            df: input dataframe if downsampling is not needed.
        """

        target_col = cfg['TE_TARGET_COL']
        percent = cfg['downsample_percent']

        maj_class = np.where(df[df[target_col] == 0].shape[0] > df[df[target_col] == 1].shape[0],0,1)
        min_class = 1 - maj_class

        maj_ratio = df[df[target_col] == maj_class].shape[0] / df.shape[0]

        if maj_ratio > percent:

            final_size = int(df[df[target_col] == min_class].shape[0] * (percent / (1 - percent)))
            maj_idx = df[df[target_col] == maj_class].index
            np.random.seed(cfg['seed'])
            df_maj_ids = np.random.choice(maj_idx, final_size, replace=False)

            df_downsampled = pd.concat([df[df[target_col] == min_class], df.loc[df.index.isin(df_maj_ids)]], axis=0)

            return df_downsampled
        else:
            return df


    def upsample_minority_class(self, df, cfg):
        """
        Upsample minority (e.g., ones, positives).

        Each instance of minority class is equally upsampled.

        Parameters:
            df (dataframe): train (might be downsampled) dataframe.
            cfg (dict): configuration dictionary.

        Returns:
            df: upsampled (if required) dataframe.
        """

        target_col = cfg['TE_TARGET_COL']
        percent = cfg['upsample_percent']

        maj_class = np.where(df[df[target_col] == 0].shape[0] > df[df[target_col] == 1].shape[0],0,1)
        min_class = 1 - maj_class

        to_upsample = df[df[target_col] == min_class]

        min_ratio = df[df[target_col] == min_class].shape[0] / df.shape[0]

        # Upsample only if the pos_ratio < percent
        if min_ratio >= percent:
            return df
        else:
            factor = int(np.ceil(percent / min_ratio))
            # Replicate the positives so that the ratio is percent%
            df = df.append([df[df[target_col] == min_class]] * factor, ignore_index=False)
            return df


    def split_sample_validate(self, df, val, cfg, val_folds, n_ctus):
        """
        Extracts the CTUs for train and test sets in validation.

        CTUs are from 1 to max.
        only the number of CTUs required are taken for each validation.
        if it requires 24 CTUs, the CTUs in the validation set + in training should be 24.
        For instance, if there are 3 CTUs (e.g., CTU 1, 2, and 3) in validation file,
        TE would create 3 validation models each with one CTU in the test set.
        train file does not include any of these CTUs.
        this function adds required CTUs from validation file to train file and
        removes those CTUs from cv test sets.

        Parameters:
            df (dataframe): train dataframe.
            val (dataframe): validation dataframe.
            cfg (dict): configuration dictionary.
            val_folds (int): number of unique CTUs in validation dataframe.
            n_ctus (int): maximum value of CTU

        Returns:
            df_val: sampled dataframe for training the validation model (excludes validation CTUs).
            val_cv: non-sampled dataframe for testing the validation model (only includes validation CTUs).
        """

        df_val = pd.concat([df[df[cfg['CTU_COL']] <=  n_ctus], val[val[cfg['CTU_COL']] > val_folds]], axis = 0)
        # Sample only the training set
        df_val = self.sample_data(df_val, cfg)
        # Get the corresponding validation fold
        val_cv = val[val[cfg['CTU_COL']] <= val_folds]

        return df_val, val_cv


    def sample_train_files(self, data_dict, cfg, n_files):
        """
        Processes each split trian file and the cv ids.

        each split train pickle has two files.
        first file consists of training data.
        second file consists of ids and CTUs in k folds.
        using the combination of ids and CTUs, each train file will be devided into two files:
        a df_cv with has the index of the first fold and a df_tr which has the index of rows not in df_cv
        the sampling process only applies on the df_tr.
        this process repeats k times:
        k df_cv files are stored in a dictionary (train_cv_dict).
        k df_tr files are stored in another dictionary (train_dict).

        Parameters:
            data_dict (dict): dictionary containing all split_train pickle files.
            cfg (dict): configuration dictionary.
            n_files: number of batches (i.e., split_train pickle files).

        Returns:
            train_dict: dictionary containing k sampled training files.
            (will be used as train data in cross validation when optimizing hyperparameters).
            train_cv_dict: dictionary containing k non-sampled cross-validation files.
            (will be used as test data in cross validation when optimizing hyperparameters).
        """
        # Create dicts equal to the number of folds
        tr_fold_list = list(range(1,cfg['folds_cv'] + 1))
        train_dict = dict([(key,pd.DataFrame()) for key in tr_fold_list])
        train_cv_dict = dict([(key,pd.DataFrame()) for key in tr_fold_list])

        f_len = 1
        for k,v in data_dict.train_set_dict_.items():
            print ("Train file:", v)
            if f_len <= n_files:
                df, cv_fold_dict = pickle.load(open(v, "rb"))
                if len(cv_fold_dict.keys()) > 0:
                    for j in cv_fold_dict.keys():
                        print ("Fold:", j)
                        fold = cv_fold_dict[j].set_index([cfg['ID_COL'],cfg['CTU_COL']])
                        df_tr = df[~df.set_index([cfg['ID_COL'],cfg['CTU_COL']]).index.isin(fold.index)]
                        df_cv = df[df.set_index([cfg['ID_COL'],cfg['CTU_COL']]).index.isin(fold.index)]
                        df_tr = self.sample_data(df_tr, cfg)
                        train_cv_dict[j] = pd.concat([train_cv_dict[j],df_cv], axis=0)
                        train_dict[j] = pd.concat([train_dict[j],df_tr], axis=0)
                        print ("\ttrain shape:",train_dict[j].shape)
                        print ("\ttrain cv shape:", train_cv_dict[j].shape)
                        del df_tr

                del cv_fold_dict, df
                f_len += 1

        cust_count = 0
        positives= 0
        for k,v in train_dict.items():
            cust_count = train_dict[k][cfg['ID_COL']].nunique() + cust_count
            positives = train_dict[k][cfg['TE_TARGET_COL']].sum() + positives

        print ("TRAIN - number of customers:", cust_count)
        print ("TRAIN- Total Positives: ", positives)

        return train_dict, train_cv_dict


    def sample_validate_files(self, data_dict, cfg, n_files):
        """
        Processes each split validation file and the cv ids.

        validation is obtained by combining the corresponding trianing file with the validation CTUs.
        applies the sampling process on the train file alone.
        creates the cv from the index before the validation file is sampled.

        Parameters:
            data_dict (dict): dictionary containing all split_train and split_validate pickle files.
            cfg (dict): configuration dictionary.
            n_files: number of batches (i.e., split_train pickle files).

        Returns:
            val_dict: dictionary containing k sampled training files.
            (will be used as train data in validation model).
            val_cv_dict: dictionary containing k non-sampled cross-validation files.
            (will be used as test data in validation model).
        """
        val_fold_list = list(range(1,cfg['folds_validation'] + 1))
        val_dict = dict([(key,pd.DataFrame()) for key in val_fold_list])
        val_cv_dict = dict([(key,pd.DataFrame()) for key in val_fold_list])

        f_len = 1
        for k,v in data_dict.train_set_dict_.items():
            if f_len <= n_files:
                print ("Validate file:", v)
                df,_ = pickle.load(open(v, "rb"))
                # For every train file, read the corresponding validate file
                val = pickle.load(open(data_dict.validate_set_dict_[k], "rb"))
                val_folds = cfg['folds_validation']
                # if there is folds for validation sample the train set
                # train set (for validation) = train file from the same cv + corresponding folds
                # When there is no fold for validation, there will be one validation file - the last ctu (ctu = 1) is used for getting the metrics
                while val_folds >= 1:
                    print ("Fold:", val_folds)
                    val_df, val_cv = self.split_sample_validate(df, val, cfg, val_folds, val_folds + cfg['total_val_ctu'])
                    val_dict[val_folds] = pd.concat([val_dict[val_folds],val_df], axis=0)
                    del val_df
                    val_cv_dict[val_folds] = pd.concat([val_cv_dict[val_folds], val_cv], axis=0)
                    del val_cv
                    print ("\tval shape:",val_dict[val_folds].shape)
                    print ("\tval cv shape:", val_cv_dict[val_folds].shape)
                    val_folds -= 1
                f_len += 1
                del df, val

        cust_count_tr = 0
        positives_tr = 0
        cust_count_te = 0
        positives_te = 0

        for k,v in val_dict.items():
            cust_count_tr = val_dict[k][cfg['ID_COL']].nunique() + cust_count_tr
            positives_tr = val_dict[k][cfg['TE_TARGET_COL']].sum() + positives_tr

            cust_count_te = val_cv_dict[k][cfg['ID_COL']].nunique() + cust_count_te
            positives_te = val_cv_dict[k][cfg['TE_TARGET_COL']].sum() + positives_te

        print ("VALIDATE - Training: number of customers: ", cust_count_tr)
        print ("VALIDATE- Training Positives: ", positives_tr)
        print ("VALIDATE - Test: number of customers: ", cust_count_te)
        print ("VALIDATE - Test Positives: ", positives_te)

        return val_dict, val_cv_dict
