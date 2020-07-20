#!/home/dev/.conda/envs/py365/bin/python3.6
import pandas as pd
import sys


import util.operation_utils as op_util

from general_preprocessing.attribute_general_preprocessing import GeneralPreprocessAttribute as attr_gen_preprocess
from general_preprocessing.general_preprocessing import GENERALPREPROCESSING as general_preprocess

from preprocessing.attribute_preprocessing import PreprocessingAttribute as attr_preprocess
from preprocessing.preprocessing import PREPROCESSING as preprocessing

from imputation.attribute_imputation import ImputationAttribute as attr_imputation
from imputation.ctu_imputation import CTUIMPUTATION as imputation

from enrich_data.attribute_enrich_data import EnrichDataAttribute as attr_enrich_data
from enrich_data.time_series_enrichment import TSENRICHDATA as enrich_data

from splitting.attribute_splitting import SplitAttribute as attr_split_data
from splitting.ctu_splitting import CTUSPLITTING as split_data

from sampling.attribute_sampling import SamplingAttribute as attr_sample_data
from sampling.sampling_high_imbalance import HIGHIMBALANCESAMPLING as sample_data

from feature_selection.attribute_feature_select import FeatureSelectionAttribute as attr_feature_select
from feature_selection.corr_feature_select import CORR_FILT as feature_select

from modeling.attribute_modeling import ModelAttribute as attr_model
from modeling.xgb import XGB as model


def main():

    # Read run agument for configuration
    config_path = sys.argv[1]
    cfg_dict = op_util.get_all_configs(config_path)

    cfg_dict['t_date'] = pd.to_datetime("today").strftime("%Y_%m_%d")

    cfg_dict = op_util.get_te_window_from_cfg(cfg_dict)

    #tdict is a dictionary of objects
    #tmap keeps track of the process stage
    tdict = {}
    tmap = {}

    "General preprocessing"
    tmap['general_preprocessing'] = 1
    tdict[0] = attr_gen_preprocess(cfg_dict['DATA_FILE_DICT'] , {})
    tdict[1] = attr_gen_preprocess({} , cfg_dict)

    gen_preproc = general_preprocess(1, tdict, tmap)
    gen_preproc.run()

    print (gen_preproc.data_plus_meta_[1].data_)

    "Preprocessing"
    tdict = gen_preproc.data_plus_meta_
    tmap = gen_preproc.racks_map_
    tmap['preprocessing'] = 2
    cfg_dict = tdict[1].config_

    tdict[2] = attr_preprocess({},cfg_dict )
    preprocess = preprocessing(2, tdict, tmap)
    preprocess.run()

    print (preprocess.data_plus_meta_[2].data_)

    "Imputation"
    tdict = preprocess.data_plus_meta_
    tmap = preprocess.racks_map_
    tmap['imputation'] = 3
    cfg_dict = tdict[2].config_

    tdict[3] = attr_imputation({},cfg_dict )
    impute = imputation(3, tdict, tmap)
    impute.run()

    print (impute.data_plus_meta_[3].data_)

    "Enrichment"
    tdict = impute.data_plus_meta_
    tmap = impute.racks_map_
    tmap['enrich_data'] = 4
    cfg_dict = tdict[3].config_

    tdict[4] = attr_enrich_data({},cfg_dict )
    enrich = enrich_data(4, tdict, tmap)
    enrich.run()

    print (enrich.data_plus_meta_[4].data_)

    "Splitting"
    tdict = enrich.data_plus_meta_
    tmap = enrich.racks_map_
    tmap['split'] = 5
    cfg_dict = tdict[4].config_

    tdict[5] = attr_split_data({},cfg_dict )
    split = split_data(5, tdict, tmap)
    split.run()

    print (split.data_plus_meta_[5].data_.train_set_dict_, split.data_plus_meta_[5].data_.validate_set_dict_)

    "Sampling"
    tdict = split.data_plus_meta_
    tmap = split.racks_map_
    tmap['sample'] = 6
    cfg_dict = tdict[5].config_

    tdict[6] = attr_sample_data({},cfg_dict )
    sample = sample_data(6, tdict, tmap)
    sample.run()

    print (sample.data_plus_meta_[6].data_.train_set_dict_, sample.data_plus_meta_[6].data_.validate_set_dict_, sample.data_plus_meta_[6].data_.predict_set_dict_)

    "FeatureSelection"
    tdict = sample.data_plus_meta_
    tmap = sample.racks_map_
    tmap['feature_select'] = 7
    cfg_dict = tdict[6].config_

    tdict[7] = attr_feature_select({},cfg_dict )
    select_feature = feature_select(7, tdict, tmap)
    select_feature.run()

    print (select_feature.data_plus_meta_[7].data_.train_set_dict_, select_feature.data_plus_meta_[7].data_.validate_set_dict_, select_feature.data_plus_meta_[7].data_.predict_set_dict_)

if __name__ == '__main__':
    main()
