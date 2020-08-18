DATE = '2020_06_30_1'
TE_PATH = '/home/boldyrek/mysoft/te/te_code_v_2_7/terminal/ATLAS_1.2'
TE_CONFIG_FILE_PATH = '/home/boldyrek/mysoft/te/te_code_v_2_7/terminal/ATLAS_1.2/config/config.py'
TE_LOAD_PIPELINE_CONFIG_FILE_PATH = '/home/boldyrek/mysoft/te/te_code_v_2_7/terminal/ATLAS_1.2/config/load_pipeline_config.py'
SAVE_DATA_PATH =  '/home/boldyrek/mysoft/te/data/'
TE_TARGET_COL = 'te_2month'
IMPUTED_Df = '/home/boldyrek/mysoft/te/te_reporting'
IMPUTATION_TRAIN_PATH = SAVE_DATA_PATH + "imputed_train_" + DATE + '.csv'
IMPUTATION_PREDICT_PATH = SAVE_DATA_PATH + "imputed_predict_" + DATE + '.csv'
PREPROCESS_PATH = SAVE_DATA_PATH + "preprocessing_" + DATE + '.csv'
GEN_PREPROCESS_PATH = SAVE_DATA_PATH + "general_preprocessing_" + DATE + '.csv'
FINAL_TRAIN_PATH = SAVE_DATA_PATH + "final_train_" + DATE + '.pkl'
TE_CONSTANTS = '../te_code_v_2_7/terminal/ATLAS_1.2/config/te_constants_master.py'
TE_CONSTATS = '../te_code_v_2_7/terminal/ATLAS_1.2/config/te_constants_master.py'
TE_CONSTANTS_FILE_PATH = '/home/boldyrek/mysoft/te/te_code_v_2_7/terminal/ATLAS_1.2/config/te_constants_master.py'
SPLIT_TRAIN_PATH = SAVE_DATA_PATH + "split_train_" + DATE + '.csv'
SPLIT_PRED_PATH = SAVE_DATA_PATH + "split_pred_" + DATE + '.csv'
# Specify aggregation column as they are in te_constants file
AGGREGATIONS = ['CUMSUM_COLS','CUMMAX_COLS']
# Specify enrichmen colums as they are in enrichment file 
# Based on base_enrich_data_pd enrichement containing the following afixes
ENRICHMENT_AFFIXES = ['_rolling_min', '_rolling_skew', '_rolling_max', '_sin','_cos', '_rate', '_lag_', '_diff', 'price_per_unit','payment_fail_ratio', '_rolling_trend']