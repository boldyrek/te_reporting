"""
Difference in descriptive statistics between train and test							
							
	Delta minimum	Delta maximum	Delta mean	Delta standard deviation	Delta median	Kl divergence	KS test
Column 1							
Column 2							
Column 3							
Column 4							
Column 5							
Column 6							
Column 7							

Load 
1. load imputed_train
   load imputed_test
2. calc min, max, sttdev, mean for train : get_descriptive_statistics_for_columns
 calc min, max, sttdev, mean for test : get_descriptive_statistics_for_columns
3.join dfs :
join select(*(col(x).alias(x + '_pre') for x in preprocessing_cols_stats_df.columns))
joined_df = preprocessing_cols_stats_df_re.join(imputed_cols_stats_df, preprocessing_cols_stats_df_re.column_pre == imputed_cols_stats_df.column)

4. diffrence delts between columns : get_delta_columns_df(joined_df):
5. Caclulate KS divergence and kl divergence
6. K test
7. join delta_df, k_test, kl_divergence
"""

import sys
sys.path.append("/home/boldyrek/mysoft/te/te_reporting/")
from helper_functions import get_imputed_df, start_spark_session, load_df
from col_stats import *
import config as cfg
from helper_functions import *
from scipy import stats



spark = start_spark_session()

test_df = spark.createDataFrame([(1,1,0),(1,1,1),(1,1,1),
                                        (2,1,1),(2,2,0),
                                            (3,1,1),(3,2,0),
                                         (3,3,1),(3,3,0),
                                        (4,3,0),(4,3,0)
                                                  ], ['party_id', 'ctu', 'te_2month'])

# step 1 loading dfs
#imputed_train = load_df( cfg.IMPUTATION_TRAIN_PATH )
#imputed_predict = load_df( cfg.IMPUTATION_PREDICT_PATH )

imputed_train = test_df
imputed_predict = test_df

# step 2 getting descriptive statistics
imputed_train_descriptive_stats = get_df_with_descriptive_stats_for_columns ( spark,  imputed_train )
imputed_predict_descriptive_stats = get_df_with_descriptive_stats_for_columns ( spark, imputed_predict )

#Step 3 join dfs
joined_descriptive_stats= suffix_and_join_dfs(
    imputed_train_descriptive_stats, imputed_predict_descriptive_stats, 'column_name' )

delta_df = get_delta_descriptive_stats_df(joined_descriptive_stats, '2' )

# Step 5 ks stats
def get_df_with_ks_stats( imputed_train, imputed_predict ):
    columns = imputed_train.schema.names
    col_ks = []
    for col in columns:
        imputed_train_col = imputed_train.select(col).toPandas()[col].tolist()
        imputed_predict_col = imputed_predict.select(col).toPandas()[col].tolist()
        try:
            ks = stats.ks_2samp(imputed_train_col, imputed_predict_col)
            p_value = str(round(ks[0], 2))
            
            kd = str(round(ks[1], 2))

        except Exception as e:
            print('col ',col ,e)
            p_value = ''
            kd = ''      
        col_ks.append((col,p_value, kd))   
    ks_stats_df = spark.createDataFrame(col_ks, ['column_name_ks', 'p_value', 'kd'])
    return ks_stats_df

ks_stats_df = get_df_with_ks_stats (imputed_train, imputed_predict )
#ks_stats_df.show()
# Step 7 Join 
delta_df.join(ks_stats_df, col('column_name') == col('column_name_ks')).\
    select('column_name','delta_min','delta_max','delta_mean','delta_stddev','delta_median',
           'p_value','kd').show()

