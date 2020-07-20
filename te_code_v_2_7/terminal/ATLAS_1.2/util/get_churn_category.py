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
import pandas as pd
import numpy as np
import pickle
import sys
import warnings 
warnings.filterwarnings('ignore')
DATA_PATH = "/efs/cerebri/model_data/"
MODEL_PATH = "/efs/cerebri/model_results/results_2_10_7/"
ENRICH_PATH_BF = MODEL_PATH + "business_failure/enriched_train_"
ENRICH_PATH_NME = MODEL_PATH + "not_meet_expectations/enriched_train_"
T_DATE = pd.to_datetime("today").strftime("%Y_%m_%d")
for i in range(1, 22):
    df = pd.read_csv(DATA_PATH + 'flb_1_10_cancel_categories_V_' + str(i) + '.tsv', sep="\t")
    df['REFERENCE_CHURN_CATEGORY'] = df['CAI_TELCO_ACCOUNT_CANCEL_CATEGORY_BUSINESS_FAILURE_CONTRACTION'] + df['CAI_TELCO_ACCOUNT_CANCEL_CATEGORY_BUSINESS_FAILURE']
#     df['REFERENCE_CHURN_CATEGORY'] = df['CAI_TELCO_ACCOUNT_CANCEL_CATEGORY_DID_NOT_MEET_EXPECTATIONS'] 
    REFERENCE_EVENT = "CAI_TELCO_EVENT_TYPE_CHURN"
    print("Dataframe shape")
    print(df.shape)
    no_customers = df['CAI_TELCO_ACCT_ID'].nunique()
    print(no_customers)
    no_category_churn = df[(df[REFERENCE_EVENT] == 1) & (df['REFERENCE_CHURN_CATEGORY']==1)]['CAI_TELCO_ACCT_ID'].nunique()
    percent_category_churn = no_category_churn/no_customers
    print(no_category_churn)
    print(percent_category_churn)
    df['temp_churn_catog_event'] = np.where((df[REFERENCE_EVENT] == 1) & (df['REFERENCE_CHURN_CATEGORY'] != 1), 1, 0)
    print("Total churn accounts...")
    print(df[df[REFERENCE_EVENT] == 1]['CAI_TELCO_ACCT_ID'].nunique())
    non_churn_catog_ids = df[df['temp_churn_catog_event'] == 1]['CAI_TELCO_ACCT_ID'].unique()
    df = df[~df['CAI_TELCO_ACCT_ID'].isin(non_churn_catog_ids)]
    all_catog_churn_ids = df[(df[REFERENCE_EVENT] == 1) & (df['REFERENCE_CHURN_CATEGORY']==1)]['CAI_TELCO_ACCT_ID'].unique()
    np.random.seed(9)
    id_removed = np.random.choice(all_catog_churn_ids, (len(all_catog_churn_ids) - int((no_customers - len(non_churn_catog_ids)) * percent_category_churn)), replace=False)
    df = df[~df['CAI_TELCO_ACCT_ID'].isin(id_removed)]
    print(df[df[REFERENCE_EVENT] == 1]['CAI_TELCO_ACCT_ID'].nunique()/df['CAI_TELCO_ACCT_ID'].nunique())
    df_enrich = pd.read_feather(MODEL_PATH + 'enriched_train_2020_01_11_' + str(i) + '.feather')
    print(df_enrich['CAI_TELCO_ACCT_ID'].nunique())
    print(df_enrich.shape)
    df_enrich = df_enrich[df_enrich['CAI_TELCO_ACCT_ID'].isin(list(df['CAI_TELCO_ACCT_ID'].unique()))]
    print(df_enrich['CAI_TELCO_ACCT_ID'].nunique())
    print(df_enrich.shape)
    df_enrich = df_enrich.reset_index()
    save_file = ENRICH_PATH_BF + T_DATE + '_' + str(i) + '.feather'
    df_enrich.to_feather(save_file)