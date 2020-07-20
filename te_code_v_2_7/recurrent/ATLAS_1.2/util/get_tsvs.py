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
from sqlalchemy import create_engine, Table, MetaData
def save_tsvs():
    # specify the Data directory and the NLP table
    DATA_PATH = "/efs/cerebri/model_data/"

    FLB_table = 'CEREBRI.AI.FLB_V1_12B_TRAIN_SH'
    NLP_table = 'CZ5_FLB_NLP_W2V_V1_0' 
    engine = create_engine('snowflake://ap_cerebri:CerebriValues1@ft32896.us-east-1/EDW_DEV/TEMPDW?warehouse=READ_WH')
    for i in range(9, 13):
        # load CZ5 FLB Table 
        print("Loading Training Data " + str(i) + ' ...')
        query = "SELECT * FROM " + FLB_table + "_" + str(i)
        df = pd.read_sql_query(query, con = engine)
        print("Shape of Training Data " + str(i) + ": "+ str(df.shape))
        df = df.sort_values(['cai_telco_acct_id', 'cai_telco_event_date'], ascending = True)
        min_date = df['cai_telco_event_date'].min()
        print(min_date)
        df['cai_telco_event_id'] = 1
        #Rename columns
        df.columns = df.columns.str.upper() 
        df.columns = df.columns.str.replace('"', '')      
        # Read NLP table
        print("Reading NLP table")
        query = "SELECT * FROM CEREBRI.AI."+ NLP_table + " WHERE CAI_TELCO_PRIMARY_KEY \
                IN (select distinct CAI_TELCO_ACCT_ID from " + FLB_table +'_' + str(i) +  ")"
        df_nlp_1 = pd.read_sql_query(query, con = engine)
        rename_cols = ['v[0]', 'v[1]', 'v[2]', 'v[3]', 'v[4]', 'v[5]', 'v[6]', 'v[7]', 'v[8]','v[9]'] 
        for col in rename_cols:
            df_nlp_1.rename(columns={col: str(col) + '_CZ5_FLB_NLP_W2V_V1_0'}, inplace=True)
        print("Shape of NLP Data " + str(i) + ": "+ str(df_nlp_1.shape))        
        df_nlp_1.columns = df_nlp_1.columns.str.upper() 
        df_nlp_1['CAI_TELCO_EVENT_DATE'] = pd.to_datetime(df_nlp_1['CAI_TELCO_EVENT_DATE']).dt.date
        df_nlp_1 = df_nlp_1.rename(columns = {'CAI_TELCO_PRIMARY_KEY':'CAI_TELCO_ACCT_ID'})
        #filter only data after min date in FLB
        df_nlp_1 = df_nlp_1[df_nlp_1['CAI_TELCO_EVENT_DATE'] >= pd.to_datetime(min_date)]
        #Remove events after churn in NLP table
        query = "SELECT DISTINCT CAI_TELCO_ACCT_ID, CAI_TELCO_TERMINATION_DATE FROM " + FLB_table + "_" + str(i)
        df_termn_dates =  pd.read_sql_query(query, con = engine)
        df_termn_dates.columns = df_termn_dates.columns.str.upper()
        df_nlp_processed = pd.merge(df_nlp_1, df_termn_dates, on='CAI_TELCO_ACCT_ID', how='inner')
        df_nlp_processed = df_nlp_processed[df_nlp_processed['CAI_TELCO_EVENT_DATE'] < df_nlp_processed['CAI_TELCO_TERMINATION_DATE']]
        df_nlp_processed.drop('CAI_TELCO_TERMINATION_DATE', inplace = True, axis=1)
        df_nlp_processed = df_nlp_processed.groupby(['CAI_TELCO_ACCT_ID','CAI_TELCO_EVENT_DATE']).sum().reset_index()
        # Merge FLB with FLB NLP
        df2 = pd.merge(df, df_nlp_processed, on=['CAI_TELCO_ACCT_ID', 'CAI_TELCO_EVENT_DATE'], how='left')
        print(f'df accounts: {len(df.CAI_TELCO_ACCT_ID.unique())}')
        print(f'df nlp accounts: {len(df_nlp_1.CAI_TELCO_ACCT_ID.unique())}')
        print(f'df nlp combined accounts: {len(df2.CAI_TELCO_ACCT_ID.unique())}')
        print(f'df shape: {df.shape}')
        print(f'df nlp shape: {df_nlp_1.shape}')
        print(f'df nlp combined shape: {df2.shape}')
        #writing the data to s3
        print("Writing the data ... ")
    #     df2 = df2.reset_index(drop=True)
        df2.to_csv(DATA_PATH + "flb_1_12B_SH_V_" + str(i) + '.tsv', sep = '\t')
    
def main():

    save_tsvs()


if __name__ == '__main__':
    main()