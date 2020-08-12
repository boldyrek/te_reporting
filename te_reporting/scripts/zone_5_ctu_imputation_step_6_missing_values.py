"""
Total number of rows imputed for an account  / 
devided   by total number of  distinct  CTUs for that customer
"""
import sys
sys.path.append("/home/boldyrek/mysoft/te/te_reporting/")
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import countDistinct

from helper_functions import get_imputed_df, start_spark_session, load_df, write_to_excel, get_module_from_path




def get_test_df():
    """
    Creating test data frame 
    """
    spark = start_spark_session()
    
    return spark.createDataFrame([
                       (1,1,0), (1,2,0), (1,3,0), (1,4,1), (1,5,0),
                       (2,1,0), (2,2,0), (2,3,1),
                       (3,1,1), (3,2,1), (3,3,1),
                       (4,1,1), (4,2,1), (4,3,1), (4,4,0),
                       (5,1,1), (5,2,1), (5,3,1), (5,4,0), (5,5,0),
                       (6,1,1), (6,2,1), (6,3,1), (6,4,0), (6,5,0),
                       (7,1,0), (7,2,0), (6,3,0), (6,4,0), (6,5,0),
                      ],['party_id','ctu','imputed_ctu'] )


def get_num_imputed_ctus_per_partyid( imputed_df ):
    """
    return df with number of imputed ctus for each party id
    """
    num_imputed_ctus_per_partyid = imputed_df.groupby('party_id','imputed_ctu').count()\
            .where("imputed_ctu = 1")\
            .select("party_id", col("count").alias("num_imputed_ctus"))
    return num_imputed_ctus_per_partyid
    
    
def get_num_distinct_ctus_per_partyid( imputed_df ):
    """
    Calculating number of distinct
    CTUs per customer
    """
    distinct_ctus_per_partyid = imputed_df.groupby('party_id').agg(countDistinct("CTU"))\
        .select("party_id", col("count(CTU)").alias("distinct_ctus"))
    return distinct_ctus_per_partyid


def join( distinct_ctus_per_partyid, num_imputed_ctus_per_partyid ):
    """
    joining number of disinct ctus per partyid with number of imputed stus per party id
    """
    imputed_distinct_cuts_nums_for_partyid = num_imputed_ctus_per_partyid.alias('a').\
        join(distinct_ctus_per_partyid.alias('b'), 
        col('a.party_id') == col('b.party_id'))
 
    return imputed_distinct_cuts_nums_for_partyid


def get_percentage_of_missing_ctus_per_party_id( imputed_distinct_ctus_df ):
    
    percentage_of_missing_ctus = imputed_distinct_ctus_df.withColumn\
        ("percentage_missing_ctus", col('num_imputed_ctus') / col('distinct_ctus') )
    
    return percentage_of_missing_ctus

def create_buckets( percentage_of_missing_ctus_per_partyid ):
    """
    Devide party ids by percentage of missing ctus into a list of 5 buckets
    > 0   < 0.25
    > 0.25 < 0.5
    > 0.5 < 0.75
    > 0.75 < 0.99
    > 0.99
    Output:
    +--------+-----------------------+-------+
    |party_id|percentage_missing_ctus|buckets|
    +--------+-----------------------+-------+
    |       1|                    0.2|    0.0|
    |       2|                   0.33|    1.0|
    |       3|                    1.0|    4.0|
    |       4|                   0.75|    3.0|
    |       5|                    0.6|    2.0|
    |       6|                    0.6|    2.0|
    +--------+-----------------------+-------+
    """
 
    bucketizer = Bucketizer(splits=[ 0, 0.25, 0.5, 0.75, 0.99, float('Inf') ], \
                            inputCol="percentage_missing_ctus", outputCol="buckets")
    df_of_buckets_ratio_between_imputed_distinct_ctus\
            = bucketizer.setHandleInvalid("keep").\
            transform(percentage_of_missing_ctus_per_partyid)
    return df_of_buckets_ratio_between_imputed_distinct_ctus


def get_num_partyids_per_backet( buckets_df ):
    """
    returns a list of rows
    """
    rows_list = buckets_df.groupby("buckets").count().\
                    orderBy("buckets", ascending = False).collect()
    return rows_list




def calculate_proportion_of_missing_ctus_per_percentile ( spark , num_partyids_with_missing_ctus_per_bucket , total_num_partyids ):
    """
    Caclulate total number of rows
    Start with bucket 4
    4. Proportion of partyids with more than 99% missing ctus. start iteration from bucket 4 get num of partyids with 99% or up 
    that have missing ctus devide it by total number of partyids
    3. Porportion of partyids with more 75% missing ctus . next we find out how many elements in bucket 3, 
    we get the value and then do plus bucket 4 and devive by total number of elements,
    2.  Porportion of partyids with more than 50% missing ctus
    1. Porportion of partyids with more than 25% missing ctus
    0. Porportion of partyids with more than 0% missing ctus
    """
    
    partyid_total_missing_ctus = 0 
    missing_percentage_per_bucket =  {4: '99%', 3: '75%', 2: '50%', 1: '25%', 0: '0%'}
    dict_to_sort = {}
    for bucket_num_partyids in num_partyids_with_missing_ctus_per_bucket:
        bucket = int(bucket_num_partyids[0])
        percent = missing_percentage_per_bucket.pop( bucket )
        #print(missing_percentage_per_bucket)
        party_ids_count = bucket_num_partyids[1]
        partyid_total_missing_ctus += party_ids_count
        dict_to_sort [percent] = round(partyid_total_missing_ctus / total_num_partyids,2) 
    # print the rest of the bucket where we have zero missing ctus
    for percent in missing_percentage_per_bucket.values():
        dict_to_sort [percent] = 0
    od = collections.OrderedDict(sorted(dict_to_sort.items()))  # sort dictionary by percentage
    df_list = []
    for percent , proportion in od.items():
       print(' proportion of partys with more than ', percent, ' missing ctus ', proportion )
       df_list.append((percent, float(proportion)))
    result_df = spark.createDataFrame(df_list, [' proportion of partys with more than ', ' missing ctus '])

    return result_df


import collections
import config as cfg

def main():
    
    """
    1. For every party ID caclulate number of CTU's imputed
    2. Calculate number of distinct CTU's per party id
    3. Devide number of CTU's imputed by distinct number of CTUs
    4. Create buckets 0.00% -24%, 25% - 49%, 50 - 74%, 75% - 98%, 99% -100 
    in every bucket all ther partyid those that fit inside thier backet
    5. Caclulcate proportion of partyids that fit inside the bucket out of total partyids
        Calculate proportion of accounts that have more than:	
        99% missing	0.20
        75% missing	0.40
        50% missing	0.60
        25% missing	0.70

    """
    spark = start_spark_session()
    imputed_df = get_imputed_df( cfg.IMPUTATION_TRAIN_PATH, cfg.IMPUTATION_PREDICT_PATH )
    
    num_imputed_ctus_per_partyid = get_num_imputed_ctus_per_partyid(imputed_df)
    num_distinct_ctus_per_partyid = get_num_distinct_ctus_per_partyid(imputed_df)
    joined_num_distinct_imputed_ctus_df = join(num_distinct_ctus_per_partyid,\
                                               num_imputed_ctus_per_partyid)
    percentage_of_missing_ctus_per_partyid = \
                get_percentage_of_missing_ctus_per_party_id( 
                        joined_num_distinct_imputed_ctus_df )

    party_id_and_its_bucket = create_buckets( 
                                        percentage_of_missing_ctus_per_partyid)
    num_partyids_with_missing_ctus_per_backet = get_num_partyids_per_backet( 
                                          party_id_and_its_bucket )
    total_num_ids = imputed_df.groupby("party_id").count().count()
    result_df = calculate_proportion_of_missing_ctus_per_percentile ( spark, num_partyids_with_missing_ctus_per_backet, \
                                          total_num_ids )
    write_to_excel(result_df, "zone_5_ctu_imp_ste_6_miss_ctus")

main()
