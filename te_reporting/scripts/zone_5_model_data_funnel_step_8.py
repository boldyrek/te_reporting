"""
Table 1	
	Value
Number of customers	
Number of rows	
Number of customers with a positive outcome	
"""

from helper_functions import get_imputed_df, start_spark_session


def get_num_party_ids_with_positive_outcome( imputed_df ):
    
    target_column = 'te_2month'
    where_clause = target_column + ' == 1'
    num_party_ids_with_positive_outcome =\               imputed_df.where(where_clause).select('party_id').distinct().count()
    
    return num_party_ids_with_positive_outcome


def main():
    
    spark = start_spark_session()
    imputed_df = get_imputed_df()

    num_party_ids = imputed_df.select("party_id").distinct().count()

    num_rows = imputed_df.count()
    
    num_party_ids_with_positive_outcome = get_num_party_ids_with_positive_outcome( imputed_df )
 
    result = spark.createDataFrame([['Number of customers',num_party_ids],
                      ['Number of rows', num_rows],
                      ['Number of customers with a positive outcome',
                       num_party_ids_with_positive_outcome]],
                      ['', 'Value']
                      )
    return result
    
result = main()
result.show()
