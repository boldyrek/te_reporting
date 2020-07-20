"""
Implements attribute class for enriching data

@author: Sathish Lakshmipathy
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""

from base.base_attribute import BaseAttribute

class EnrichDataAttribute(BaseAttribute):
        # slots help in avoiding dynamically create attributes
    __slots__ = ['__original_columns_number' , '__after_enriching_columns_number']


    @property
    def original_columns_number_(self):
        return self.__original_columns_number


    @original_columns_number_.setter
    def original_colmns_number_(self, df):
        self.__original_columns_number = len(list(df))


    @original_columns_number_.deleter
    def original_columns_number_(self):
        self.__original_columns_number = None


    @property
    def after_enriching_columns_number_(self):
        return self.__after_enriching_columns_number


    @after_enriching_columns_number_.setter
    def after_enriching_columns_number_(self, df):
        self.__after_enriching_columns_number = len(list(df))


    @after_enriching_columns_number_.deleter
    def after_enriching_columns_number_(self):
        self.__after_enriching_columns_number = None
