"""
Implements attribute class for modeling methods

@author: Sathish Lakshmipathy
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""
from base.base_attribute import BaseAttribute


class ModelAttribute(BaseAttribute):

    # slots help in avoiding dynamically create attributes
    __slots__ = ['__model' , '__score_train' , '__score_test' , '__cv_score']


    @property
    def model_(self):
        return self.__model


    @model_.setter
    def model_(self, model):
        self.__model = model


    @model_.deleter
    def model_(self):
        self.__model = None


    @property
    def score_training_(self):
        return self.__score_train


    @score_training_.setter
    def score_training_(self, score):
        self.__score_train = score


    @score_training_.deleter
    def score_training_(self):
        self.__score_train = None


    @property
    def score_test_(self):
        return self.__score_test


    @score_test_.setter
    def score_test_(self , score):
        self.__score_test = score


    @score_test_.deleter
    def score_test_(self):
        self.__score_test = None


    @property
    def cv_score_(self):
        return self.__cv_score


    @cv_score_.setter
    def cv_score_(self, score):
        self.__cv_score = score


    @cv_score_.deleter
    def cv_score_(self):
        self.__cv_score = None
