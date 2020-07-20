"""
Implements attribute class for splitting methods

@author: Ameneh Boroomand
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""

class SplitDataset:

    __slots__ = ['__predict_set_dict' , '__validate_set_dict' , '__train_set_dict']

    def __init__(self):

        self.train_set_dict_ = {}
        self.predict_set_dict_ = {}
        self.validate_set_dict_ = {}

    @property
    def predict_set_dict_(self):
        return self.__predict_set_dict

    @predict_set_dict_.setter
    def predict_set_dict_(self, predict_set):
        self.__predict_set_dict = predict_set

    @predict_set_dict_.deleter
    def predict_set_dict_(self):
        self.__predict_set_dict = None


    @property
    def validate_set_dict_(self):
        return self.__validate_set_dict

    @validate_set_dict_.setter
    def validate_set_dict_(self, validate_set):
        self.__validate_set_dict = validate_set

    @validate_set_dict_.deleter
    def validate_set_dict_(self):
        self.__validate_set_dict = None


    @property
    def train_set_dict_(self):
        return self.__train_set_dict

    @train_set_dict_.setter
    def train_set_dict_(self, train_set):
        self.__train_set_dict = train_set

    @train_set_dict_.deleter
    def train_set_dict_(self):
        self.__train_set_dict = None
