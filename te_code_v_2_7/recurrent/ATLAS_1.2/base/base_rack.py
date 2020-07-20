"""
Implements base class for base silos

@author: Sathish Lakshmipathy
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""
import numpy as np
import random

class BaseRack:

    # slots help in avoiding dynamically create attributes
    __slots__ = ['__rack_order' , '__data_plus_meta' ,'__racks_map']


    def __init__(self, module_name, version_name , rack_order, data_plus_meta, racks_map):
        """
        parameters:
        module_name: name of object module.
        version_name: version of object module.
        rack_order: It's a number between 0-4 that shows which Silo we are in according to this dictionary: {"GeneralPreprocessing":1, "Preprocessing":2, "Imputing":3,"Enriching":4,"Splitting":5, "Sampling":6, "FeatureSelection":7, "Modeling":8}
        data_plus_meta: A dictionary including all resuired attributes for the object class
        racks_map : Its a dictionary that represents the relation between rack_rack_order number and Silo name
        """

        self.rack_order_ = rack_order
        self.data_plus_meta_ = data_plus_meta
        self.racks_map_ = racks_map
        #self.version_name_ = version_name
        #self.module_name_ = module_name

        np.random.seed(self.data_plus_meta_[self.rack_order_].config_["seed"])
        random.seed(self.data_plus_meta_[self.rack_order_].config_["seed"])

        self.data_plus_meta_[self.rack_order_].version_name_ = version_name
        self.data_plus_meta_[self.rack_order_].module_name_ = module_name


    @property
    def racks_map_(self):
        return self.__racks_map


    @racks_map_.setter
    def racks_map_(self, map):
        self.__racks_map = map


    @racks_map_.deleter
    def racks_map_(self):
        self.__racks_map = None


    @property
    def data_plus_meta_(self):
        return self.__data_plus_meta


    @data_plus_meta_.setter
    def data_plus_meta_(self, val):
        self.__data_plus_meta = val


    @data_plus_meta_.deleter
    def data_plus_meta_(self):
        self.__data_plus_meta = None


    @property
    def rack_order_(self):
        return self.__rack_order


    @rack_order_.setter
    def rack_order_(self, rack_order):
        self.__rack_order = rack_order


    @rack_order_.deleter
    def rack_order_(self):
        self.__rack_order = None
