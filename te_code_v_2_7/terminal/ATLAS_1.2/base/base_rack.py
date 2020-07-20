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
