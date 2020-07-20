"""
Implements attribute class for feature selection methods.

@author: Sathish Lakshmipathy
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""

from base.base_attribute import BaseAttribute


class FeatureSelectionAttribute(BaseAttribute):


    @property
    def feature_selection_type_(self):
        return self.feature_selection_type


    @feature_selection_type_.setter
    def feature_selection_type_(self, name):
        self.feature_selection_type = name


    @feature_selection_type_.deleter
    def feature_selection_type_(self):
        self.feature_selection_type = None
