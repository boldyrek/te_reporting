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
