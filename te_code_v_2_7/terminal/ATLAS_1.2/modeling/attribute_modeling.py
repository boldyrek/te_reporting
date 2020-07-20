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
