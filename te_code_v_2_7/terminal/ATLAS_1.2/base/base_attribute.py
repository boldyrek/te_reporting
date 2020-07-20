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
Implements base class for attributes setup

@author: Sathish Lakshmipathy
@version: 1.2
Confidential and Proprietary (c) Copyright 2015-2019 Cerebri AI Inc. and Cerebri AI Corporation
"""

class BaseAttribute:

    # slots help in avoiding dynamically create attributes
    __slots__ = ['__module_name' , '__version_name' , '__data' , '__config' , '__processing_time']

    def __init__(self, data, config):

        """
        Constructor to initialize the base attribute module.
        :param
        	data: A placeholder of an  object for datasets
            config: A placeholder for the  all required configurations
        :raises none
        """

        #slots help in avoiding dynamically creation of attributes and improves memory efficacy
        self.data_ = data
        self.config_ = config


    @property
    def data_(self):
        return self.__data


    @data_.setter
    def data_(self, data):
        self.__data = data


    @data_.deleter
    def data_(self):
        self.__data = None


    @property
    def config_(self):
        return self.__config


    @config_.setter
    def config_(self, config):
        self.__config = config


    @config_.deleter
    def config_(self):
        self.__config = None


    @property
    def module_name_(self):
        return self.__module_name


    @module_name_.setter
    def module_name_(self,val):
        self.__module_name = val


    @module_name_.deleter
    def module_name_(self):
        self.__module_name = None


    @property
    def version_name_(self):
        return self.__version_name


    @version_name_.setter
    def version_name_(self,val):
        self.__version_name = val


    @version_name_.deleter
    def version_name_(self):
        self.__version_name = None

    @property
    def processing_time_(self):
        return self.__processing_time


    @processing_time_.setter
    def processing_time_(self, value):
        self.__processing_time = value


    @processing_time_.deleter
    def processing_time_(self):
        self.___processing_time = None
