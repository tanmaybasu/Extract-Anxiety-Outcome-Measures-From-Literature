#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:16:19 2020

@author: basut
"""

from data_extraction import data_extraction
from build_training_data import build_training_data

#opt = input("Enter Data Path \n")
#clf=data_extraction(opt)

btd=build_training_data('/Users/basut/anxiety_outcome_measures_data_extraction/')
btd.build_training_data()

clf=data_extraction('/Users/basut/anxiety_outcome_measures_data_extraction/')
clf.classify()
