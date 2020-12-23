#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday, December 22, 2020 @ 19:45:28

@author: Tanmay Basu
"""

from data_extraction import data_extraction
from build_training_data import build_training_data

#opt = input("Enter Data Path \n")
#clf=data_extraction(opt)

btd=build_training_data('/Users/basut/anxiety_outcome_measures_data_extraction/')
btd.build_training_data()

clf=data_extraction('/Users/basut/anxiety_outcome_measures_data_extraction/')
clf.classify()
