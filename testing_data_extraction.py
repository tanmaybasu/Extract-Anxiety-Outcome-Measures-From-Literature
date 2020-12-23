#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday, December 22, 2020 @ 20:22:47

@author: Tanmay Basu
"""

from data_extraction import data_extraction


de=data_extraction('/Users/basut/anxiety_outcome_measures_data_extraction/')
de.build_training_data() 
de.data_extraction()
