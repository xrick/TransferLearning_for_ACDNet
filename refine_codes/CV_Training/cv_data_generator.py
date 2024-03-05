#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:29:23 2024

@author: rick
"""
import os;
import sys;
import numpy as np;
import random;

sys.path.append("../");
sys.path.append("../../");

import common.utils as U

class TH_DataGenerator(object):
    def __init__(self, samples, labels, options):
        random.seed(42);
        # Initialization
        self.data = [(samples[i], labels[i]) for i in range(0, len(samples))];
        self.opt = options;
        self.batch_size = options.batchSize;
        self.preprocess_funcs = self.preprocess_setup();

    def __len__(self):
        #Denotes the number of batches per epoch
        return int(np.floor(len(self.data) / self.batch_size));

    def __getitem__(self, batchIndex):
        # Generate one batch of data
        batchX, batchY = self.generate_batch(batchIndex);
        batchX = np.expand_dims(batchX, axis=1);
        batchX = np.expand_dims(batchX, axis=3);
        return batchX, batchY






