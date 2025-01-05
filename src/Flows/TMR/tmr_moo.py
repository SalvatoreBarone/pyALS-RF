
"""
Copyright 2021-2025 Antonio Emmanuele <antonio.emmanuele@unina.it>
                    Salvatore Barone <salvatore.barone@unina.it>
                    
This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
pyALS-RF; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
import logging, joblib, numpy as np
from multiprocessing import cpu_count, Pool
from itertools import combinations, product
from tqdm import tqdm
from ...Model.Classifier import Classifier
from ...Model.DecisionTree import *
from ..GREP.GREP import GREP
import time
import csv 
import os
import json5
import pyamosa
from scipy.stats import norm # For cut-offs.
from sklearn.model_selection import train_test_split
import re

""" Computes the number of test set sizes to obtain an extimation of the accuracy loss.
    number of samples =                      test_set_size
                            -----------------------------------------------------
                                                                test_set_size - 1
                            1   +   error_margin^2 -----------------------------------------------------
                                                    cut_off^2 * individual_prob * (1 - individual_prob)
    test_set_size: Size of the test set.
    error_margin : Given a Probability Peval this defines the error interval size [Peval - error_margin, Peval + error_margin]
    cut_off:    The quantile of the standard normal distribution assumed a specififc confidence level (i.e. the probability that 
                the acc loss is within the interval centered in Peval).
                This value is computed internally of the function that takes as input the confidence level.
    individual_prob : The probability that a sample is present.
""" 
def compute_sample_size(test_set_size, error_margin, confidence_level, individual_prob):
    cut_off = norm.ppf(confidence_level) 
    return int(test_set_size / (1 + pow(error_margin,2) * ( (test_set_size - 1) / (pow(cut_off,2) * individual_prob * (1 - individual_prob)) ) ))


class TMR_MOO:

    """ 
        Split MOP samples and validation samples.  
    """
    def sample_dse_samples(self):
        #def sample_dse_samples(self, per_class_subsampling = False):    
        classifier = self.classifier
        # if per_class_subsampling:
        #     considered_classess = []
        #     self.x_
        #     for c in classifier.model_classes:
        #         class_indexes = np.where(classifier.y_test == np.int64(int(c)))[0]
        #         if len(class_indexes) > 0 : 
        #             mop_size = compute_sample_size(test_set_size = len(class_indexes), error_margin = 0.05, confidence_level = 0.95, individual_prob = 0.5)
        #             portion = mop_size / len(class_indexes)
        #             self.x_mop, self.x_val, self.y_mop, self.y_val, self.mop_indexes, self.validation_indexes = train_test_split(self.classifier.x_test[class_indexes], self.classifier.y_test[class_indexes], class_indexes, train_size = portion)       


        indexes = np.arange(0, len(classifier.x_test))
        y_flat = self.classifier.y_test.ravel()
        mop_size = compute_sample_size(test_set_size = len(classifier.x_test), error_margin = 0.05, confidence_level = 0.95, individual_prob = 0.5)
        portion = mop_size / len(self.classifier.x_test)
        self.x_mop, self.x_val, self.y_mop, self.y_val, self.mop_indexes, self.validation_indexes = train_test_split(self.classifier.x_test, y_flat, indexes, train_size = portion, stratify = y_flat)       
        """ It is fundamental that each test set class is in the set of sampled classes """
        self.sampled_classes = list(set(self.y_mop))
        x_test_classess = list(set(y_flat))
        for x in x_test_classess:
            if x not in self.sampled_classes:
                self.logger.error("[TMR-MOO] ERROR: In sampling, each class in the test set should be considered.")
                exit(1)
    
    """ Compute the cost of each leaf as the number of nodes in that specific leaf. """
    def compute_leaves_costs(self):
        """ For each tree mantains the number of nodes involved in each class.  """
        self.cost_per_tree = []
        for t in self.classifier.trees:
            cost_per_class = [0 for c in self.classifier.model_classes]
            for leaf in t.leaves:
                # Find the number of minterms.
                node_count = len(re.findall(r'Node_\d+', leaf["sop"]))
                # Increase the cost per each class.
                cost_per_class[int(leaf["class"])] += node_count
            # Append the cost of the single tree.
            self.cost_per_tree.append(cost_per_class)
        
    
    def compute_tree_prediction_per_sample(self):
        leaves = self.classifier.get_leaf_index_ensemble(self.x_mop)
        for tree_leaves in leaves: 
            votes = [0 for x in self.classifier.model_classes]
            for tree_id, leaf in enumerate(tree_leaves):
                # If the vote actually happened
                if leaf > 0 :
                    tree = self.classifier.trees[tree_id]
                    votes[int(tree.leaves[leaf]["class"])] += 1
            predicted_class = np.argmax(votes)
            

    def __init__(self, classifier):
        self.logger = logging.getLogger("pyALS-RF")

        self.logger.info("[TMR-MOO] Initializing the module")
        self.classifier : Classifier = classifier    
        self.logger.info("[TMR-MOO] Sampling classess..")
        self.sample_dse_samples()
        self.logger.info("[TMR-MOO] Sampling completed.")
        self.logger.info(f"[TMR-MOO] MOO-Samples: {(len(self.x_mop))}")
        self.logger.info(f"[TMR-MOO] Validation-Samples: {(len(self.x_val))}")
        # Compute leaves costs
        self.logger.info(f"[TMR-MOO] Initializing leaves costs.")
        self.compute_leaves_costs()
        self.logger.info(f"[TMR-MOO] Computed leaves costs.")
        self.logger.info(f"[TMR-MOO] Leaves costs: \r\n {self.cost_per_tree}")
        # Compute Predictions         
        self.logger.info(f"[TMR-MOO] Initializing samples per leaf.")
        self.compute_tree_prediction_per_sample()
        # Compute accuracy
        
        # self.compute_baseline_accuracy()