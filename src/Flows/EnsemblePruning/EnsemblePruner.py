"""
Copyright 2021-2025 Antonio Emmanuele <antonio.emmanuele@unina.it> , Salvatore Barone <salvatore.barone@unina.it>, 

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
from multiprocessing import cpu_count
from itertools import combinations, product
from tqdm import tqdm
from ...Model.Classifier import Classifier
from ...Model.DecisionTree import *
from ..GREP.GREP import GREP
import time
import csv 
import os
import json5
from scipy.stats import norm # For cut-offs.
from sklearn.model_selection import train_test_split
import copy


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

class Pruner:

    @staticmethod
    def initialize_tree_ranking(classifier: Classifier, Xprun, yprun):
        tree_preds = []
        pred_vectors = [[0 for x in classifier.model_classes] for sample in Xprun]
        for tree in tqdm(classifier.trees, desc = "Evaluating tree inferences"): # TODO: Add multicore ranking.
            tree_preds.append(tree.visit_by_leaves(Xprun))  
        for tree_id in tqdm(range(len(classifier.trees)), desc= "Evaluating the tree predictions"):
            for x in tqdm(range(len(Xprun)), desc = "Constructing prediction vectors"):
                pred_vectors[x][tree_preds[tree_id][x]] += 1
        return tree_preds, pred_vectors
    
    @staticmethod
    def per_sample_margin(pred_vectors, yprun):
        # Sum of the number of votes = Ntrees
        ntrees = sum(pred_vectors[0])
        margins = []
        for p, y in zip(pred_vectors, yprun):
            votes_correct = p[int(y)]
            sorted_args = np.argsort(p)
            if sorted_args[-1] == int(y):        # If the majority class is y then simply take y.
                second_class = p[-2]
            else:                       # Otherwise, simply mantain the majority class.
                second_class = p[-1]
            margins. append((votes_correct - second_class) / ntrees)
        return margins

    @staticmethod
    def update_margins(tree_preds, pred_vectors, remaining_trees, yprun):
        new_margins = []
        new_pv = {}
        for considered_tree in tqdm(remaining_trees, desc = "Updating tree resiliency"):
            new_pred_vectors = copy.deepcopy(pred_vectors)
            # Subtract the tree.
            for sample_id, sample in enumerate(tree_preds[considered_tree]):
                new_pred_vectors[sample_id][tree_preds[considered_tree][sample_id]] -= 1
            new_pv.update({considered_tree: new_pred_vectors})
            new_margins.append(Pruner.per_sample_margin(new_pred_vectors, yprun))
        return new_margins, new_pv
    
    @staticmethod
    def update_margins_dict(tree_preds, pred_vectors, remaining_trees, yprun):
        new_margins = {}
        new_pv = {}
        for considered_tree in tqdm(remaining_trees, desc = "Updating tree resiliency"):
            new_pred_vectors = copy.deepcopy(pred_vectors)
            new_margins[considered_tree] = []
            # Subtract the tree.
            for sample_id, sample in enumerate(tree_preds[considered_tree]):
                new_pred_vectors[sample_id][tree_preds[considered_tree][sample_id]] -= 1
            new_pv.update({considered_tree: new_pred_vectors})
            new_margins[considered_tree].extend(Pruner.per_sample_margin(new_pred_vectors, yprun))
        return new_margins, new_pv

    @staticmethod
    def evaluate_mean_dm(old_margins, new_margins):
        averages = []
        for new_margin_vector in new_margins:
            average = 0
            for old_margin_sample, new_margin_sample in zip(old_margins, new_margin_vector):
                average += (old_margin_sample - new_margin_sample)
            averages.append(average / len(old_margins))
        #averages = np.sort(averages)
        return averages
    
    @staticmethod
    def evaluate_mean_dm_dict(old_margins, new_margins):
        averages = {}
        for tree_id in new_margins.keys():
            new_margin_vector = new_margins[tree_id]
            average = 0
            for old_margin_sample, new_margin_sample in zip(old_margins, new_margin_vector):
                average += (old_margin_sample - new_margin_sample)
            averages[tree_id] = (average / len(old_margins))
        #averages = np.sort(averages)
        return averages
    
    @staticmethod
    def evaluate_min_dm(old_margins, new_margins):
        min_margins = []
        for new_margin_vector in new_margins:
            margin_decreases = []
            for old_margin_sample, new_margin_sample in zip(old_margins, new_margin_vector):
                margin_decreases.append((old_margin_sample - new_margin_sample))
            min_margins.append(np.min(margin_decreases))
        #min_margins = np.sort(min_margins)
        return min_margins
        
    methods = {
        "MeanD-M":  0,
        "MinD-M":   1
    }

    def __init__(self, classifier : Classifier, method : str = "MeanD-M", number_of_remaining_trees : int = 1):
        assert number_of_remaining_trees < len(classifier.trees)
        self.number_of_remaing_trees = number_of_remaining_trees
        self.classifier = classifier
        self.method = method
        self.logger = logging.getLogger("pyALS-RF")
        self.is_splitted = False
        self.actual_trees = [i for i in range(len(self.classifier.trees))]
        self.pruned_trees = []

    def prune_test_split(self,fraction: float = None):
        classifier = self.classifier
        indexes = np.arange(0, len(classifier.x_test))
        y_flat = self.classifier.y_test.ravel()
        if fraction == None :
            mop_size = compute_sample_size(test_set_size = len(classifier.x_test), error_margin = 0.05, confidence_level = 0.95, individual_prob = 0.5)
            portion = mop_size / len(self.classifier.x_test)
            self.x_prun, self.x_val, self.y_prun, self.y_val, self.prun_indexes, self.validation_indexes = train_test_split(self.classifier.x_test, y_flat, indexes, train_size = portion)       
        else:
            self.x_prun, self.x_val, self.y_prun, self.y_val, self.prun_indexes, self.validation_indexes = train_test_split(self.classifier.x_test, y_flat, indexes, train_size = fraction)       
        self.is_splitted = True
    
    def evaluate_accuracy_test_set(self):
        correctly_classified = 0
        pred_vectors = [[0 for x in self.classifier.model_classes] for sample in self.x_val]
        for tree_idx, tree_id in tqdm(enumerate(self.actual_trees), desc = "Evaluating the prediction vectors of the pruned classifier"):
            preds = self.classifier.trees[tree_id].visit_by_leaves(self.x_val)
            for p_id,p in enumerate(preds):
                pred_vectors[p_id][p] += 1
        
        for pv, y in zip(pred_vectors, self.y_val):
            if np.argmax(pv) == int(y) and not Classifier.check_draw(pv)[0]:
                correctly_classified += 1
        return correctly_classified
    
    def prune(self):
        if not self.is_splitted:
            assert 1 == 0, "You should prune your test set !"
        
        self.logger.info("Initializing accuracy on the test set !")
        self.baseline_correctly_classified = self.evaluate_accuracy_test_set()
        self.baseline_accuracy = (self.baseline_correctly_classified / len(self.y_val)) * 100.0
        
        self.logger.info(f"Baseline accuracy is {self.baseline_accuracy}")
        start = time.time()
        # Initialize the predictions per each tree and the prediction vector
        original_tree_preds, pred_v = Pruner.initialize_tree_ranking(self.classifier, Xprun=self.x_prun, yprun=self.y_prun)
        # Initialize the margins.
        current_margins = Pruner.per_sample_margin(pred_v, self.y_prun)
        while len(self.actual_trees) > self.number_of_remaing_trees:
            # Evaluate the possible margin update for each tree.
            margins_per_tree, new_prediction_vectors = Pruner.update_margins(original_tree_preds, pred_v, self.actual_trees, self.y_prun)
            # Evaluate the metric
            metrics = Pruner.evaluate_mean_dm(current_margins, margins_per_tree)
            # Take the minimum value of the metric and select the corresponding tree
            selected_tree_id = np.argsort(metrics)[0]
            selected_tree = self.actual_trees[selected_tree_id]
            # Remove the tree
            self.actual_trees.remove(selected_tree)
            # Prune the tree.
            self.pruned_trees.append(selected_tree)
            # Update the margins.
            pred_v = new_prediction_vectors[selected_tree]
            current_margins = margins_per_tree[selected_tree_id]
        end = time.time()
        delta = end - start
        
        self.logger.info(f"Heuristic ended in {delta}: seconds")
        self.logger.info(f"Evaluating the accuracy on the validation set")
        self.pruned_correctly_classified = self.evaluate_accuracy_test_set()        
        self.pruned_accuracy = ( self.pruned_correctly_classified / len(self.y_val) ) * 100.0
        self.loss = self.baseline_accuracy - self.pruned_accuracy
        self.logger.info(f"Pruned Accuracy on validation {self.pruned_accuracy} Loss: {self.loss}")
        self.pruning_conf = []
        
        # generate the pruning configuration.
        for tree in self.pruned_trees:
            for leaf in self.classifier.trees[tree].leaves:
                    self.pruning_conf.append((str(int(leaf["class"])), str(tree), str(leaf["sop"])))
        