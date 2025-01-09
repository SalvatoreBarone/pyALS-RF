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
from scipy.stats import norm # For cut-offs.
from sklearn.model_selection import train_test_split
import re
import time
from multiprocessing import cpu_count, Pool
from pyalslib import list_partitioning

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

class MrAxC:

    """ 
        Split MOP samples and validation samples.  
    """
    def sample_dse_samples(self):
        #def sample_dse_samples(self, per_class_subsampling = False):    
        classifier = self.classifier
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
                self.logger.error("[MR-AXC] ERROR: In sampling, each class in the test set should be considered.")
                exit(1)
    
    """ Compute the cost of each leaf as the number of nodes in that specific leaf. """
    def compute_leaves_costs(self):
        """ For each tree mantains the number of nodes involved in each class.  """
        self.cost_per_tree = []
        self.total_cost = 0
        for t in self.classifier.trees:
            cost_per_class = [0 for c in self.classifier.model_classes]
            for leaf in t.leaves:
                # Find the number of minterms.
                node_count = len(re.findall(r'Node_\d+', leaf["sop"]))
                # Increase the cost per each class.
                cost_per_class[int(leaf["class"])] += node_count
            # Append the cost of the single tree.
            self.cost_per_tree.append(cost_per_class)
            # Increase the total cost of the ensemble.
            self.total_cost += np.sum(cost_per_class)

    """ Transform the set of per_tree_classes (i.e. a vector where for each tree the set of classes is present)  
        into a vector where for each sample the vector of classes for each tree is considered.
    """
    def per_tree_classess_into_classes_per_tree(per_tree_classes):
        classes_per_tree = [[-1 for t in per_tree_classes] for sample in per_tree_classes[0]]
        for tree_id, tree_classes in enumerate(per_tree_classes):
            for sample_id, class_pred in enumerate(tree_classes):
                classes_per_tree[sample_id][tree_id] = class_pred
        return np.array(classes_per_tree)
    
    def initialize_tree_prediction_per_sample(self):
        self.logger.info("[MR-AXC] Initiating accuracy evaluation on X_MOP..")
        start = time.time()
        x_mop_leaves = self.classifier.get_leaf_index_ensemble(self.x_mop)
        _, self.x_mop_baseline_accuracy = self.classifier.get_accuracy_by_leaves_idx(x_mop_leaves, self.y_mop)
        end = time.time()
        x_mop_classes = self.classifier.transform_leaves_into_classess(x_mop_leaves)
        self.x_mop_classes = MrAxC.per_tree_classess_into_classes_per_tree(x_mop_classes)
        self.logger.info(f"[MR-AXC] Accuracy on X_MOP and Leaves initialized in ms {(end - start)* 1000}")
        self.logger.info(f"[MR-AXC] Accuracy on X_MOP :{self.x_mop_baseline_accuracy}")
        # If multicore evaluation function is used.
        if self.num_cores > 1:
            # The partitioning is ordered.
            self.p_xmop_classes = list_partitioning(self.x_mop_classes, self.num_cores)
            self.p_ymop = list_partitioning(self.y_mop, self.num_cores)
        self.logger.info("[MR-AXC] Initiating accuracy evaluation on X_VAL..")
        start = time.time()
        x_val_leaves = self.classifier.get_leaf_index_ensemble(self.x_val)
        _, self.x_val_baseline_accuracy = self.classifier.get_accuracy_by_leaves_idx(x_val_leaves, self.y_val)
        end = time.time()
        self.logger.info(f"[MR-AXC] Accuracy on X_VAL and Leaves initialized in ms {(end - start)* 1000}")
        self.logger.info(f"[MR-AXC] Accuracy on X_VAL :{self.x_val_baseline_accuracy}")
        
    """ Get the set of TMR vector predictions.
        given the set of classes per each tree (i.e. classes_per_tree) and the modular redundant configuration (i.e. class configuration)
        this function returns the output of a TMR structure ( a set of 0 or 1 for each class).
    """
    @staticmethod
    def get_mr_vectors(classes_per_tree, class_configurations):
        assert len(np.shape(classes_per_tree)) == 2, "Invalid input vector, provide per each tree the list of classes for input samples"
        num_tree_per_cfg = [sum(1 for tree in cfg if tree > 0) for cfg in class_configurations]
        thds = [int(np.ceil(num_trees/2)) for num_trees in num_tree_per_cfg]
        to_ret = []
        # For each inference
        for tree_votes in classes_per_tree:
            out_vector = []
            # For each class configuration
            for c_id, config in enumerate(class_configurations):
                # If there is at least one tree in the cfg.
                if num_tree_per_cfg[c_id] > 0 :
                    # Get the predictions of the trees in configuration. 
                    tree_preds = tree_votes[config]
                    voting_trees = np.sum(tree_preds == c_id)
                    # Append 0 or 1 depending on the final outcome
                    if voting_trees > thds[c_id]:
                        out_vector.append(1)
                    else:
                        out_vector.append(0)
                else: # If the configuration has no tree directly append 0
                    out_vector.append(0)
            # Append the configuration.
            to_ret.append(out_vector)
        # Return to_ret
        return np.array(to_ret)
    
    """ Given a tmr_vector predictions and an oracle y returns the accuracy considering the draw as a missclassification and the 
        one not considering a draw as misclassification.
    """
    @staticmethod
    def get_correctly_predicted_from_vectors(tmr_vectors, y):
        assert len(tmr_vectors) == len(y), "The number of TMR vectors should be equal to the number of different cfgs."
        correct_draw = 0
        correct_no_draw = 0
        for vector, correct_class in zip(tmr_vectors, y):
            # Get the number of active modular redundant structures
            active_modules = np.where(vector == 1)[0]
            nro_actives = len(active_modules)
            # If at least one cfg
            if nro_actives > 0:
                # Take always the first class.
                predicted_class = active_modules[0]
                if predicted_class == correct_class:
                    # If there is only one active module and the class is correct increase the size.
                    if nro_actives > 1 :
                        correct_no_draw += 1
                    else:
                        correct_no_draw += 1
                        correct_draw += 1
        # Return the accuracy considering the draw condition as a misclassification and the one with no missclassification.
        return correct_draw, correct_no_draw


    """ Given a tmr_vector predictions and an oracle y returns the accuracy considering the draw as a missclassification and the 
        one not considering a draw as misclassification.
    """
    @staticmethod
    def get_accuracy_from_vectors(tmr_vectors, y):
        assert len(tmr_vectors) == len(y), "The number of TMR vectors should be equal to the number of different cfgs."
        correct_draw, correct_no_draw = MrAxC.get_correctly_predicted_from_vectors(tmr_vectors, y)
        # Return the accuracy considering the draw condition as a misclassification and the one with no missclassification.
        return 100 * (correct_draw / len(y)), 100 * (correct_no_draw / len(y))

    @staticmethod
    def evaluate_mr_cfg_corr_class(per_tree_classes, y, cfg):
        pred_vectors = MrAxC.get_mr_vectors(per_tree_classes, cfg)
        return MrAxC.get_correctly_predicted_from_vectors(pred_vectors, y)
    
    """ Evaluates the accuracy of a configuration.
        per_tree_classes:   Vector where for each input sample, the set of votes (predicted classes), for each tree 
                            is inserted.
        y:                  Set of oracle predictions for each different tree.
        cfg:                CFG per classess ( i.e. for each different class it contains the set of trees voting for that class)
        Returns:            A tuple consisting on:
                                1-  Accuracy considering the draw condition as missclassifications.
                                2-  Accuracy considering not considering the draw conditions as missclassifications but
                                    with the first class considered.
    """
    @staticmethod
    def evaluate_mr_cfg_accuracy( per_tree_classes, y, cfg):
        pred_vectors = MrAxC.get_mr_vectors(per_tree_classes, cfg)
        return MrAxC.get_accuracy_from_vectors(pred_vectors, y)
    
    def __evaluate_xmop_single_core(self, mr_cfg):
        return MrAxC.evaluate_cfg_xmop(self.x_mop_classes, self.y_mop, mr_cfg)
    
    def __evaluate_xmop_multi_core(self, mr_cfg):
        args = [(x,y)for x, y in zip(self.p_xmop_classes, self.p_ymop)]
        corr_classified_draw_list, corr_classified_no_draw_list = self.pool.starmap(MrAxC.evaluate_mr_cfg_corr_class, args)
        return 100 * (np.sum(corr_classified_draw_list) / len(self.y_mop)), 100 * (np.sum(corr_classified_no_draw_list) /len(self.y_mop)) 
        

    """ Evaluate the accuracy on X_MOP. """
    def evaluate_mr_cfg_xmop(self, mr_cfg):
        return self.__xmop_priv_eval(mr_cfg)
    
    """ Evaluate the savings of the current cfg.
        The cost is computed as the actual cost minus the cost of the removed parts.        
    """
    def evaluate_mr_cfg_cost(self, new_cfg):
        current_cost = self.total_cost
        # For each tree, if the class is no longer classifier 
        for tree_id, tree_costs in enumerate(self.cost_per_tree):
            # If the tree no longer classifies a class then remove the actual cost
            for class_id, class_cfg in enumerate(new_cfg):
                if tree_id not in class_cfg:
                    current_cost -= tree_costs[class_id]
        return current_cost
    
    def __init__(self, classifier: Classifier, num_cores: int = 1):
        self.logger = logging.getLogger("pyALS-RF")
        self.logger.info("[MR-AXC] Initializing the module")
        self.classifier : Classifier = classifier   
        self.num_cores = num_cores
        # Select the correct function for the multicore evaluation.
        if self.num_cores > 1:
            self.__xmop_priv_eval = self.__evaluate_xmop_multi_core
        else:
            self.__xmop_priv_eval = self.__evaluate_xmop_single_core
            
        self.logger.info("[MR-AXC] Sampling classess..")
        self.sample_dse_samples()
        self.logger.info("[MR-AXC] Sampling completed.")
        self.logger.info(f"[MR-AXC] MOO-Samples: {(len(self.x_mop))}")
        self.logger.info(f"[MR-AXC] Validation-Samples: {(len(self.x_val))}")
        # Compute leaves costs
        self.logger.info(f"[MR-AXC] Initializing leaves costs.")
        self.compute_leaves_costs()
        self.logger.info(f"[MR-AXC] Computed leaves costs.")
        self.logger.info(f"[MR-AXC] Leaves costs: \r\n {self.cost_per_tree}")
        # Compute Predictions         
        self.logger.info(f"[MR-AXC] Initializing samples per leaf.")
        self.initialize_tree_prediction_per_sample()
    