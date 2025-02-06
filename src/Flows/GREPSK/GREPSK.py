"""
Copyright 2021-2025 Salvatore Barone <salvatore.barone@unina.it>
                    Antonio Emmanuele <antonio.emmanuele@unina.it> 

This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
import logging, numpy as np
from multiprocessing import cpu_count
from tabulate import tabulate
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from ...Model.Classifier import *
from ...Model.DecisionTree import *
from ...plot import boxplot
from scipy.stats import norm # For cut-offs.
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from ..GREP import GREP

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

class GREPSK:
    
    class CostCriterion:
        depth = 1,      # higher the depth higher the cost
        activity = 2,   # lower the frequency of activation higher the cost
        combined = 3    # both the previous, combined; thus, leaves with the same costs in terms of depth but with lower frequency of activations cost more!
        
    def __init__(self, classifier : RandomForestClassifier, pruning_set_fraction : float = 0.5, max_loss : float = 5.0, min_resiliency : int = 0, ncpus : int = cpu_count()):
        self.classifier = classifier
        self.pruning_set_fraction = pruning_set_fraction
        self.max_loss = max_loss
        self.min_resiliency = min_resiliency
        self.ncpus = min(ncpus, len(self.classifier.trees))
        self.logger = logging.getLogger("pyALS-RF")
        self.pruning_configuration = []
        self.removed_boxes = 0
        self.removed_ands = 0

    def prune_leaf(self, tree_id, leaf_to_prune):
        children_left = self.classifier.estimators_[tree_id].children_left
        children_right = self.classifier.estimators_[tree_id].tree_.children_right
        
        # Find the parent node
        parent_node = None
        for i in range(self.classifier.estimators_[tree_id].tree_.node_count):
            if children_left[i] == leaf_to_prune or children_right[i] == leaf_to_prune:
                parent_node = i
                break

        if parent_node is None:
            print("Leaf node not found or already pruned.")
            return
        
        sibling_id = 0
        # Ensure we are pruning a leaf
        if children_left[parent_node] == leaf_to_prune:
            sibling = children_right[parent_node]
            sibling_id = 0
        else:
            sibling = children_left[parent_node]
            sibling_id = 1

        # Turn parent into a leaf node
        children_left[parent_node] = -1
        children_right[parent_node] = -1
        return parent_node, sibling, sibling_id
    
    def get_train_test(self):
        pass
    
    def restore_pruned_leaf(self, tree_id, parent_node, pruned_leaf, sibling, sibling_id):
        # If the children was left
        if sibling_id == 0:
            self.classifier.estimators_[tree_id].children_left[parent_node] = pruned_leaf
            self.classifier.estimators_[tree_id].children_right[parent_node] = sibling
        elif sibling_id == 1:
            self.classifier.estimators_[tree_id].children_left[parent_node] = sibling
            self.classifier.estimators_[tree_id].children_right[parent_node] = pruned_leaf
        else:
            self.logger.error("Invalid sibling id !")
            assert 1 == 0
    
        
    def redundancy_boxplot(self, outfile):
        boxplot([ i[1] for i in self.redundancy_vector ], "", "Redundancy", outfile, figsize = (2, 4), annotate = False, integer_only= True)
            
    def get_cost(self):
        cost = 0 
        for tree in self.classifier.estimators_:
            cost += tree.tree_.node_count
        return cost        
    
    # Evaluates the EPi metric.
    @staticmethod
    def evaluate_EPI(predicted_class, considered_class, votes_vector):
        return np.ceil((votes_vector[predicted_class] - votes_vector[considered_class]) / 2)
    
    # For each sample mantains the leaf.
    def evaluate_error_resiliency(self, predicted_classes, sample_leaves):
        predicted_classes = []
        self.redundancy_vector = [] # Sample Idx, Redundancy, EPI vectors.
        self.logger.info("Initiating error resiliency evaluation")
        self.logger.info("Computing sample per leaf")

        # For each sample
        for sample_id, tree_leaves_per_sample in enumerate(sample_leaves):
            # Initialize the prediction vector.
            pred_vector = [0 for cl in self.classifier.classes_]
            # For each tree.
            for tree_id, leaf in enumerate(tree_leaves_per_sample):
                # Construct the prediction vector by increasing of one vote the maximum class.
                pred_vector[np.argmax(self.classifier.estimators_[tree_id].tree_.value[leaf])] += 1
            # Take the maximum class.
            predicted_classes.append(np.argmax(pred_vector))
            # If the class is correct then add the sample in pruning configuration.
            if predicted_classes[-1] == self.y_pruning[sample_id]:
                preds_epi = [0 for cl in self.classifier.classes_]
                # For each class evaluate the EPI
                for c in range(self.classifier.n_classes_):
                    preds_epi[c] = GREPSK.evaluate_EPI(predicted_classes[-1], c, predicted_classes)
                # Save the EPI vectors 
                self.sample_epis.append(preds_epi)
                # Take the minimum value and its index. 
                sorted_epi_indexes = np.argsort(preds_epi)
                # The 0 is always the sample itself.
                minimum_resiliency_class = sorted_epi_indexes[1]
                redundancy = preds_epi[minimum_resiliency_class]
                self.redundancy_vector.append((self.y_pruning_idxs[sample_id], redundancy, preds_epi))
        
    
    def store_pruning_conf(self, outfile : str):
        pass

    def split_test_dataset(self, mode, pruning_set_fraction : float = 0.5):
        self.logger("This function called here has no effect !")
        pass

    # TODO: FIX THIS FUNCTION
    def evaluate_accuracy(self):
        pass
        
    def sort_leaves_by_cost(self, cost_criterion : CostCriterion):
        logger = logging.getLogger("pyALS-RF")
        # compute the cost of each leaf first, based on depth and activations
        for leaf, info in self.leaves_info.items():
            literals = len(leaf[2].split("and"))
            activations = len(info["samples"])
            if cost_criterion == GREP.CostCriterion.depth:
                info["cost"] = literals
            elif cost_criterion == GREP.CostCriterion.activity:
                info["cost"] = 1 / activations
            elif cost_criterion == GREP.CostCriterion.combined:
                info["cost"] = literals / activations # leaves with the same costs in terms of literals but with less activity cost more!
            logger.debug(f"Cost of {leaf} is {literals}/{activations}={info['cost']}")
        # now, for each of the activing sample, sort the list of leaves based on their cost
        for info in self.samples_info.values():
            leaves_and_their_cost = [ (leaf, self.leaves_info[leaf]["cost"]) for leaf in info["leaves"] ]
            logger.debug(f"Sorting leaves\n{leaves_and_their_cost}")
            leaves_and_their_cost.sort(key=lambda x: x[1], reverse = True)
            info["leaves"] = [ l[0] for l in leaves_and_their_cost]
            logger.debug(f"Sorted leaves\n{info['leaves']}")

      
    
    """ Given in input a classifier and a set of leaves indexes to prune, this function 
        returns pruning configuration. 
        pruned_leaves_idx_per_tree is a dictionary ( or a tree indexed list), containing
        for each tree the pruned leaves for each class.
     """
    @staticmethod
    def get_pruning_cfg_from_leaves_idx(classifier, pruned_leaves_idx_per_tree):
        pruning_cfg = []
        # For each tree.
        for tree_id, tree in enumerate(classifier.trees):
            pruned_leaves_per_class = pruned_leaves_idx_per_tree[tree_id]
            tree_pruning_cfg = []
            # For each class
            for considered_class, pruned_leaves in pruned_leaves_per_class.items():
                # For each pruned leaf per class.
                for pruned_leaf in pruned_leaves:
                    tree_pruning_cfg.append((str(considered_class), str(tree_id), tree.leaves[pruned_leaf]["sop"]))
            pruning_cfg.extend(tree_pruning_cfg)
        return pruning_cfg

    def trim(self, cost_criterion : CostCriterion):
        logger = logging.getLogger("pyALS-RF")
        logger.info(f"Test set: {len(self.classifier.x_test)} samples")
        logger.info(f"Pruning set fraction: {self.pruning_set_fraction}")
        self.split_test_dataset(self.pruning_set_fraction)
        logger.info(f"Pruning set: {len(self.x_pruning)} samples")
        logger.info(f"Validation set: {len(self.x_validation)} samples")
        self.p_tree = self.classifier.p_tree
        self.args_evaluate_pruning = [[t, self.x_pruning] for t in self.p_tree]
        self.args_evaluate_validation = [[t, self.x_validation] for t in self.p_tree]
        self.pool = self.classifier.pool
        self.baseline_accuracy = self.evaluate_accuracy()
        logger.info(f"Baseline accuracy (on validation set) : {self.baseline_accuracy}%")
        self.original_cost = self.get_cost()
        logger.info(f"Original cost: {self.original_cost}")
        self.accuracy = self.baseline_accuracy
        self.loss = 0
        logger.info("Performing Boolean networks backup")
        self.backup_bns()
        self.evaluate_redundancy()
        self.sort_leaves_by_cost(cost_criterion)
        self.pruning_configuration = []