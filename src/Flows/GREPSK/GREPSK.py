"""
Copyright 2021-2025 Antonio Emmanuele <antonio.emmanuele@unina.it> 
                    

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
from sklearn.model_selection import train_test_split
from ...plot import boxplot
from scipy.stats import norm # For cut-offs.
from sklearn.ensemble import RandomForestClassifier
import json5
import os
import joblib

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
        
        depth =     1    # higher the depth higher the cost
        activity =  2    # lower the frequency of activation higher the cost
        combined =  3    # both the previous, combined; thus, leaves with the same costs in terms of depth but with lower frequency of activations cost more!
        
        @staticmethod
        def crit_to_str(criterion):
            if criterion == 1:
                return "depth"
            elif criterion == 2:
                return "activity"
            elif criterion == 3:
                return "combined"
        
    def __init__(self, classifier : RandomForestClassifier, pruning_set_fraction : float = 0.5, ncpus : int = cpu_count()):
        self.classifier = classifier
        self.pruning_set_fraction = pruning_set_fraction
        self.ncpus = 0
        self.logger = logging.getLogger("pyALS-RF")
        self.pruning_configuration = []
        self.removed_boxes = 0
        # Mantain the set of leaves depths
        self.leaf_dephts =[tree.tree_.compute_node_depths() for tree in self.classifier.estimators_]
        # Mantains the set of leaves activations.
        self.node_counts =[ [ tree.tree_.n_node_samples[node] for node in range(0, tree.tree_.node_count)] for tree in self.classifier.estimators_]
        self.origina_node_cost = self.get_cost()
        # Initialize and node counter for the accellerator implementation.
        self.original_and_nodes = 0
        for tree_id in range(0,len(self.node_counts)):
            for node_id in range(self.classifier.estimators_[tree_id].tree_.node_count):
                if self.classifier.estimators_[tree_id].tree_.children_left[node_id] == -1 and self.classifier.estimators_[tree_id].tree_.children_right[node_id] == -1:
                    self.original_and_nodes += self.node_counts[tree_id][node_id] - 1 


        self.removed_and_nodes = 0

    def store_prunign_conf(self, pruning_conf_out):
        path_report = os.path.join(pruning_conf_out, "pruning_conf.json5")
        model_path = os.path.join(pruning_conf_out, "pruned_classifier.joblib")
        to_dump = []
        # Avoid errors with json5 dump (doesn't like arrays bruh)
        for j in range(0, len(pruning_conf_out) - 1):
            pass

        with open(path_report, "w") as f:
            json5.dump(to_dump, f, indent = 2)
        joblib.dump(self.classifier, model_path)

    def prune_leaf(self, tree_id, leaf_to_prune):
        children_left = self.classifier.estimators_[tree_id].tree_.children_left
        children_right = self.classifier.estimators_[tree_id].tree_.children_right

        # Find the parent node
        parent_node = None
        for i in range(self.classifier.estimators_[tree_id].tree_.node_count):
            if children_left[i] == leaf_to_prune or children_right[i] == leaf_to_prune:
                parent_node = i
                break

        if parent_node is None:
            print("Leaf node not found or already pruned.")
            print(f"Tree Id {tree_id} Leaf {leaf_to_prune}")
            assert 1 == 0
        
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
        old_value = self.classifier.estimators_[tree_id].tree_.value[parent_node]
        self.classifier.estimators_[tree_id].tree_.value[parent_node] = self.classifier.estimators_[tree_id].tree_.value[sibling]
        return parent_node, sibling, sibling_id, old_value
    
    def split_pruning(self, X, y):
        assert len(X) == len(y)
        indexes = np.arange(len(X))
        self.x_pruning, self.x_test, self.y_pruning, self.y_test, self.idx_prun, self.idx_test = train_test_split(X, y, indexes, train_size=self.pruning_set_fraction) # Use stratify = self.classifier.x_test.ravel() ensures that all classess are considered. 

    def restore_pruned_leaf(self, tree_id, parent_node, pruned_leaf, sibling, sibling_id, old_value):
        self.classifier.estimators_[tree_id].tree_.value[parent_node] = old_value
        # If the children was left
        if sibling_id == 0:
            self.classifier.estimators_[tree_id].tree_.children_left[parent_node] = pruned_leaf
            self.classifier.estimators_[tree_id].tree_.children_right[parent_node] = sibling
        elif sibling_id == 1:
            self.classifier.estimators_[tree_id].tree_.children_left[parent_node] = sibling
            self.classifier.estimators_[tree_id].tree_.children_right[parent_node] = pruned_leaf
        else:
            self.logger.error("Invalid sibling id !")
            assert 1 == 0
    
    def redundancy_boxplot(self, outfile):
        boxplot([ i for i in self.red_vec ], "", "Redundancy", outfile, figsize = (2, 4), annotate = False, integer_only= True)
            
    def get_cost(self):
        cost = 0 
        for tree in self.classifier.estimators_:
            cost += tree.tree_.node_count
        return cost        
    
    # Evaluates the EPi metric.
    @staticmethod
    def evaluate_EPI(predicted_class, considered_class, votes_vector):
        return np.ceil((votes_vector[predicted_class] - votes_vector[considered_class]) / 2)

    def evaluate_error_resiliency(self):
        self.logger.info("Initiating error resiliency evaluation")
        self.predicted_classes = []
        self.logger.info("Computing node depths")
        self.logger.info("Generating leaf per pruning sample")
        # Find the set of leaves for each tree Dim = [Num_trees, samples]
        sample_leaves = self.classifier.apply(self.x_pruning)
        self.logger.info("Initializing sample per leaf, leaf info and sample redundancy")
        self.x_pruning_correct = []
        self.x_pruning_correct_leaves = []
        self.red_vec = []
        # For each sample
        for sample_id, tree_leaves_per_sample in enumerate(sample_leaves):
            # Initialize the prediction vector, i.e. the set of votes per each class-
            pred_vector = [0 for cl in self.classifier.classes_]
            # For each tree.
            for tree_id, leaf in enumerate(tree_leaves_per_sample):
                # Construct the prediction vector by increasing of one vote the maximum class.
                pred_vector[np.argmax(self.classifier.estimators_[tree_id].tree_.value[leaf])] += 1
            # Take the maximum class.
            predicted_class = np.argmax(pred_vector)
            # If the class is correct then add the sample in pruning configuration.
            if predicted_class == self.y_pruning[sample_id]:
                self.x_pruning_correct_leaves.append(tree_leaves_per_sample)
                self.predicted_classes.append(predicted_class)
                preds_epi = [0 for cl in self.classifier.classes_]
                # For each class evaluate the EPI
                for c in range(self.classifier.n_classes_):
                    preds_epi[c] = GREPSK.evaluate_EPI(self.predicted_classes[-1], c, pred_vector)
                # # Save the EPI vectors 
                # self.sample_epis.append(preds_epi)
                # Take the minimum value and its index. 
                sorted_epi_indexes = np.argsort(preds_epi)
                # The 0 is always the sample itself.
                minimum_resiliency_class = sorted_epi_indexes[1]
                redundancy = preds_epi[minimum_resiliency_class]
                #self.redundancy_vector.append(redundancy)
                self.x_pruning_correct.append(self.x_pruning[sample_id])
                self.red_vec.append(redundancy)
        # Sort redundancy vector, list of used pruning samples, list of leaves (used during pruning) and the ensemble prediction per sample.
        self.red_vec = np.array(self.red_vec)
        sorted_idxs = np.argsort(self.red_vec)[::-1]
        self.red_vec = self.red_vec[sorted_idxs]
        self.x_pruning_correct = np.array(self.x_pruning_correct)
        self.x_pruning_correct = self.x_pruning_correct[sorted_idxs]
        self.x_pruning_correct_leaves = np.array(self.x_pruning_correct_leaves)
        self.x_pruning_correct_leaves = self.x_pruning_correct_leaves[sorted_idxs]
        self.predicted_classes = np.array(self.predicted_classes)
        self.predicted_classes = self.predicted_classes[sorted_idxs]


    def update_error_resiliency(self, pruned_tree, pruned_leaf, new_node_id):
        self.x_pruning_correct_leaves = self.classifier.apply(self.x_pruning_correct)
        
        for x_id, new_leaves_per_tree in enumerate(self.x_pruning_correct_leaves):
            # Update leaves.
            # Construct the prediction vector.
            pred_vector = [0 for c in self.classifier.classes_]
            pred_epis = [0 for c in self.classifier.classes_]
            for tree_id, l in enumerate(new_leaves_per_tree):
                pred_vector[np.argmax(self.classifier.estimators_[tree_id].tree_.value[l])] += 1
            # Update Epis.
            for c in self.classifier.classes_:
                pred_epis[c] = GREPSK.evaluate_EPI(self.predicted_classes[x_id], c, votes_vector = pred_vector)
            sorted_epis = np.sort(pred_epis)
            if sorted_epis[0] < 0:
                self.logger.debug("No longer classifying !")
                new_redundancy = sorted_epis[0]
            elif sorted_epis[0] == 0:
                new_redundancy = sorted_epis[1]
            else:
                self.logger.error("Unexpected condition in updating EPis, terminating")
                assert 1 == 0
            self.red_vec[x_id] = new_redundancy

        # End by reordering the vector.                
        sorted_idxs = np.argsort(self.red_vec)[::-1]
        self.red_vec = self.red_vec[sorted_idxs]
        self.x_pruning_correct = self.x_pruning_correct[sorted_idxs]
        self.x_pruning_correct_leaves = self.x_pruning_correct_leaves[sorted_idxs]
        self.predicted_classes = self.predicted_classes[sorted_idxs]
    
    def computed_episVector(self, predicted_class, considered_leaves):
        pred_vector = [0 for i in self.classifier.classes_]
        pred_epis = []
        for tree_id, l in enumerate(considered_leaves):
            pred_vector[np.argmax(self.classifier.estimators_[tree_id].tree_.value[l])] += 1
        for c_id in range(0, self.classifier.n_classes_):
            pred_epis.append(GREPSK.evaluate_EPI(predicted_class, c_id, pred_vector))
        return pred_epis
    
    @staticmethod
    def compute_sample_res_from_EpiVector(pred_epis):
        sorted_epis = np.sort(pred_epis)
        if sorted_epis[0] < 0:
            redundancy = sorted_epis[0]
        elif sorted_epis[0] == 0:
            redundancy = sorted_epis[1]
        else:
            assert 1 == 0
        return redundancy

    def get_best_leaf(self, leaves_per_samples, cost_criterion):
        self.logger.debug("Computing cost per each leaf")
        costs = []
        # Sort each leaf of the sample by cost.
        for tree_id, leaf in enumerate(leaves_per_samples):
            if cost_criterion == GREPSK.CostCriterion.depth:
                costs.append(self.leaf_dephts[tree_id][leaf])
            elif cost_criterion == GREPSK.CostCriterion.activity:
                costs.append( 1 / self.node_counts[tree_id][leaf])            
            elif cost_criterion == GREPSK.CostCriterion.combined:
                costs.append(self.leaf_dephts[tree_id][leaf] / self.node_counts[tree_id][leaf])
            else:
                print("Invalid cost")
                print(cost_criterion)
                print(GREPSK.CostCriterion.depth)
                exit(1)
        best_tree = np.argmax(costs)
        # Return the leaf of the tree with the highest cost.
        return best_tree, leaves_per_samples[best_tree]

    def sort_leaves_by_cost(self, cost_criterion : CostCriterion):
        self.logger.debug("Computing cost per each leaf")
        leaves_id = []
        costs = []
        # compute the cost of each leaf first, based on depth and activations
        for leaf, info in self.leaves_info.items():
            leaves_id.append(leaf)
            if cost_criterion == GREPSK.CostCriterion.depth:
                costs.append(info[0])
            elif cost_criterion == GREPSK.CostCriterion.activity:
                costs.append( 1 / info[1])            
            elif cost_criterion == GREPSK.CostCriterion.combined:
                # info["cost"] = info[0] / info[1] # leaves with the same costs in terms of literals but with less activity cost more!
                costs.append(info[0] / info[1])
            self.logger.debug(f"Cost of {leaf} is {info[0]}/{info[1]}")
        self.logger.debug("Sorting cost per leaf")
        sorted_args = np.argsort(costs)[::-1][len(costs)]
        leaves_id = leaves_id[sorted_args]
        costs = costs[sorted_args]
        return leaves_id, costs