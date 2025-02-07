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
from sklearn.model_selection import train_test_split
from ...plot import boxplot
from scipy.stats import norm # For cut-offs.
from sklearn.ensemble import RandomForestClassifier


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
    
    def split_pruning(self, X, y):
        indexes = np.arange(len(self.classifier.x_test))
        self.x_pruning, self.x_test, self.y_pruning, self.y_test, self.idx_prun, self.idx_test = train_test_split(X, y, train_size=self.pruning_set_fraction) # Use stratify = self.classifier.x_test.ravel() ensures that all classess are considered. 
    
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
    def evaluate_error_resiliency(self, predicted_classes):
        self.logger.info("Initiating error resiliency evaluation")
        predicted_classes = []
        self.redundancy_vector = [] # Sample Idx, Redundancy, EPI vectors.
        self.leaf_info = [ {} for t in self.classifier.estimators_]         # Node count and activations
        self.logger.info("Computing node depths")
        self.leaf_dephts =[tree.tree_.compute_node_depths for tree in self.classifier.estimators_]
        self.logger.info("Generating leaf per pruning sample")
        sample_leaves = self.classifier.apply(self.x_pruning)
        self.logger.info("Initializing sample per leaf, leaf info and sample redundancy")
    
        # For each sample
        for sample_id, tree_leaves_per_sample in enumerate(sample_leaves):
            # Initialize the prediction vector.
            pred_vector = [0 for cl in self.classifier.classes_]
            # For each tree.
            for tree_id, leaf in enumerate(tree_leaves_per_sample):
                # Construct the prediction vector by increasing of one vote the maximum class.
                pred_vector[np.argmax(self.classifier.estimators_[tree_id].tree_.value[leaf])] += 1
                # Update the leaf info 
                if leaf not in self.leaf_info:
                    self.leaf_info[tree_id].update({ leaf : (self.leaf_dephts[tree_id][leaf] , self.classifier.estimators_[tree_id].tree_.n_node_samples[leaf])})
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
        self.logger.info("Sample per leaf and ")
    
    def update_error_resiliency(self, pruned_leaf):
        # For each sample update a vector.
        pass 
    
    def update_leaf_info(self, tree_id, pruned_leaf):
        pass

    def store_pruning_conf(self, outfile : str):
        pass

    # TODO: FIX THIS FUNCTION
    def evaluate_accuracy(self):
        pass
        
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
      

    def trim(self, cost_criterion : CostCriterion):
        pass
        # logger = logging.getLogger("pyALS-RF")
        # logger.info(f"Test set: {len(self.classifier.x_test)} samples")
        # logger.info(f"Pruning set fraction: {self.pruning_set_fraction}")
        # self.split_test_dataset(self.pruning_set_fraction)
        # logger.info(f"Pruning set: {len(self.x_pruning)} samples")
        # logger.info(f"Validation set: {len(self.x_test)} samples")
        # self.p_tree = self.classifier.p_tree
        # self.args_evaluate_pruning = [[t, self.x_pruning] for t in self.p_tree]
        # self.args_evaluate_test = [[t, self.x_test] for t in self.p_tree]
        # self.pool = self.classifier.pool
        # self.baseline_accuracy = self.evaluate_accuracy()
        # logger.info(f"Baseline accuracy (on validation set) : {self.baseline_accuracy}%")
        # self.original_cost = self.get_cost()
        # logger.info(f"Original cost: {self.original_cost}")
        # self.accuracy = self.baseline_accuracy
        # self.loss = 0
        # logger.info("Performing Boolean networks backup")
        # self.backup_bns()
        # self.evaluate_redundancy()
        # self.sort_leaves_by_cost(cost_criterion)
        # self.pruning_configuration = []